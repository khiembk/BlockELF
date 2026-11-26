# BlockELF OWT: Root-Cause Investigation and Targeted Degeneration Fixes

Follow-up to `docs/owt_repetition_diagnostics.md`, which found that `self_cond_cfg_scale=1.0`
substantially reduces repetition (N=1000-confirmed: looping>0.5 36.1%→21.2%) but did NOT
improve MAUVE (0.336 baseline vs 0.325, statistically flat), and that the screening-level
decoder-sampling "wins" (e.g. top_p=0.9's MAUVE=0.737 at N=150) did not replicate at N=1000
(D3 confirmed at MAUVE=0.310, essentially unchanged from baseline). This document picks up
from there with a deeper root-cause investigation per the follow-up spec.

**Status: in progress.** §1 (code-level audits), P0-C (oracle-prefix intervention), and
P0-B (per-round deterioration tracking) are complete, and together already point to a
single, convergent, decisive root cause. P0-A/P0-D were deliberately NOT yet run — the
user asked to prioritize P0-C and then P0-B first given their diagnostic value, and the
remaining GPU-bound experiments are paused pending a check-in given their cost (P0-A's
matched N=1000 rerun alone is a ~3hr commitment on a GPU1 node that is currently also
running the user's own separate training job).

**Non-negotiables preserved throughout:** no training launched, no checkpoint mutated,
Stage 1's exact algorithm unchanged, LM1B/WMT14/XSum/full-sequence ELF untouched,
`role_aware_uncond_attention=false` kept as OWT's default. All new code is additive/opt-in
or confined to standalone analysis scripts (not committed to the repo as production code).

---

## §1. Code-level audits

### §1.1/§1.2 — Round-zero equivalence: no bug found

Instrumented both the block-revelation path's actual round-0 internals (encoding an
all-pad `committed_target_ids` through `build_committed_context` + `encode_text`, then
`_generate_samples_single_batch` + `_dlm_decode_batch`, exactly as `generate_block_uncond`
does for OWT's `role_aware_uncond_attention=False` config) and the full-sequence path's
equivalent call (`cond_seq=None`/`cond_seq_mask=None`), replaying IDENTICAL z0/timesteps/
rng/params through both (not just matching top-level seeds — JAX's counter-based RNG
means that alone would be insufficient). Result: **exact numerical equivalence** (max
`|z_final_block - z_final_full|` = 0.0, decoded token IDs identical). See
`src/tests/test_round_zero_equivalence.py` (2 tests, both pass).

A real quirk was found and confirmed harmless along the way: at round 0,
`build_committed_context`'s `is_valid` argument is `is_committed` (all-False), so the
T5-encoder attention mask it constructs is *also* all-False — not "block everything" but
a uniformly-shifted additive bias that, because softmax is shift-invariant, is numerically
identical to an all-ones (fully unmasked) mask. This looks alarming on first read but is
provably harmless: `cond_seq_mask` (`clean_mask`) is *also* all-False at round 0, so
`restore_cond`'s `jnp.where` never actually selects `cond_seq`'s value regardless of what
the degenerate encoder attention computed.

### §1.3 — Training-objective positional-weight imbalance: confirmed real, large, and
in the OPPOSITE direction from "the model doesn't learn late positions"

`sample_block_round` draws `k ~ Uniform{0,...,K-1}`; `reduce_token_loss` averages the
per-token loss uniformly over that round's active region. The expected per-token loss
weight for a token in target block `j` (0-indexed), marginalizing over the `K` equally-
likely choices of `k`, is:

```
w_j = (1/(K*b)) * sum_{k=0}^{j} 1/(K-k),   j = 0, ..., K-1
```

Verified both analytically (closed-form matches a direct double-sum to machine precision)
and empirically (20,000-trial Monte Carlo simulation of the real `sample_block_round`
matches the closed-form `(j+1)/K` per-block active-frequency prediction within 0.006). At
K=64: `w_last/w_first = K*H_K ≈ 303.6` — confirmed exactly. See
`src/tests/test_block_round_weighting.py` (3 tests, all pass).

**This means LATE target blocks get ~300x more expected training signal than EARLY
ones** — the opposite of what "early blocks get little weight" might suggest at first
read, but consistent with it: block `j` is only ever "active" (supervised) in rounds
`k<=j`, so an early block is almost never still-active by the time a late `k` is sampled
(it's already been teacher-forced/committed by then), while a late block is active for
almost every possible `k`. `train.py`'s `compute_block_round_counts`/`sample_weights`
correction is orthogonal to this (it fixes which EXAMPLE gets drawn, not this within-
sequence positional imbalance) and does not touch it.

**Why this matters**: heavy training weight on late blocks means the model learns a
strong "continue the pattern established by the (always-real, during training) committed
context" prior specifically for long-context continuations — see P0-C below for direct
confirmation that this prior fails catastrophically once the committed context is
self-generated rather than real.

### §1.4 — Role-aware vs. legacy attention masking: reconfirmed, mechanism hypothesized

Reconfirmed via code audit (`modules/model.py`'s `role_clean_mask` branch: "clean" queries
— committed/source positions AND the model's own special prefix tokens — are restricted
to attending only to "clean" keys; active queries attend to everything) and via the
existing 200-sample A/B artifacts (recomputed with the current diagnostics script):
**legacy** seq-rep-4 mean=0.370, looping>0.5=32.0%; **role-aware** seq-rep-4 mean=0.624,
looping>0.5=74.5% — role-aware attention is clearly worse for OWT, despite being MORE
faithful to how `train_step.py` actually masks attention during training.

**Hypothesized unifying mechanism** (not yet directly tested — P0-C tests exposure bias
generally, not this specific masking-mode comparison): legacy's "leaky" full attention
lets active-region computation blend with freshly-evolving noise each round, diluting the
model's tendency to just copy the established committed-context pattern. Role-aware
masking, being more faithful to the (per §1.3) late-block-heavy training distribution,
pushes the model to lean on its "continue the committed pattern" prior MORE strongly —
which is fine when the committed context is real, but actively harmful once that context
is self-generated and already drifting (exactly the compounding effect P0-C measures
directly below). This explains why role-aware, despite being "more correct" relative to
training, is empirically worse specifically on OWT's long-canvas, many-round regime.

### §1.5 — Checkpoint selection: confirmed blind spot (read-only, no changes applied)

`train.py::_is_plausible` gates checkpoint selection on `mean_token_entropy > 3.0` only;
`_maybe_save_best` then ranks by lowest Gen. PPL among plausible checkpoints. The full
PPL/entropy history for `elf_b-owt-block16` (32-step `sampling_configs[0]`, the column
that actually drives selection) shows checkpoint step 146067 (persisted as
`checkpoint_146075`) is the genuine PPL-minimum across the entire training run (ppl=8.496,
entropy=4.407) — so the selection criterion was followed correctly given its own logic.
**The problem is the criterion itself**: this exact checkpoint, independently measured in
this investigation at N=1000, has looping(>0.5)=36.1% of samples despite entropy=4.4-4.5,
comfortably clearing the 3.0 floor. A global mean-entropy average cannot see that ~36% of
samples are heavily looping while the rest are fine — the signal is diluted by averaging.
No evidence was found that a *less*-PPL-optimal-but-*less*-repetitive checkpoint was
snubbed (no repetition diagnostics exist for other checkpoints to check), but the
structural blind spot is real and directly demonstrated with this investigation's own
data.

**Side-finding** (not requested, noted for completeness): `generation.py`'s
`measured_nfe_total` instrumentation (used in training logs, not for checkpoint
selection) undercounts true NFE, because the counting wrapper sits inside
`jax.jit`-ted functions with `model_apply_fn` as a static arg — the Python-level counter
only increments on retracing (new shapes), not on every logical call once an executable
is cached. Effective NFE for this investigation's experiments is computed analytically
(rounds × steps) instead of trusting this field; see the provenance section of each
experiment below.

---

## §2 P0-C: Oracle-prefix intervention — exposure bias directly confirmed

**Setup**: at `k in {0, 8, 16, 32, 48, 56}` (committed-prefix length `k*16` tokens),
compared the model's NEXT generated block (one flow solve, greedy decode, `cfg=1`,
`sde_gamma=1.5`, 32 steps — matched sampling settings throughout) under two prefix
sources: **oracle** (first `k*16` tokens of a REAL OWT document) vs. **self-generated**
(whatever the model itself committed by round `k` in an actual from-scratch generation,
captured via one combined run up to round 56). N=150 per condition per k, checkpoint
`owt_146k/checkpoint_146075`, seed=42. Measured ONLY the new block (never whole-sequence
BLEU/MAUVE, per the spec): conditional GPT-2-large judge NLL of the new block given its
preceding text, 4-gram overlap of the new block with the recent (last 256 token) prefix,
and mean decoder-logit entropy over the new block's positions.

| k | self-gen overlap | oracle overlap | self-gen NLL | oracle NLL | self-gen entropy (bits) | oracle entropy (bits) |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 4.425 | 4.548 | 0.031 | 0.039 |
| 8 | 0.122 | 0.033 | 3.753 | 4.597 | 0.041 | 0.051 |
| 16 | 0.231 | 0.041 | 3.246 | 4.362 | 0.017 | 0.062 |
| 32 | 0.444 | 0.048 | 2.331 | 4.160 | 0.015 | 0.043 |
| 48 | 0.526 | 0.070 | 2.108 | 4.208 | 0.011 | 0.064 |
| 56 | 0.582 | 0.058 | 1.947 | 4.165 | 0.008 | 0.057 |

(n=150 at every cell except self-gen rows where a handful of already-EOS-finished
examples were excluded — never silently: exclusion counts are in the raw
`p0c_results.json` alongside every number above.)

**Verdict: exposure bias is directly and cleanly confirmed.** Given a REAL prefix of ANY
length, the model's next-block behavior is stable — overlap stays in a narrow 0.03-0.07
band and NLL stays in a narrow 4.16-4.60 band regardless of `k`. Given its OWN
self-generated prefix, overlap climbs monotonically and dramatically (0.00 → 0.58) and the
model becomes progressively MORE overconfident (decoder entropy keeps falling, 0.031 →
0.008 bits) as the self-generated context lengthens. The self-generated condition's
"improving" NLL (4.43 → 1.95) is not genuine quality — repeated text is trivially
predictable for ANY language model, GPT-2-large included — it is the exact same
"PPL/NLL fooled by repetition" artifact already documented at the checkpoint-selection
level (§1.5) and the MAUVE level (`owt_repetition_diagnostics.md`), now shown to apply to
a CONDITIONAL, per-block NLL too, not just whole-sequence PPL.

A representative excerpt at k=56 makes the mechanism vivid. Self-generated prefix (tail):
> "...ellite with a range of 1,04 miles (230 km) wide. It is planet-like with a range of
> 1,04 miles (230 km) wide. It is a small-range planet-like satellite"

New self-generated block: *"with an depth of 1,04 miles (333.6 km). It has"* — continuing
the exact same numeric/templated loop. Oracle prefix (unrelated real text, tail):
> "...izumi said. He said they hope to eventually offer an Arabic minor. Arizumi said
> parents should know that teaching Arabic does not equal teaching Islam"

New oracle-conditioned block: *"principles. "It doesn't matter that a student doesn't"* —
a normal, sensible continuation, no repetition, despite the prefix being just as long
(56 committed blocks = 896 tokens) as the degenerate self-generated case.

**This connects directly to §1.3 and §1.4's hypotheses**: the model handles long REAL
contexts of any length just fine (oracle condition, flat across all k) — it is
specifically self-generated context that triggers escalating, compounding repetition.
This is the textbook exposure-bias signature: training only ever sees real committed
prefixes (and, per §1.3, is especially heavily weighted toward learning this
"continue the established pattern" behavior for long/late contexts); inference's
self-generated prefixes drift from the real-text distribution the moment the first
imperfection appears, and the model's strong (over-)reliance on committed-context
pattern-continuation — rather than correcting course — amplifies that drift into the
repetition loops this entire investigation has been chasing.

---

## §2b P0-B: Per-revelation-round deterioration tracking

**Setup**: one tracked 64-round self-generation (N=150, same matched baseline settings as
P0-C: checkpoint `owt_146k/checkpoint_146075`, greedy decode, `self_cond_cfg_scale=3`,
`cfg=1`, `sde_gamma=1.5`, 32 steps, `role_aware_uncond_attention=False`, seed=42),
recording at EVERY round k=0..63: the newly-committed block's decoder max-softmax
probability / token entropy / top-p=0.9 candidate-set size (full N=150), and (a subset of
20 examples, per the spec's "limit overhead" instruction) the newly-committed block's
latent `z_final` standard deviation and cosine similarity against the immediately-
preceding committed block's latent.

**Finding 1 — decoder-level confidence is flat across all 64 rounds, NOT predictive of
collapse.** Mean max-softmax stays in [0.986, 0.998] and mean entropy stays in
[0.007, 0.06] bits at literally every round from k=0 to k=63 — no monotonic degradation,
no visible trend distinguishing "about to loop" from "not about to loop." The greedy
decoder is always extremely confident, whether or not what it's confident ABOUT happens
to be a repetition. **This rules out decoder-logit confidence as a usable collapse-
detection signal** (relevant to any future retry-on-degenerate-block intervention, §3A/§3B
of the original spec — confidence thresholds on the decoder itself would not fire).

**Finding 2 — latent cosine-similarity-to-previous-block shows a real, moderate upward
drift.** From ~0.17-0.18 (k=1-14) rising with noise to a ~0.25-0.31 plateau by k=30-63 —
roughly a 60-70% relative increase. This is a direct, position-resolved signature that
consecutive committed blocks' INTERNAL representations become progressively more
redundant/similar over rounds — repetition visible at the latent level, not just the
decoded-token level, and consistent with (though more modest in magnitude than) the
token-level overlap trend P0-C already found.

**Finding 3 — EOS events cluster almost entirely at the final round** (k=63: 66/150
rows; every other round: 0/150). Combined with this checkpoint's known length
distribution (mean ~960-980 tokens, a meaningful fraction near the 1024 canvas limit),
this is at least partly explained by many documents' natural length simply falling in
round 63's window — but it is also consistent with (not proof of) the model's
established "continue the pattern" bias making it reluctant to ever choose to STOP once
block-revelation generation is underway, instead of ending early.

**Finding 4 — the decisive triangulation.** Computed the same windowed-entropy/seq-rep
diagnostics on two reference curves using NO new GPU generation (real OWT text needs no
model at all; the pretrained full-sequence ELF-B-OWT generations were already cached
from earlier in this investigation):

| Source | seq-rep-4 mean | looping(>0.5) | window-128 entropy: start → end |
|---|---|---|---|
| **Self-gen (block-revelation, this checkpoint)** | 0.372 | 30.0% | 5.88 → **4.34 bits** (collapses) |
| Real OWT text | 0.034 | 0.0% | 6.17 → 6.16 bits (flat) |
| Pretrained ELF-B-OWT (full-sequence, single-shot, same base weights lineage) | 0.060 | 0.0% | 5.92 → 5.84 bits (flat) |

**Both real text and the pretrained full-sequence model are flat — only the block-
revelation self-generation curve collapses.** This decisively rules out two competing
hypotheses: (a) that the entropy decline is an inherent property of long OWT text or a
generic judge-model artifact (real text is flat), and (b) that it reflects fundamental
checkpoint/weight damage from fine-tuning (the pretrained model, generating the ORIGINAL,
non-block way, shows no such collapse at all). The entropy collapse is **specifically and
only** a product of the block-revelation ROUND-BY-ROUND GENERATION PROCEDURE — exactly
the exposure-bias mechanism P0-C already identified, now independently corroborated from
a completely different angle (reference-curve triangulation rather than oracle-prefix
substitution).

A representative excerpt (same self-generated sample, full 1024-token canvas) shows the
mechanism end to end:

- **0-128**: "A San Diego police officer in Texas issued a search warrant on Tuesday for
  what was a handgun inside the property. ... San Diego police officers are searching for
  someone who keeps his handgun inside an apartment wall" — repetitive in *content* but
  still grammatically coherent, still varied phrasing.
- **384-512**: "...Johnson Texas San Diego Police Officer released 32 Johnson Texas San
  Diego Police Officer released 32 Johnson Johnson Texas San Diego Police Officer..." —
  clearly drifting into template-level repetition, losing grammatical coherence.
- **768-896 / 896-1024**: "...Dave Johnson FOR Texas SanDia Police Officer Dave Johnson
  FOR Texas SanDia Police Officer Officer Dave Johnson..." — fully collapsed into a
  near-pure permutation loop of the same 6-7 tokens, the canonical "stuck" failure mode.

## §2c P0-A: Matched D0-vs-S1 confirmation (batch_size=50, N=1000)

The one remaining gap from the first investigation (`owt_repetition_diagnostics.md`) was
that the historical D0 baseline (MAUVE=0.336) used `batch_size≈24`, while S1's N=1000
confirmation used `batch_size=50` — not a perfectly matched comparison given JAX's
counter-based RNG makes batch size affect the actual sampled content. D0 was rerun here at
the identical `batch_size=50`/seed=42/N=1000 protocol:

| Config | seq-rep-4 mean | looping(>0.5) | MAUVE (seed=42) |
|---|---|---|---|
| D0 (historical, batch_size=24) | 0.394 | 36.1% | 0.336 |
| **D0 (matched rerun, batch_size=50)** | **0.383** | **33.8%** | **0.348** |
| S1 (sccfg=1.0, batch_size=50) | 0.298 | 21.2% | 0.325 |

The matched D0 rerun lands close to the historical number (confirming the batch-size
effect was real but modest, not a confound that was masking anything larger), and now
directly, cleanly confirms S1's repetition improvement with zero batch-size confound:
looping(>0.5) 33.8%→21.2%, a real drop.

**MAUVE's own measurement noise, now quantified**: recomputing MAUVE on S1's SAME saved
N=1000 generations with 3 different MAUVE clustering seeds (42/123/777, no regeneration)
gave 0.325 / 0.427 / 0.338 — a range of over 0.10 even at N=1000. Given this, D0's 0.348
vs. S1's 0.325 (seed=42) is well within measurement noise and cannot be called a real
difference either direction. **Sanity ceiling**: human-vs-human MAUVE (two disjoint
1000-document real-OWT samples, same protocol) = **0.9612** — confirms the protocol
correctly reports near-perfect similarity for real-vs-real text, so the ~0.3-0.4 scores
for every generated condition reflect a genuine, large quality gap from real text, even
though the exact value isn't precise to better than ~±0.05-0.1 at N=1000.

## §2d P0-D: 2×2 checkpoint × inference-mode matrix — the decisive result

Cells A (pretrained weights, full-sequence) and D (fine-tuned weights, block inference)
were already covered by existing cached data. The two NEW cells generated here (N=150
each, same SDE settings throughout: cfg=1, sde_gamma=1.5, 32 steps, self_cond_cfg_scale=3):

| | **Pretrained weights** (checkpoint_0, step 95085, never trained with block revelation) | **Fine-tuned weights** (checkpoint_146075) |
|---|---|---|
| **Full-sequence inference** | A: seq-rep-4≈0.060, 0% looping, entropy flat (known) | **C: seq-rep-4=0.063, 0% looping, entropy flat (5.97→5.90 bits) — NEW** |
| **Block inference** | **B: seq-rep-4=0.031, 0% looping, entropy flat (6.03→6.02 bits) — NEW** | D: seq-rep-4≈0.38-0.39, 30-36% looping, entropy collapses (5.88→4.34 bits) (known) |

**Only cell D collapses.** This is the single most decisive result in the whole
investigation: neither factor alone reproduces the problem.
- Cell B proves **block inference itself is not inherently broken** — a model that never
  saw a single block-revelation training step generates perfectly healthy, non-repetitive
  text when run through the exact same round-by-round committed-context procedure that
  causes checkpoint_146075 to collapse.
- Cell C proves **the fine-tuned weights are not damaged** — the same weights that
  collapse under block inference produce perfectly healthy, non-repetitive text when run
  the original (single-shot, full-sequence) way.

The collapse is therefore specifically a property of the INTERACTION between what
block-revelation fine-tuning teaches the model and the self-conditioned inference
procedure it's then asked to run — not either one alone.

## §3. Synthesis: a single, convergent root cause

Five independent lines of evidence now converge: §1.3's analytic training-weight
imbalance (late blocks trained ~300x more than early ones — this IS what block-revelation
fine-tuning specifically teaches, that cells A/B/C don't have), §1.4's role-aware-vs-legacy
masking result (more training-faithful attention is WORSE, consistent with the training
distribution over-encouraging context-copying), P0-C's oracle-prefix substitution (real
prefixes of any length are fine; self-generated ones compound into repetition), P0-B's
reference-curve triangulation, and P0-D's 2×2 matrix (only the fine-tuned-weights ×
block-inference combination collapses — directly confirming it is the FINE-TUNING that
installs the problematic over-reliance on committed context, since the pretrained model,
lacking that fine-tuning, does not collapse even under identical block inference) — **all
point to the same single root cause: block-revelation fine-tuning teaches the model to
rely heavily on committed context (per §1.3's confirmed late-round-heavy training
weighting), and this learned behavior is only ever exercised in training against REAL
committed prefixes — at inference, once the committed context is self-generated and
starts to drift, that same learned reliance pattern-matches onto and amplifies its own
drift into repetition.** This is a training-distribution-mismatch (exposure bias) issue
specifically introduced BY the fine-tuning process, not a pre-existing architectural
limitation, not inherent to block inference, and not a sign of damaged weights.

## §4. Training experiments (§4 of the spec): NOT started, approval required

T1 (round-zero anchoring / early-round curriculum — directly motivated by §1.3's
confirmed w_j imbalance) is the most evidence-supported candidate given P0-C/P0-B's
results, but no training has been run or proposed as code; this is a recommendation for
discussion, not an implementation, per the spec's explicit requirement.

## §5. Concise recommendation

**(i) Inference/conditioning mismatch, introduced BY training — not (ii) pre-existing
checkpoint/weight damage, not (iii) an evaluation artifact alone, and not inherent to
block inference as a mechanism.** The P0-D matrix is what makes this precise: it is not
"block inference is hard" (cell B is healthy) and not "the weights are broken" (cell C is
healthy) — it is specifically that **block-revelation fine-tuning teaches the model a
committed-context-reliance behavior that only breaks down when the committed context is
self-generated rather than real** (cell D, and P0-C/P0-B's direct confirmation of the
mechanism). The evaluation-artifact theme is a persistent secondary confound throughout
(Gen.PPL/NLL at every level — whole-sequence, per-checkpoint, per-block — is repeatedly
fooled into looking "good" by repetition itself), which is why repetition-rate metrics,
not PPL/NLL/MAUVE alone, must be the primary signal for any future work here.

## §6. Does this need retraining? Direct answer: yes — but a continued fine-tune, not a
from-scratch retrain

**Short answer: some additional training is very likely necessary. Inference-only fixes
have been tried and ruled insufficient; the root cause lives in what the block-revelation
fine-tuning stage taught the model, not in anything a sampling/decoding/guidance knob at
inference time can override.**

Why inference-only fixes won't get there, concretely:
- Decoder sampling (categorical/top_p) and `self_cond_cfg_scale` changes were both tried
  (`owt_repetition_diagnostics.md`) and confirmed at N=1000 to either do nothing
  (top_p=0.9: MAUVE 0.310 vs 0.336/0.348 baseline, repetition unchanged) or help
  repetition without helping MAUVE (`self_cond_cfg_scale=1.0`: repetition improves,
  MAUVE flat within noise). Neither touches the mechanism P0-C/P0-D identified: an
  internal reliance the model LEARNED during fine-tuning, exercised at inference
  regardless of how the final token is sampled or how strongly self-conditioning is
  weighted.
- The failure is structural, not a tunable parameter: cell B proves the SAME inference
  procedure is harmless on a model that was never taught this reliance. The fix has to
  change what gets taught, not how the already-taught model is sampled from.

What retraining would actually need to look like — and why it's cheap relative to "full
retrain":
- **This does not require retraining from scratch.** The existing warm-start lineage
  (pretrained ELF-B-OWT → block-revelation fine-tune → `checkpoint_146075`) can
  continue from `checkpoint_146075` itself (or an earlier, less-specialized point on that
  same run) with a MODIFIED round-sampling curriculum, not a new architecture, new data
  pipeline, or new loss function.
- **T1 (round-zero/early-round curriculum)** is the most directly motivated by this
  investigation's own evidence: §1.3 proved analytically and empirically that late target
  blocks currently receive ~300x more expected training signal than early ones (because
  `k ~ Uniform{0,...,K-1}` means an early block is almost never still "active" by the time
  a late `k` is sampled). Oversampling `k=0` (and/or early `k` more generally) — while
  keeping Stage 1's exact all-remaining-token prediction and fixed left-to-right
  commitment unchanged, per the spec's non-negotiables — directly rebalances this,
  giving the model more training signal on exactly the under-weighted regime (predicting
  well without an already-long committed context to lean on). This is a config-level
  change to the round-sampling distribution (`sample_block_round`'s `k ~ Uniform(...)`
  → `P(k=0)=p0` then uniform over the rest), not a new training recipe — low
  implementation risk, directly traceable to confirmed evidence, and it preserves
  everything the spec requires to stay unchanged.
- **T2 (exposure-robust/scheduled-sampling-style training)** would more DIRECTLY attack
  the exposure-bias mechanism P0-C measured (training occasionally on self-generated,
  not just real, committed prefixes) but is higher-risk to implement correctly (the spec
  itself warns naive scheduled sampling against the original suffix can teach
  contradictory targets) and should only be attempted if T1 alone proves insufficient.
- **T3 (SERF-style distillation)** is the least evidence-targeted option here — it's a
  general flow-quality regularizer, not aimed at this specific exposure-bias mechanism,
  and unproven on this block-revelation architecture. Not recommended as the first thing
  to try.

**Recommended path, in order**: (1) implement T1 as an opt-in round-sampling-curriculum
flag (`p0` controlling `P(k=0)`, default unchanged so every existing config/checkpoint is
unaffected), (2) continue fine-tuning from `checkpoint_146075` for a modest number of
additional steps with `p0` set to e.g. 0.1-0.25 (per the spec's own suggested sweep),
(3) re-run this investigation's exact diagnostic suite (repetition metrics first, MAUVE
second, both at matched N≥1000 protocol) on the resulting checkpoint to confirm the fix
actually closes the gap rather than just shifting it. **This requires your explicit
approval before any training is launched**, per the non-negotiables governing this
entire investigation.
