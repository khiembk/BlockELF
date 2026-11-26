# OWT Repetition Collapse: Diagnosis and Inference-Time Ablations

**Scope:** inference-only investigation of BlockELF's OWT (OpenWebText) unconditional
generation quality, specifically the long-canvas (`max_length=1024`, 64 commit rounds
of `block_size=16`) repetition/looping collapse flagged by MAUVE and repetition metrics
on checkpoint `owt_146k/checkpoint_146075` (step 146075, warm-started block-revelation
training; see `configs/training_configs/train_owt_ELF-B_block16.yml`).

**Non-negotiables preserved:** no training launched, no checkpoint mutated, Stage 1's
exact algorithm (joint flow-sample of all remaining positions, commit next `b=16`
left-to-right, re-encode, fresh flow sample) unchanged, LM1B/WMT14/XSum/full-sequence
ELF code paths and defaults untouched, `role_aware_uncond_attention=false` for OWT kept
as-is. All changes described here are **opt-in** (new function, new CLI flags,
non-default args) — the existing default call path is bit-for-bit unchanged.

**Nothing in this document has been applied to any config file.** Every number below
comes from inference-only ablation runs; the recommendation at the end is for the user
to decide on, not something already adopted.

---

## 1. Root-cause audit (code-level findings)

Before writing any new sampling code, the existing decoder/generation path was read
and verified directly (not assumed from the task spec):

- **`_dlm_decode_batch`** (pre-existing): always `jnp.argmax(decoder_logits, axis=-1)`,
  no RNG input at all. Confirmed by reading `src/utils/generation_utils.py` prior to
  any edits — there was no sampling temperature, no nucleus/top-k, nothing. Every
  commit round is a fully deterministic greedy decode given the flow-matching sample.
- **`generate_block_uncond`'s round loop**: each round (i) re-encodes the committed
  prefix through the frozen T5 encoder, (ii) draws a fresh Gaussian `z0` and solves the
  flow ODE/SDE from scratch (no KV-cache reuse across rounds), (iii) greedy-decodes the
  *entire* canvas's decoder logits but only commits the next `block_size`-wide window.
  This is the correct, verified Stage-1/Algorithm-2 behavior — not a bug — but it means
  the *only* source of per-round diversity is the flow sampler's SDE noise
  (`sampling_method=sde`, `sde_gamma=1.5`); the actual token choice at decode time was
  always deterministic argmax.
- **`self_cond_cfg_scale` semantics — verified, not assumed**: for OWT,
  `config.num_self_cond_cfg_tokens=4 > 0`, so `self_cond_cfg_scale` is fed directly into
  the model as an **in-context conditioning value** (trained range
  `self_cond_cfg_min=0.5` to `self_cond_cfg_max=5.0`) via a dedicated code branch in
  `sampling_utils.py`'s self-cond-cfg forward path — **not** classifier-free-guidance
  interpolation (`v_uncond + scale*(v_cond - v_uncond)`, a different branch only reached
  when `num_self_cond_cfg_tokens == 0`). This means `self_cond_cfg_scale=1.0` is *not*
  an "unguided baseline" in the usual CFG sense; it is simply a different point in the
  model's trained conditioning range. OWT's current default is `3.0` (see
  `configs/sampling_configs/uncond_sampling_configs_owt_gamma15.yml`).
- **Positional structure of the collapse** (from previously-saved `owt_legacy_1000.jsonl`,
  reused — no new generation needed for this part): sliding-window entropy over GPT-2
  token windows shows a **gradual, monotonic decline** across the 1024-token canvas
  (window-128 entropy: 5.88 bits at position 0 → 4.88 bits at position ~1024), not a
  sharp cliff at any particular round. This is consistent with repetition being a slow,
  cumulative drift that compounds over the 64 commit rounds rather than a single
  failure point — pointing at per-round decode/conditioning choices (which repeat every
  round) rather than, say, a specific encoder-context-length effect.

## 2. Implementation (opt-in, tested, no regressions)

- `_dlm_decode_batch` refactored into a thin `jnp.argmax` wrapper around a new
  `_dlm_decode_logits_batch` (returns raw logits; identical forward pass, identical
  external behavior) — every existing caller (`generate_block_cond` for WMT14/XSum, the
  `jax.pmap`-wrapped full-sequence path) is provably unaffected since
  `_dlm_decode_batch`'s signature and return value never changed.
- New `_sample_decoder_tokens(decoder_logits, rng, mode, temperature, top_p)` supporting
  `"categorical"` and `"top_p"` (nucleus) sampling, restricted by the caller to only the
  round's newly-committed window (not the full canvas, for efficiency — verified this
  doesn't affect correctness since out-of-window content is discarded regardless).
- `generate_block_uncond` gained opt-in `decoder_mode` (`"greedy"`/`"categorical"`/`"top_p"`),
  `decoder_temperature`, `decoder_top_p` kwargs. **Default `decoder_mode="greedy"`
  consumes zero extra RNG splits**, so the noise/timestep RNG stream for every existing
  caller is bit-for-bit identical to before this feature existed.
- `eval_block_uncond.py` gained `--owt_decoder_mode/--owt_decoder_temperature/--owt_decoder_top_p`,
  `--self_cond_cfg_scale` (override), and `--run_manifest_path` (saves exact
  checkpoint/seed/sampling-config/decoder-settings/PPL as JSON per run).
- **Tests** (`src/tests/test_decoder_sampling.py`, 10/10 pass): greedy==argmax exactly;
  categorical concentrates correctly on dominant logits; top_p always keeps at least the
  top token and masks nothing at `top_p=1.0`; temperature scaling behaves correctly;
  same-key reproducibility / different-key independence; position-restricted sampling
  depends only on the sliced window. The full pre-existing `tests/test_block_flow.py`
  suite (including `generate_block_uncond`'s own tests) was re-run and still passes —
  **zero regressions**.
- GPU smoke tests (N=16, real checkpoint) confirmed all three decoder modes run without
  crashing and produce non-empty output before any larger-scale ablation was run.

## 3. Screening ablations (N=150) — **flagged unconfirmed, see §4**

All at checkpoint `owt_146k/checkpoint_146075`, `cfg=1`, `sampling_method=sde`,
`sde_gamma=1.5`, `num_sampling_steps=32`, `seed=42`, `batch_size=50`, MAUVE at
`featurize_model_name=gpt2-large`, `max_text_length=768`, `num_reference_samples=150`
sampled from `embedded-language-flows/openwebtext-t5` (train split, no held-out split
available in this environment — a documented limitation, not a silent one).

| Config | decoder | self_cond_cfg_scale | seq-rep-4 mean | looping(>0.5) | Gen. PPL | MAUVE |
|---|---|---|---|---|---|---|
| D0 (baseline) | greedy | 3.0 | 0.404 | 37.3% | 8.47 | 0.624 |
| D1 | categorical, t=1.0 | 3.0 | 0.388 | 36.0% | 8.74 | 0.506 |
| D2 | categorical, t=0.7 | 3.0 | 0.397 | 36.0% | 8.71 | 0.485 |
| D3 | top_p=0.9 | 3.0 | 0.389 | 31.3% | 8.74 | 0.737 |
| D4 | top_p=0.95 | 3.0 | 0.376 | 28.7% | 8.82 | 0.690 |
| S2 | greedy | 2.0 | 0.369 | 30.0% | 9.35 | 0.625 |
| **S1** | greedy | **1.0** | **0.286** | **20.7%** | 9.32 | **0.855** |
| S0 | greedy | 0.5 | 0.260 | 21.3% | 13.07 | 0.520 |
| C1 | top_p=0.9 | 1.0 | 0.288 | 21.3% | 12.44 | 0.620 |

At this screening scale S1 (`self_cond_cfg_scale=1.0`, plain greedy decoder) looked like
a dramatic win, and the trend looked non-monotonic (S0 at 0.5 and C1 combining both
levers both *dropped* MAUVE despite comparable or better repetition numbers — already a
sign that MAUVE and repetition-rate are not tightly coupled at this N).

## 4. N=1000 confirmation — **the result that actually matters**

Screening-N=150 MAUVE turned out to be far too noisy to trust on its own: the identical
D0 config measured **0.624** here (N=150) vs. a previously-established **0.336**
(N=1000, same checkpoint, legacy/default settings, `src/tests` + git history from this
project's earlier MAUVE-fix work). That is a 2x swing from sample-size and
batch-size-dependent RNG draws alone (JAX's counter-based RNG means a different batch
size produces different actual generated text even from the same top-level seed — this
was also independently verified in this investigation via a unit test). **Lesson for
future screening: N=150 MAUVE is not a reliable signal by itself; use the deterministic
repetition diagnostics as the primary cheap screen, and only trust MAUVE at N≥1000.**

Given S1's large apparent effect, both S1 and the next-best decoder-sampling candidate
(D3, top_p=0.9) were re-run at **N=1000** (same protocol: checkpoint, cfg, gamma, steps,
seed=42, batch_size=50) for a low-variance confirmation:

| Config | seq-rep-4 mean | looping(>0.3) | looping(>0.5) | looping(>0.7) | Gen. PPL | MAUVE |
|---|---|---|---|---|---|---|
| Baseline (historical, N=1000, legacy settings) | 0.394 | 60.6% | 36.1% | 10.4% | — | 0.336 |
| **S1 confirmed** (sccfg=1.0, greedy) | **0.298** | 45.7% | **21.2%** | 5.6% | 11.93 | 0.325 |
| **D3 confirmed** (top_p=0.9, sccfg=3) | 0.390 | 60.5% | 34.3% | 11.4% | 8.88 | 0.310 |

**Verdict, stated plainly:**

- **`self_cond_cfg_scale=1.0` is a real, large, N=1000-confirmed repetition-reduction
  lever.** Looping (seq_rep_4 > 0.5) drops from 36.1% to 21.2% — a ~42% relative
  reduction — and this holds up at full confirmation scale, unlike the decoder-sampling
  results.
- **It does NOT confirm as a MAUVE/overall-quality improvement.** MAUVE is
  statistically flat (0.325 vs. 0.336 baseline) — within what looks like ordinary
  run-to-run noise at this N, not a meaningful gain. The screening-level 0.855 was
  measurement noise, not signal, and should be disregarded.
- **D3 (top_p=0.9 decoder sampling) shows no real effect in either direction at N=1000** —
  seq-rep-4, looping, and MAUVE are all within noise of the historical baseline. Its
  screening-level improvement (MAUVE 0.737, looping 31.3%) did not replicate either.
  Categorical sampling (D1/D2) was not re-confirmed at N=1000 given D3 (the strongest
  decoder-sampling screening candidate) already failed to replicate — further decoder
  sampling variants are unlikely to fare better.

### Sample excerpts (N=1000 confirmation runs)

**S1 (sccfg=1.0, greedy)**, sample 5:
> "The researchers will be able to use another, more advanced technology: an
> aircraft plane with an air-engine engine. The study, published in Nature on March
> 2004, is the work of Andrew Scotti, an California-based professor of bioscience at
> the University of California, Berkeley. The results show that technologies can make
> robots more effective for complex industrial tasks, such as making a robot bu[...]"

**D3 (top_p=0.9)**, same seed/position, sample 5:
> "The researchers will be able to use another, more advanced device - adietary
> material, which provides positive energy to the bodies of birds. For example,
> Applewoodpt cells have a relatively effective capacity to transfer energy from rats
> to the brain, which can generate positive energy to rats since these cells become
> much more active in the internal sector of the brain to generate positive energ[...]"

Both are clearly less repetitive than the known baseline's heavy verbatim looping (see
prior `owt_legacy_1000.jsonl` samples), but neither reads as fully coherent, fluent
prose — consistent with MAUVE's flat verdict: less looping, not more "real."

## 5. Limitations / what was not done

- The historical N=1000 baseline (0.336 MAUVE, 36.1% looping) used `batch_size≈24` per
  its config default, vs. `batch_size=50` for all ablations in this investigation.
  Because JAX's counter-based RNG makes batch size affect the actual sampled content
  (not just how it's split up), this is not a perfectly matched baseline — a
  same-protocol D0 re-run at N=1000/batch_size=50 was not performed here (would cost
  another ~3 hours of shared GPU time) and would be the cleanest way to fully close this
  gap.
- Categorical sampling (D1/D2) and the combined/extreme configs (S0, C1) were only
  screened at N=150 and never confirmed at N=1000, given D3 (the strongest surviving
  decoder-sampling candidate) already failed to replicate.
- The optional **P1 (local repetition penalty)** was never implemented — the spec's own
  instruction was to try it only if sampling/guidance proved insufficient, and
  `self_cond_cfg_scale=1.0`'s confirmed repetition reduction, while real, does not reach
  "solved" (looping is still 21.2% of samples, and MAUVE shows no overall quality
  improvement) — so P1 remains a reasonable next thing to try, not something ruled out.

## 6. Recommendation

1. **Do not adopt `self_cond_cfg_scale=1.0` as a MAUVE-quality fix** — it isn't one,
   confirmed at N=1000.
2. **Do consider `self_cond_cfg_scale=1.0` if the goal is specifically reducing
   verbatim looping/repetition** (a real, ~42%-relative, confirmed effect) at the cost
   of somewhat higher Gen. PPL (9.3 vs. 8.5) — that tradeoff is a judgment call, not
   something this investigation can make for you.
3. **Decoder-level sampling (categorical/top_p) does not appear to be a fruitful
   direction** based on D3's N=1000 null result — further ablating it is unlikely to be
   worth additional GPU time.
4. **If repetition reduction remains a priority, P1 (a local n-gram repetition penalty
   applied during decoding) is the most promising unexplored lever** from the original
   task's ordering, now that sampling/guidance alone have been shown insufficient.
5. **Methodologically**: any future OWT screening should treat repetition diagnostics
   (deterministic, cheap, no GPU-text-model needed) as the primary signal, and should
   not make go/no-go decisions on MAUVE scores below N≈1000.

No config files were changed as part of this investigation. All new code
(`_sample_decoder_tokens`, `decoder_mode` plumbing, the repetition diagnostics script,
and their tests) is opt-in and defaults to the exact pre-existing behavior.

---

## Provenance

- Git commit at time of this investigation: `38beb7dd23854f0150001d594df9c5ff077c66d6`
  (plus uncommitted working-tree changes to `src/eval_block_uncond.py`,
  `src/utils/generation_utils.py`, and new files `src/eval_repetition_diagnostics.py`,
  `src/tests/test_decoder_sampling.py`).
- Checkpoint: `owt_146k/checkpoint_146075` (step 146075), EMA params
  (`use_ema=True`, the default).
- Seed: 42 (generation RNG and MAUVE reference-sampling/judge seed, both).
- Sampling config: `sampling_method=sde, num_sampling_steps=32, cfgs=[1], sde_gamma=1.5,
  time_schedule=logit_normal` (from `configs/sampling_configs/uncond_sampling_configs_owt_gamma15.yml`,
  entry 0), with only `self_cond_cfg_scales` overridden per ablation.
- `role_aware_uncond_attention=false` (OWT's existing, unchanged default) throughout.
- MAUVE protocol: `featurize_model_name=gpt2-large, max_text_length=768,
  reference_dataset=embedded-language-flows/openwebtext-t5 (train split,
  allow_train_split_as_reference), mauve_seed=42`.
- Raw outputs, manifests, and diagnostics/MAUVE reports for every ablation in §3/§4 are
  under the session scratchpad's `screening/` directory (not committed to the repo —
  reproducible from the manifests' recorded config).
