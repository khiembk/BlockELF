# OWT Stage-1 training fix: balanced revelation rounds (T1) — implementation

Follow-up to `docs/owt_collapse_root_cause_findings.md` (root-cause: block-revelation
fine-tuning teaches the model to over-rely on committed context, which only breaks down
when that context is self-generated rather than real). This document implements Phase 0
(reproducibility harness) and Phase 1 (T1: `block_round_zero_mix`) of the external
review's fix-it spec, and **does not launch any training** — per that spec's own
non-negotiable ("Do not launch a long training job or overwrite existing checkpoints"),
the actual T0/T1a/T1b pilot runs require your explicit go-ahead.

**Non-negotiables preserved:** no architecture/checkpoint-format change, `rho=0.0`
(the default for every existing config) reproduces the exact original code path
bit-for-bit, `committed_mask`/`active_mask`/`boundary`/source handling/flow/decoder
losses untouched, LM1B/WMT14/XSum unaffected (confirmed by test).

---

## Correction to the earlier root-cause document

The external review correctly caught a real error in `docs/owt_collapse_root_cause_
findings.md`'s §1.3: that analysis derived `w_j` (expected per-token training weight by
target-block index) assuming `reduce_token_loss` normalizes **per example**. It does not
— `train_step.py::reduce_token_loss` computes `(safe_loss * loss_mask).sum() /
jnp.maximum(loss_mask.sum(), 1.0)` with **no axis argument**, pooling the entire
per-device minibatch's active-token count into one shared scalar denominator. The
claimed `K*H_K ≈ 303.6x` late-vs-early ratio is the per-example-ISOLATED idealization,
not the real training objective.

**Corrected analysis** (real wiring, verified directly): `global_batch_size=24`, 2 local
devices, `grad_accum_steps` implemented via `optax.MultiSteps` (which delays the
optimizer *apply*, not a further batch split) → each `train_step` call's `reduce_token_
loss` pools exactly **12** examples. Real OWT T5-token `sequence_length` (sampled
directly from `embedded-language-flows/openwebtext-t5`, 50,000 rows): mean 956, median
987, so `K_i = ceil(len/16)` is heavily concentrated near 56-64 (mean K_i≈60.2, median
62). A Monte Carlo simulation of the actual batch-pooled objective (12 real-length-drawn
examples per step, 300,000 simulated steps, both with and without
`compute_block_round_counts`'s K_i-proportional example reweighting — which barely
changes anything for OWT specifically, since its K_i is already tightly concentrated)
gives:

| | w_j[last] / w_j[first] |
|---|---|
| Flawed per-example-isolated analytic claim | 303.6x |
| **Corrected batch-pooled simulation (rho=0)** | **~70x** |

The qualitative conclusion is unchanged and still real — late target blocks dominate the
training signal by roughly two orders of magnitude — but the earlier document
overstated the exact multiplier by ~4.3x. `src/tests/test_block_round_weighting.py` has
been annotated with this correction in place; the corrected, batch-pooled simulation is
what `src/tests/test_block_round_zero_mix.py`'s
`test_batch_pooled_weighting_imbalance_shrinks_with_rho` actually tests going forward.

---

## Phase 0: reproducibility harness

Added to `train.py` (`_log_reproducibility_header`, called at the very top of
`run_training`, before any data/model loading):
- Git SHA + dirty status (full `git status --porcelain` diff list), read-only,
  never raises (degrades to a logged warning if git is unavailable).
- The full effective config (as loaded + any `--config_override`s applied), YAML-dumped.
- PRNG seed, resume path, warm-start checkpoint path.

Added at the checkpoint-loading site (after resume/warm-start logic resolves): one
consolidated log line reporting `starting_step`, the exact source of params/EMA
(`resume:<path>` / `warm_start:<path>` / `random_init`), and whether `opt_state` is
resumed or fresh (warm-start never carries `opt_state` — confirmed by reading the
existing warm-start code path, which only copies `params`/`ema_params1`).

**Earlier checkpoint for a prevention experiment**: the training run's own periodic-eval
history (`docs/owt_collapse_root_cause_findings.md` §1.5's PPL/entropy table) shows
monotonically improving Gen. PPL from step 10001 through 146067 with no earlier
"obviously non-degenerate by repetition" checkpoint identified yet — repetition
diagnostics were never computed for any OWT block checkpoint besides 146075 during this
investigation. **This is a real gap**: before running a T1 prevention pilot, the earlier checkpoint
candidates already saved on disk (`elf_b-owt-block16`'s output directory has saved
checkpoints spanning roughly steps 138k-158k, per the periodic-eval log) would need
repetition diagnostics computed (cheap,
pure-text, reusing `eval_repetition_diagnostics.py`) to identify a genuinely earlier,
less-specialized starting point, OR a prevention pilot could simply restart from the
pretrained (non-block) warm-start checkpoint, matching how the original production run
itself started. Not yet done — flagged here as the next concrete step before any T0/T1a/
T1b pilot, not assumed.

---

## Phase 1: `block_round_zero_mix` (T1)

### Implementation

`utils/block_utils.py::sample_block_round` gained an opt-in `rho: float = 0.0` parameter
implementing the mixture `q_rho(k|K_i) = (1-rho)/K_i + rho*1[k=0]`:

```python
if rho == 0.0:
    u = jax.random.uniform(rng, cond_len.shape)
    k = jnp.floor(u * num_rounds)
else:
    rng_u, rng_mix = jax.random.split(rng)
    u = jax.random.uniform(rng_u, cond_len.shape)
    k_uniform = jnp.floor(u * num_rounds)
    force_zero = jax.random.bernoulli(rng_mix, rho, cond_len.shape)
    k = jnp.where(force_zero, jnp.zeros_like(k_uniform), k_uniform)
```

`rho == 0.0` takes the **exact original code path** (same `u = jax.random.uniform(rng,
...)` call, same PRNG consumption) — not an equivalent-but-differently-coded path, the
literal original lines. `configs/config.py` gained `block_round_zero_mix: float = 0.0`;
`train_step.py` reads it via `getattr(config, "block_round_zero_mix", 0.0)` (so even
test fixtures / older config objects that don't define the field at all still default
cleanly to historical behavior) and passes it as `rho` to `sample_block_round`. No
changes to `committed_mask`/`active_mask`/`boundary` construction, encoder masking, or
any loss computation — `rho` only shifts which `k` gets sampled before those are built
from it.

Added metrics (logged whenever `use_block_revelation and not use_staggered_noise`,
regardless of `rho`, so `rho=0` runs get visibility into the historical `k`-distribution
too): `frac_k0`, `mean_k`, and a coarse 3-bin `k`-histogram (`frac_k_1_to_7`,
`frac_k_8_to_31`, `frac_k_32_plus`).

### Tests (`src/tests/test_block_round_zero_mix.py`, 6/6 pass)

1. `test_rho_zero_is_bit_exact_regression` — `rho=0.0` byte-identical to pre-change
   `sample_block_round`, 20 random fixture shapes.
2. `test_rho_one_always_selects_k0` — `rho=1.0` forces `k=0` for every example, `K_i` in
   {1, 2, 8, 64}.
3. `test_empirical_p_k0_matches_closed_form_q_rho` — empirical `P(k=0)` matches
   `(1-rho)/K_i + rho` within Monte Carlo noise, `K_i` in {1,2,8,64}, `rho` in
   {0, 0.1, 0.25, 0.5}.
4. `test_mask_invariants_hold_for_rho_greater_than_zero` — disjoint committed/active, no
   padding leak, cond fully committed, active non-empty iff target content remains,
   random `rho` in [0,1], 200 trials.
5. `test_batch_pooled_weighting_imbalance_shrinks_with_rho` — the REAL (batch-pooled,
   not per-example) simulation confirms `rho=0.25` measurably shrinks the late-vs-early
   weighting ratio relative to `rho=0` (quick in-test version: ~69x → ~4x, consistent
   with the larger offline simulation above).
6. `test_train_step_unaffected_without_opting_in` — `train_step` end-to-end, both
   `use_block_revelation=False` (LM1B/WMT14/XSum-style) and `=True`, is unaffected by
   this change when `block_round_zero_mix` is left unset/default.

Full pre-existing suite (`test_block_flow.py`, `test_round_zero_equivalence.py`,
`test_decoder_sampling.py`) re-run and still passes — zero regressions.

### T1 rho sweep — simulated effect size (NOT a trained result)

Using the corrected batch-pooled simulation, with the real OWT length distribution and
`compute_block_round_counts` reweighting:

| rho | P(k=0) (empirical, matches theory) | w_j[last]/w_j[first] |
|---|---|---|
| 0.0 (current default) | 0.0156 | ~70x |
| 0.10 (T1a) | 0.1147 | ~9.3x |
| 0.25 (T1b) | 0.2620 | ~4.0x |

This is a **simulation of the training-signal distribution only** — it says nothing yet
about whether rebalancing this signal actually improves generation quality/repetition.
That requires actually training T0/T1a/T1b pilots and running this investigation's full
diagnostic suite (repetition metrics first, MAUVE second, both at matched N≥1000
protocol with ≥3 clustering seeds) on the resulting checkpoints — not done here.

---

## Experiment table (scaffold — pilots NOT yet run)

| Run | Checkpoint source | Seed | rho | Steps planned | Status |
|---|---|---|---|---|---|
| T0 | TBD (see "earlier checkpoint" gap above) | 42 | 0.0 | short pilot | **not run** |
| T1a | same as T0 | 42 | 0.10 | short pilot | **not run** |
| T1b | same as T0 | 42 | 0.25 | short pilot | **not run** |

Device config, optimizer/EMA policy, and MAUVE protocol would match this
investigation's established conventions exactly (same checkpoint/seed/batch-size
matching across conditions, N=150-200 screening only, N≥1000 with ≥3 MAUVE clustering
seeds for confirmation, `gpt2-large`/`max_text_length=768` fixed, saved-text reuse for
MAUVE recomputation) — see `docs/owt_collapse_root_cause_findings.md`'s own established
protocol, which this would reuse unchanged.

## What's NOT done and why

- **No training launched.** Implementing T1 and verifying it with unit tests is
  complete; actually running T0/T1a/T1b pilot checkpoints, and evaluating them, is a
  separate, resource-committing step explicitly excluded from this pass pending your
  go-ahead.
- **Phase 2 (prefix-corruption robustness, T2) and Phase 3 (training-time legacy
  attention, T3) are intentionally not implemented** — both are explicitly scoped as
  conditional ("only if T1 does not sufficiently improve...") in the review spec, so
  implementing them before T1 has even been trained would be premature.
- **The "earlier non-degenerate checkpoint" identification is incomplete** (see Phase 0
  section above) — this is the most concrete next step before any pilot can actually be
  launched.
