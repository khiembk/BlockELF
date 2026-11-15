# BlockELF — Conditional Generation Fixes (WMT14 De→En and XSum Only)

> **Task for Claude Code:** Inspect the CURRENT repository before editing, confirm the issues below against the actual code and checkpoint/configuration, implement minimal conditional-only fixes, and run regression tests. **Unconditional generation on LM1B and OWT already works well: preserve it exactly.**
>
> Repository: https://github.com/khiembk/BlockELF
> Scope: Stage 1, fixed-block, source-conditioned generation only. Do not implement Stage 2, GRPO, new flow objectives, or distillation.

## A. Non-negotiable requirements

1. **Do not change unconditional behavior** for LM1B/OWT. In particular, do not change the algorithm, initial noise, sampling trajectory, attention semantics, tokenization, default sampling arguments, decoder, RNG usage, or evaluation in `generate_block_uncond`, `test_generation_uncond`, and the unconditional evaluation path. If a shared function must be extended, use **optional arguments defaulting to the original behavior** and enable the new code **only** in `generate_block_cond` (or an explicit conditional-block branch). Verify no output changes for seeded unconditional regression tests.
2. Keep existing **ELF continuous-embedding flow model and pretrained/checkpoint compatibility**. Do not replace it with categorical flow matching, masked diffusion, or a new architecture. Avoid parameter-shape changes and preserve existing checkpoints.
3. Preserve the Stage 1 algorithm: **at every round predict all remaining target positions** using a full flow solve, then permanently commit **only the next `b` target positions** in left-to-right order. Do **not** replace this with current-block-only denoising or predict loss on only the next `b` positions.
4. Keep WMT14 and XSum training objective unchanged unless a demonstrated conditional-specific bug requires a minimal patch. Do not modify already-working LM1B/OWT training.
5. Keep original non-block ELF inference and all unconditional paths as separately runnable baselines. No reference target tokens/lengths may enter conditional inference.
6. Make claims only after **showing code evidence and test results**. Separate confirmed defects from possible distribution shifts/quality hypotheses. Report exactly which commands were run and their outputs.

## B. Start with a short code-path audit (before patching)

Inspect these current locations (the repository may have evolved; resolve the real signatures before editing):

- `src/train_step.py`: where fixed-block `clean_mask`, `role_clean_mask`, `denoiser_attention_mask` and label dropping are built and passed to `state.apply_fn`.
- `src/utils/generation_utils.py`: `generate_block_cond`, `_generate_samples_single_batch`, `_sample_step_for_scan`, `_dlm_decode_batch`, target packing, target extraction, EOS stopping.
- `src/utils/sampling_utils.py`: `_forward_sample_self_cond`, `_forward_sample`, `_ode_step`, `_sde_step`, `restore_cond`, and CFG.
- `src/modules/model.py`: how `attention_mask` and `role_clean_mask` affect *all* transformer layers and special/prefix tokens.
- `src/utils/block_utils.py`: `sample_block_round`, `build_committed_context`.
- `src/utils/data_utils.py`, `src/generation.py`, `src/eval_block.py` if present, `src/configs/config.py`, and the actual WMT14/XSum YAMLs.

First present a short train/infer call graph for conditional block generation, with tensor shapes and current attention/CFG arguments. Save the base git commit hash and the exact configs/checkpoint identifiers used in testing.

## C. P0 FIX #1 — Conditional inference must use training-equivalent denoiser attention

### Evidence to verify

In the reviewed `main` branch, `train_step.py` passes `attention_mask=denoiser_attention_mask` and `role_clean_mask=role_clean_mask` to the denoiser during fixed-block training. In contrast, `generate_block_cond` invokes `_generate_samples_single_batch(...)` using only `cond_seq` and `cond_seq_mask`. `_forward_sample_self_cond`'s `model_apply_fn(...)` calls do not pass either denoiser mask. Thus conditional block inference appears to use a different attention pattern from fixed-block training.

### Desired attention contract

Within one conditional generation round, let:

- `S` = actual source positions;
- `C` = previously generated, committed target positions;
- `A` = all currently unrevealed target positions in the *configured valid target canvas*;
- `P` = invalid/padding positions.

For transformer query/key pairs **in the denoiser**:

| Query → Key | S ∪ C | A | P |
| --- | ---: | ---: | ---: |
| S ∪ C | allow | **deny** | deny |
| A | allow | **allow all A** | deny |
| P | irrelevant/deny | irrelevant/deny | deny |

Important: Source and committed representations must not acquire information from noisy remaining positions, including indirectly through time/mode/prefix tokens over multiple layers. Check the actual implementation of special tokens in `model.py` rather than assuming a 2D mask suffices.

### Implementation instructions

1. Within **conditional block generation only**, derive:
   - `source_mask` (actual source, not fixed-width source padding),
   - `committed_mask` (previously committed target positions),
   - `role_clean_mask = source_mask | committed_mask`,
   - `validity_mask = source_mask | committed_mask | active_target_mask` (excluding unused canvas/padding).
2. Thread `role_clean_mask` and `attention_mask=validity_mask` through the full conditional flow-sampling call chain, including every model forward in the ODE/SDE, self-conditioning, and CFG branches. Use optional arguments defaulting to `None` so unconditional and legacy paths retain their exact old behavior.
3. Ensure the masks retain correct shapes with per-example source lengths and variable committed lengths, and are stable under JAX `jit`/`lax.scan`/`pmap` as applicable.
4. For the *final token decoder call*, inspect its **training** mask semantics first. Do not blindly apply denoiser masks to decoder mode if decoder training deliberately used a different pattern. If decoder masking must change, justify it with a dedicated ablation and preserve the default decoder path outside conditional-block mode.
5. **Do not change** the all-remaining-token active-to-active bidirectional attention: prediction remains joint across every unrevealed position.

### Required tests

- A clean-context invariance test: with fixed source/committed tokens and same timestep, changing only active noisy latents must **not** change hidden representations at source/committed positions (within numerical tolerance). Test all layers or the final clean hidden states as available.
- An active-dependence test: changing active noisy latents **may** change active outputs; confirm active positions can attend to all valid active positions.
- A padding invariance test: changing padded input values must not change any valid-position output.
- A direct-forward vs sampling-forward test: with guidance disabled, same latent/timestep/masks and self-conditioning setting, the denoiser prediction must match the direct model call.
- At least one batch with different source lengths and different committed lengths; ensure all indexing is per-example.
- Unconditional regression tests with fixed PRNG seeds: outputs/forward-call argument structure remain unchanged.

## D. P0 FIX #2 — Source-only classifier-free guidance (CFG)

### Evidence to verify

Training's label-drop path in `train_step.py` uses `batch["cond_seq_mask"]` for **source positions only**, whereas fixed-block generation's `cond_seq_mask` represents `source + committed target`. In `sampling_utils._forward_sample`, the CFG-unconditional branch zeroes **every** position indicated by `cond_seq_mask`; this can erase previously committed targets in the unconditional branch, which is different from source-only dropout during training.

### Desired conditional CFG semantics

The **conditional** branch sees source + committed targets. The **source-unconditional** branch removes source conditioning while **retaining committed target context**:

- `v_cond = v_theta(active noisy targets | source, committed targets)`;
- `v_uncond = v_theta(active noisy targets | committed targets)`;
- `v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)`.

`clean_mask` must still freeze **source + committed targets** for numerical integration. **Do not confuse** this clean-restoration mask with the **source-only dropout mask** used in CFG. Distinguish `cfg_source_drop_mask` from `restore_clean_mask` (names illustrative; fit the actual code).

### Implementation instructions

1. Implement **conditional-block-only** CFG source masking, keeping committed-target latents intact in both CFG branches.
2. Preserve committed-token values after *every* ODE/SDE step, independently of whether CFG is enabled.
3. Match training's label-drop semantics as closely as possible, including any **T5 encoder attention removal** done during label dropout. The trained source-drop mode may also have differences in target-context embeddings, so **do not claim that merely zeroing source latents perfectly reproduces training**. Inspect and test this before adding complex re-encoding or retraining.
4. Keep self-conditioning CFG (`self_cond_cfg_scale`) separate from *source* CFG (`cfg_scale`); do not accidentally rewrite self-conditioning logic.
5. Crucially, support `cfg_scale=1.0` as a control that bypasses the source-unconditional branch. Test this first; only add CFG extrapolation after the direct conditional path is correct.
6. Preserve the old `_forward_sample` behavior when invoked from unconditional or non-block paths (optional arguments/defaults or a conditional-only wrapper).

### Required tests/ablations

- Verify that in the CFG-unconditional branch **source positions are dropped, committed targets are preserved**, and unrevealed targets remain noisy.
- Confirm bitwise/near-numerical identity between the old conditional branch and new one at `cfg_scale=1`, except for intentional attention-mask effects.
- Run the same conditional checkpoint with identical seeds at `cfg_scale=1` and `cfg_scale=2` on a fixed dev subset; log BLEU (WMT14), ROUGE (XSum), lengths, EOS rate, and sample outputs.
- Confirm CFG changes do not affect LM1B/OWT even when they share generic sampling helpers.

## E. P1 INVESTIGATION #3 — Target canvas, EOS, and padding (do not assume the fix in advance)

### Observation to verify

`generate_block_cond` currently derives `target_capacity = max_length - source_lengths`. This is a **capacity**, not a predicted target length. For a short source, the sampler may denoise many more target positions than the target lengths actually supervised during training. With active-to-active attention, such extra positions can affect earlier predictions. This is a plausible train/infer distribution shift, **not proof of a bug**. **Never use reference target lengths at inference.**

### Investigation and minimal intervention

1. Log source length, target capacity, actual active size per round, generated length, first committed EOS position, and whether sequences hit the hard capacity. Inspect the WMT14 and XSum length distributions using the actual training/validation loader.
2. Check `pad_token=eos` carefully: distinguish uncommitted placeholder EOS/padding from **actually committed** EOS. Do not let a candidate EOS in a future uncommitted position stop generation. Verify decoding strips special tokens correctly.
3. Compare **without retraining**:
   - current capacity `max_length - actual_source_length`;
   - optional **fixed per-task target canvas** based on a declared target limit (e.g. `max_length - max_input_length`, typically 64 for current WMT14/XSum configs), retaining **tight source→target packing** and correct absolute position indices.
4. For either canvas, make the *validity/attention mask* exclude unused target slots and padding. Do not treat the padded tail as active.
5. Make a fixed-canvas option **conditional-only**, disabled by default until ablation evidence supports it. Never change `max_length` globally to force the result, and never shift target positions back to `max_input_length` (previous position mismatch fix).
6. Keep source padding and target padding distinguishable. Preserve original tokenizer and EOS conventions unless a test demonstrates a genuine bug.

### Tests

- Two examples with different source lengths map target-relative position 0 to their respective real source ends, with no mid-sequence padding gap.
- Active positions do not attend to extra padded positions.
- Committed tokens never change in subsequent rounds.
- EOS termination and trimming work for short/long sources and when EOS equals the padding ID.
- Report quality vs generated length under both canvas choices. Do not claim fixed canvas is universally better before measurement.

## F. P1 INVESTIGATION #4 — Conditional training/evaluation configuration audit (especially XSum)

1. Verify **effective runtime** configs and checkpoint metadata, not only YAML files. The currently inspected `train_xsum_ELF-B.yml` lacks `use_block_revelation: true` and therefore may be the *standard* ELF configuration; the user may have an additional local/override config. Do not assume the user's run uses this YAML.
2. Confirm `use_block_revelation=True`, chosen `block_size`, `use_staggered_noise=False` (for the fixed-commitment experiment), checkpoint epoch/step, source/target length limits, `pad_token`, source dropout, CFG, sampling steps and correct evaluator routing for both WMT14 and XSum.
3. Create **separate opt-in conditional block config files or override examples** if needed. Do not overwrite canonical configs or modify LM1B/OWT configs.
4. Confirm `run_generation` / manual evaluation actually calls `generate_block_cond()` when block mode is on, and not original full-sequence sampling. Preserve the original non-block path for matched control experiments.
5. Report translation **corpus BLEU and chrF** (if available) and summarization **ROUGE-1/2/L**, but do not assume a low metric reflects a model bug before checking evaluation preprocessing and reference alignment. Use the same dev subset/seeds across ablations.
6. Audit whether conditional model training budget (batch size, steps/epochs, number of supervised target positions) matches the intended ELF warm-start comparison. Treat undertraining as a possible confound, not a confirmed implementation bug.

## G. Required diagnostic experiments (before costly retraining)

Use the **existing conditional checkpoint** first; change one factor at a time:

| Test | Comparison / observation | Reason |
| --- | --- | --- |
| G1 | `cfg_scale=1` old vs masked conditional inference | Isolate attention mismatch |
| G2 | `cfg_scale=1` vs source-only CFG `cfg_scale=2` | Isolate guidance |
| G3 | current vs fixed 64-token target canvas, same seeds | Check length/canvas distribution shift |
| G4 | same noise, same committed prefix, different source sentences | Detect source insensitivity; compare denoiser outputs and generated texts |
| G5 | generated committed prefix vs **teacher-forced** committed prefix (diagnostic only) | Isolate error propagation |
| G6 | compare real runtime configs/checkpoints on WMT14 and XSum | Catch evaluation/configuration mismatch |

Teacher forcing may use ground-truth targets **only in explicitly labeled diagnostics**, never in real generation/scoring.

Prioritize G1 and G2; if a mask change alters the model's training distribution, note whether a short **conditional-only** fine-tune is needed and justify it. Do not launch a full training run without evidence.

## H. Acceptance criteria — must pass before saying 'fixed'

- [ ] WMT14 **and** XSum conditional block generation call the model with the intended role+validity masks for every flow step; the resulting clean context is invariant to active noisy values.
- [ ] Source-only CFG keeps committed tokens intact; numerical integration also preserves committed tokens at every step.
- [ ] Train/infer source and target absolute positions match; no source-to-target padding gap.
- [ ] All remaining valid target positions are denoised each round; only next `b` target positions are committed.
- [ ] Target canvas, padding and EOS are handled without reference-length leakage.
- [ ] **Seeded unconditional LM1B/OWT regression remains unchanged** (both `use_block_revelation=True` unconditional path and original non-block path, if applicable).
- [ ] No change to checkpoint parameter trees/shapes; original ELF checkpoints still load.
- [ ] Conditional smoke tests and quantitative paired ablations actually run; report failures/limitations transparently.

## I. What to deliver

1. A concise *evidence table* with: issue, exact current code location, confirmed/probable/not reproduced, minimal change, and test.
2. Minimal code diffs focused on conditional generation. Shared functions may acquire backwards-compatible optional arguments, but **the unconditional branches must remain behaviorally identical**.
3. Automated tests added to the repository and the actual test commands/results.
4. Paired WMT14/XSum before–after outputs and available BLEU/ROUGE numbers, with fixed checkpoints and seeds. If GPU/TPU/checkpoints are unavailable, say so and provide runnable commands—**do not invent metrics**.
5. A summary of remaining limitations (teacher-forced prefix exposure bias, possible target-canvas mismatch, T5 label-drop matching) and whether conditional-only retraining is justified.

**Do not** implement GRPO, continuous→categorical model conversion, masked diffusion, self-distillation, more sampling steps as a substitute for fixing bugs, or architectural rewrites. **Do not touch working unconditional generation behavior.**
