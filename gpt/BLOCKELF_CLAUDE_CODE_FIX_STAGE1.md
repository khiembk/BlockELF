# Claude Code task: Fix BlockELF Stage-1 conditioning, positions, attention, and evaluation

**Repository:** https://github.com/khiembk/BlockELF  
**Tasks in scope:** WMT14 German-to-English (conditional) and LM1B (unconditional).  
**Goal:** Make the existing **ELF-based**, fixed-block Stage-1 implementation correct, reproducible, and evaluable. **Keep ELF and its pretrained/checkpoint initialization pathways.** This is a focused debugging/engineering change, **not** a rewrite using Set Diffusion or Categorical Flow Maps.

## Non-negotiable research algorithm

For a target of length `L`, fixed commit size `b`, and revelation round `k`:

- `C_k = {1, ..., min(k*b, L)}`: **clean, previously committed** target positions; WMT14 additionally has a clean German source.
- `A_k = {min(k*b, L)+1, ..., L}`: **every remaining** target position.
- **Training:** sample one valid `k` per example, teacher-force the committed prefix, noise **every valid position in `A_k`**, and compute the existing ELF denoising loss over **all** `A_k` (not only the next block). Preserve ELF's continuous embedding-space flow matching and decoder-head training.
- **Inference:** at each round run ELF's flow sampler on **all** remaining target positions; decode candidate tokens; commit **only the next `b` target positions**, then repeat using the generated committed prefix. Never use ground-truth target tokens or reference target length during sampling.
- `b` controls **commitment size and context progression**, not the width of the predicted/denoised region.
- For LM1B there is **no source sequence**: only the clean committed prefix and the remaining noisy tokens.

Do **not** change this into current-block-only denoising, masked discrete diffusion, categorical flow matching, random-order revelation, GRPO, learned controllers, staggered schedules, or one-/few-step distillation. Do **not** introduce KV caching in this task; first make uncached sampling correct.

## First step: inspect and verify assumptions before editing

Check the current commit and actual call graph; do not blindly trust this document if the repository has changed. Inspect, at minimum:

- `src/train_step.py` — currently calls `encode_text` before sampling the fixed block round.
- `src/utils/data_utils.py`, `src/utils/encoder_utils.py`, `src/utils/block_utils.py` — source/target packing and teacher/committed/active masks.
- `src/modules/model.py`, `src/modules/layers.py` — ELF denoiser attention, special timestep/mode tokens, RoPE, 2-D/3-D mask interfaces.
- `src/utils/generation_utils.py` — already has `generate_block_cond` and `generate_block_uncond`; **reuse them**.
- `src/generation.py`, `src/eval.py` and any training-time evaluation entry points.
- Both WMT14 and LM1B configuration files; actual sampling/CFG settings and checkpoint loading paths.

Before changes, briefly document the observed behavior and reproduce relevant issues with small deterministic tests. If one reported problem is not present on the current commit, explain the evidence and do not alter it unnecessarily.

## Fix A — T5 teacher-encoder future-target leakage (previous review: Problem 1)

**Suspected current behavior:** in `train_step.py`, `batch["encoder_attention_mask"]` is consumed to produce `x0` with frozen T5 **before** `sample_block_round()` chooses a committed prefix. The original encoder mask protects only the German source, so previously committed target tokens may be contextualized using unrevealed reference tokens. At inference, the committed prefix is encoded without those references.

**Required implementation:**

1. When `use_block_revelation=True` and `use_staggered_noise=False`, sample `committed_mask`, `active_mask`, and the boundary **before** the frozen T5 `encode_text` call.
2. Build a per-example encoder attention mask for three roles: source (WMT14 only), committed target, and active target. Define `clean = source ∪ committed`.
   - **Clean query** may attend only to **valid clean keys**.
   - **Active query** may attend to all **valid** source/committed/active keys to retain ELF's existing teacher-endpoint construction. Active teacher embeddings are supervision targets; they must not be reused as uncorrupted conditioning embeddings.
   - **Padding keys** must be excluded. Handle padded query rows safely.
3. Encode using this new mask for the blockwise branch; preserve original encoder behavior when block mode is disabled.
4. Keep `add_noise`, clean restoration, active-only FM loss, label dropout, CFG/self-conditioning, and the ELF decoder branch compatible. In particular, do not silently change the meaning of the existing ELF losses.
5. Ensure training and inference encode the **same source + committed token IDs** under equivalent allowed-key masks. For LM1B, the empty committed prefix at `k=0` must work.

**Regression test:** hold source and committed token IDs fixed, replace **all unrevealed reference token IDs** with different values, and re-encode. Source/committed hidden states must remain invariant (up to numeric tolerance). Test WMT14 and LM1B, `k=0`, middle `k`, and near-last `k`. Ensure the active teacher embeddings/loss are still sensitive to the active target IDs. Also check that PAD/EOS-as-pad variants do not leak padding.

## Fix B — Consistent training/inference target positions (Problem 2)

**Suspected mismatch:** training's `get_dataloader()` packs `[actual source tokens][actual target tokens]` before padding (`cond_len` varies per example), whereas `generate_block_cond()` currently constructs `full_ids = concat(source_ids, committed_target_ids)` with `source_ids` padded to `max_input_length`, making generated targets start at fixed absolute index `max_input_length`. RoPE then sees different source-to-target distances; a padding gap may be treated incorrectly.

**Required implementation:**

1. Establish **one canonical serialization/layout**, matching ELF's existing packed training representation wherever possible:
   `valid source tokens | target canvas (committed prefix + all remaining slots) | trailing padding`.
   For LM1B the source segment is absent. No unmasked padding **between** valid source and target positions.
2. In WMT14 block inference, target position `j` must occupy the same absolute position as during training for a source of the same length: `absolute_index = actual_source_length + j` (subject to the repository's indexing convention). Use explicit per-example offsets/role masks; do not assume a batch has identical source lengths.
3. Respect `max_length`, `max_input_length`, per-example truncation, and JAX's fixed tensor shapes. It is okay to implement a static-sized packed/scatter/gather representation if token positions and masks are correct.
4. Reconcile **target canvas capacity** with the real source length (not a fixed padding gap); do **not** use the reference target length at inference. EOS termination is based only on generated, committed positions. Ensure `pad_token == "eos"` cannot accidentally signal completion from uncommitted filler.
5. Verify that decoder slicing, committing `b` tokens, EOS detection, and final detokenization use *target-relative* indices rather than an invalid fixed absolute offset.
6. Preserve original full-sequence ELF evaluation behavior when block mode is off.

**Regression tests:** use a batch of source lengths e.g. `5, 17, 63` (where supported), with targets containing known sentinel IDs. Check train/infer role and positional-index equivalence, no interstitial unmasked PAD, correct target extraction/commit order, no overlap into source, correct final partial block, and independence from reference target tokens/length. Repeat for LM1B `source_length=0`.

## Fix C — Block-aware attention in the ELF denoiser (Problem 3)

**Current concern:** `src/modules/model.py` can accept an `attention_mask`, and `src/modules/layers.py` handles 2-D or 3-D masks, but train/sampling calls may omit a correct *query–key* role mask. Ordinary bidirectional attention allows clean source/committed representations to read the active noisy target; this can break conditioning consistency and eventual KV-cache reuse.

For valid content positions, implement this mask (1 = allowed attention):

| Query \ Key | clean source/committed `C` | noisy remaining `A` |
|---|---:|---:|
| clean `C` | 1 | **0** |
| active `A` | 1 | **1** |

**Important ELF-specific details:**

1. Account for the model's prepended **time**, **self-cond CFG**, and optional **model-mode** tokens. Audit their query/key access to prevent **indirect leakage across transformer layers** (`active -> special token -> clean`). A safe initial rule is that special-token representations cannot read active target keys if clean queries can later attend to them. Active queries still need access to time/CFG/mode conditioning. Document the final special-token rules.
2. Respect padding keys/queries and verify attention semantics with existing `scaled_dot_product_attention` (which supports masks shaped `[B, Q, K]`). Avoid fully-masked valid query rows/NaNs. Check the mask extension logic in `ELF.__call__`, which currently prepends 1-D prefix masks: adapt it for **both** 2-D key masks and 3-D query–key masks without changing learned parameter names/shapes.
3. Thread the block-role mask through the **denoiser forward calls** in training (including self-conditioning auxiliary passes) and **each ODE/SDE denoising call** during blockwise inference. Ensure any CFG branches use matching masks. Do not assume masking only the frozen T5 encoder solves this issue.
4. Preserve the model's **all-active-to-all-active** attention. Do not copy Block Diffusion's current-block-only denoising mask.
5. Do not claim working KV caching at this milestone. ELF's timestep-conditioned special tokens may change with time, so cache validity would need separate work.
6. Keep legacy ELF operation (including loading original checkpoints) behind the old/default attention path when `use_block_revelation=False`. Prefer **no new parameters**; attention changes require fine-tuning and do not guarantee immediate zero-shot quality from an unmodified checkpoint.

**Regression tests:**

- At fixed `t`, change noisy inputs at all active positions; **clean hidden representations** at every denoiser layer must stay invariant, including for depth >1 and with special tokens present. Active predictions should generally change.
- Verify exact allowed edges for toy role layouts; no active-key path into clean representations, including indirect paths through special tokens.
- Verify masks in self-cond/CFG branches and `num_model_mode_tokens` configurations; pad-only query handling must be stable.
- With `use_block_revelation=False`, original model behavior/checkpoint shape and reference evaluation should remain unchanged (within numerical reproducibility tolerances).

## Fix D — Route evaluations to the actual block sampler (Problem 6)

**Suspected current behavior:** `src/eval.py` and `src/generation.py` call `test_generation_cond()`/`test_generation_uncond()`, which invoke `_make_pmap_pair()` and `_generate_samples_single_batch()` **once over the whole sequence**; they do not dispatch to the already-implemented `generate_block_cond()`/`generate_block_uncond()` loop. This can make Stage-1 metrics measure the **wrong inference algorithm**.

**Required implementation:**

1. If fixed-block Stage 1 is enabled (or an explicit `eval_mode=block` flag is used), dispatch **WMT14** to `generate_block_cond()` and **LM1B** to `generate_block_uncond()`. When disabled, preserve the original ELF full-sequence evaluator and metrics exactly.
2. Reuse the existing generation helpers, updating them as necessary for fixes B/C and for correct handling of replicated/unreplicated JAX states and encoder parameters. Since helpers are currently single-device/eager, a correct single-device evaluation path is acceptable initially. **Never silently fall back** to full-sequence evaluation when a requested block mode cannot run; fail clearly.
3. Use the **same** sampling method (ODE/SDE), scheduler, step count, CFG/self-conditioning parameters, EMA selection, tokenizer, and seed conventions as the chosen configuration. Preserve ELF's output logging and avoid cross-contamination between full and block eval runs.
4. Add explicit run labels and metadata: `generation_mode`, `block_size`, `num_reveal_rounds`, `flow_steps_per_round`, total generator `NFE` (count actual model forwards; note auxiliary CFG/self-cond calls), wall-clock latency, `sampling_method`, `seed`, and checkpoint ID. NFE and decoder/T5 calls should be tracked separately.
5. For **WMT14**, compute BLEU against references **only after generation**. Do not pass target IDs or reference lengths to the sampler.
6. For **LM1B**, continue the repository's unconditional generation metrics (e.g. Gen-PPL/entropy) and reuse its evaluator if compatible. Ensure the metric code sees the generated **blockwise** samples and keep full-sequence results separately labeled.
7. If generation is invoked by both `eval.py` and training's periodic evaluation, route both consistently. Keep CLI/YAML overrides ergonomic; add example commands only after confirming real flags.

**Integration test:** monkeypatch or instrument the two block sampler entry points to count invocations. With block mode on, the relevant block sampler **must** run and the original full-sequence sampler **must not**. With block mode off, the reverse must hold. Run for WMT14 and LM1B; verify no reference IDs are passed into any sampler. Confirm BLEU/Gen-PPL is calculated from block output and that distinct modes never overwrite the same output files.

## Tests and acceptance criteria (mandatory)

Create a lightweight test suite (CPU/JAX where practical), with deterministic PRNG seeds and tiny synthetic batches. Include at least:

1. **Teacher T5 no-leak test:** varying uncommitted reference tokens never changes clean source/committed embeddings for WMT14 or committed embeddings for LM1B.
2. **Denoiser no-leak test:** varying active noise cannot change clean hidden states under the block-aware mask, including through prefix time/mode tokens and multiple layers.
3. **All-remaining supervision test:** with `L=12, b=4`, training masks must be: `k=0` active `1..12`; `k=1` active `5..12`; `k=2` active `9..12`. No loss on source, committed, or pad positions.
4. **Train/infer layout test:** variable WMT14 source lengths and LM1B's zero source length preserve target-relative positions and exclude gaps/padding.
5. **Commit test:** with `L=12, b=4`, all 12 positions are sampled each round; only `1..4`, then `5..8`, then `9..12` are committed. Earlier committed tokens never change.
6. **EOS/padding test:** EOS only terminates if it has been generated and committed; filler EOS from `pad_token=eos` must not terminate early.
7. **Evaluation routing test:** block mode always runs blockwise sampling; full ELF mode always runs the original sampler; both conditional/unconditional paths work.
8. **Checkpoint compatibility test:** verify original ELF checkpoint parameter tree/names/shapes still load, and the legacy `use_block_revelation=False` path remains runnable. Report numerical differences rather than asserting bitwise equality when stochastic generation is involved.
9. **Tiny end-to-end smoke tests:** a few WMT14 pairs and short unconditional LM1B samples, both with and without block mode. Validate finite loss, finite output, token IDs in vocab, no unexpected JAX tracing/pmap shape errors.

**Important:** if checkpoint weights are unavailable locally or GPU/TPU runtime is inaccessible, run the available CPU/synthetic tests and document exactly which integration/quality tests remain unexecuted; do not invent BLEU/Gen-PPL improvements.

## Implementation order

1. **Baseline audit:** record Git commit hash, configs, current evaluation dispatch, and a deterministic minimal reproduction. Preserve baseline behavior under a legacy config.
2. **Fix D (eval dispatch)** first so the test metric actually measures the intended sampler. Write routing tests.
3. **Fix B (layout)** and add train/infer position parity tests.
4. **Fix A (T5 masking)** and prove teacher-forced committed context does not contain future reference information.
5. **Fix C (denoiser masking + special tokens)**, threading masks through every relevant train/inference/CFG/self-conditioning call. Add multi-layer hidden-state invariance tests.
6. **Run regression + smoke tests**, then a small side-by-side comparison with matched seeds/configs on the *same subset*: original full-sequence ELF, old BlockELF if runnable, fixed BlockELF after fixes. Distinguish effects of the new masks (which may require retraining) from pure inference fixes.

## Deliverables

- Actual code edits in this repository, not only a design write-up.
- Focused tests executed and the commands/results reported.
- A short `docs/STAGE1_FIXES.md` with: root-cause confirmation or rejection for each issue A/B/C/D; list of changed files and exact behavior; mask/position diagrams; checkpoint compatibility notes; commands for WMT14/LM1B block and full-sequence evaluation; and remaining limitations.
- A concise final summary listing **verified** fixes, tests that passed/failed/skipped, and any follow-up that genuinely requires new training.

## Guardrails

- Preserve ELF's continuous embedding-space flow objective, frozen T5 encoder, learned decoder, optimizer/EMA, self-conditioning/CFG options, tokenizer, and existing checkpoint initialization/restore paths.
- Do not claim that an ELF checkpoint trained with ordinary bidirectional attention will perform well immediately under a new block mask. It may require **Stage-1 fine-tuning** on the corrected masks.
- Do not replace the source conditioning of WMT14 with decoder-only LM behavior; do not insert artificial German source tokens in LM1B.
- Do not introduce RL, set-causal random revelation, distillation, new flow objectives, cache acceleration, or unnecessary architecture changes.
- Never use reference translations or their true lengths to determine generation canvas, stopping time, or committed tokens.
- Favor small, verifiable, backwards-compatible changes over large refactors. When an assumption proves wrong, document it instead of forcing an unnecessary patch.

**Start by inspecting the repository and writing the minimal failing tests; then implement the fixes in the above order. Continue through execution and report observed outcomes rather than just proposing changes.**
