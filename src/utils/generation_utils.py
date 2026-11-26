from functools import partial
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
from flax import jax_utils
from jax import Array

from configs.config import Config, SamplingConfig
from utils.logging_utils import log_for_0
from utils.sampling_utils import (
    restore_cond, _ode_step, _sde_step, get_sampling_steps,
)
from utils.encoder_utils import encode_text
from utils.block_utils import build_committed_context
from utils.data_utils import get_pad_token_id
from modules.t5_encoder import get_encoder

PRNGKey = jax.random.PRNGKey


# ============================================
# Generation utilities
# ============================================


def mask_after_eos(predicted_ids, eos_token_id, pad_token_id):
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = predicted_ids == eos_token_id
    keep_mask = jnp.cumsum(eos_mask, axis=1) == 0
    return jnp.where(keep_mask, predicted_ids, pad_token_id)


def shift_left(x, shift_per_sample, pad_value=0, axis=1):
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.ndim < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    axis = axis if axis >= 0 else x.ndim + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    shift_per_sample = shift_per_sample.astype(jnp.int32)
    if axis != 1:
        x = jnp.moveaxis(x, axis, 1)
    seq_len = x.shape[1]
    base_idx = jnp.arange(seq_len)[None, :]
    gather_idx = shift_per_sample[:, None] + base_idx
    valid = gather_idx < seq_len
    gather_idx = jnp.clip(gather_idx, 0, seq_len - 1)
    if x.ndim == 2:
        shifted = jnp.take_along_axis(x, gather_idx, axis=1)
        shifted = jnp.where(valid, shifted, pad_value)
    else:
        expand_axes = tuple(range(2, x.ndim))
        shifted = jnp.take_along_axis(x, jnp.expand_dims(gather_idx, expand_axes), axis=1)
        shifted = jnp.where(jnp.expand_dims(valid, expand_axes), shifted, pad_value)
    if axis != 1:
        shifted = jnp.moveaxis(shifted, 1, axis)
    return shifted


# ============================================
# Multi-device helpers (pmap)
# ============================================

def _sample_step_for_scan(
    model_apply_fn, model_params, config, sampling_config: SamplingConfig,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask, rng=None,
    attention_mask=None, role_clean_mask=None, cfg_source_mask=None,
):
    """Create a scan-compatible step function.

    For method == "sde", `rng` must be provided and the scan carry must include a step index
    (z, x_pred, step_idx); fold_in is done per step. Other methods use a (z, x_pred) carry.

    attention_mask/role_clean_mask/cfg_source_mask: optional, see `_forward_sample`'s
    docstring in sampling_utils.py (conditional_gen_fix.md P0 Fix #1/#2). Default None
    preserves the original behavior for every existing caller.
    """
    method = sampling_config.sampling_method
    base_kwargs = dict(
        model_apply_fn=model_apply_fn, model_params=model_params,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        attention_mask=attention_mask, role_clean_mask=role_clean_mask,
        cfg_source_mask=cfg_source_mask,
    )

    if method == "sde":
        assert rng is not None, "SDE method requires rng to be passed to _sample_step_for_scan"
        sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)

        def step_fn(carry, t_pair):
            z, x_pred, step_idx = carry
            t, t_next = t_pair
            step_rng = jax.random.fold_in(rng, step_idx)
            z_new, x_pred_new = _sde_step(
                z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                gamma=sde_gamma, rng=step_rng, **base_kwargs,
            )
            return (z_new, x_pred_new, step_idx + 1), None
        return step_fn

    if method == "ode":
        base_step_fn = _ode_step
    else:
        raise ValueError(f"Invalid sampling method: {method}")

    def step_fn(carry, t_pair):
        z, x_pred = carry
        t, t_next = t_pair
        z_new, x_pred_new = base_step_fn(
            z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **base_kwargs,
        )
        return (z_new, x_pred_new), None
    return step_fn


def _generate_samples_single_batch(
    model_params, model_apply_fn, rng: PRNGKey, z: Array, t_steps: Array,
    cond_seq: Array, cond_seq_mask: Array, config: Config, sampling_config: SamplingConfig,
    cfg_scale: float, self_cond_cfg_scale: float,
    attention_mask=None, role_clean_mask=None, cfg_source_mask=None,
) -> Array:
    """Generate samples for a single batch (pmap-compatible, uses lax.scan).

    attention_mask/role_clean_mask/cfg_source_mask: optional, see `_forward_sample`'s
    docstring in sampling_utils.py (conditional_gen_fix.md P0 Fix #1/#2). Default None
    preserves the original behavior for every existing (unconditional/non-block) caller.
    """
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = jnp.zeros((batch_size, max_length, d_model))
        cond_seq_mask = jnp.zeros((batch_size, max_length))
    step_kwargs = dict(
        model_apply_fn=model_apply_fn, model_params=model_params,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        attention_mask=attention_mask, role_clean_mask=role_clean_mask,
        cfg_source_mask=cfg_source_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)

    t_pairs = jnp.stack([t_steps[:-2], t_steps[1:-1]], axis=1)
    if method == "sde":
        step_fn = _sample_step_for_scan(sampling_config=sampling_config, rng=rng, **step_kwargs)
        (z, x_pred, _), _ = jax.lax.scan(step_fn, (z, x_pred, jnp.int32(0)), t_pairs)
    else:
        step_fn = _sample_step_for_scan(sampling_config=sampling_config, **step_kwargs)
        (z, x_pred), _ = jax.lax.scan(step_fn, (z, x_pred), t_pairs)

    # Last step always with ode
    z, x_pred = _ode_step(
        z=z, t=t_steps[-2], t_next=t_steps[-1], x_pred_prev=x_pred, **step_kwargs,
    )
    return z


def _dlm_decode_logits_batch(z, model_params, model_apply_fn, t_final_val, config, self_cond_cfg_scale):
    """Shared DLM-decoder-head forward pass; returns raw `decoder_logits` (not argmax)
    so callers can choose argmax (the default/legacy path, via `_dlm_decode_batch`
    below) or a sampling distribution (OWT-only opt-in, via `_sample_decoder_tokens`)
    without duplicating this forward pass. `_dlm_decode_batch`'s own signature/behavior
    is unchanged by this refactor -- it just now delegates here internally -- so every
    existing caller (`generate_block_cond`, `generate_block_uncond`'s greedy default,
    and the `jax.pmap`-wrapped `_make_pmap_pair` path) is unaffected.
    """
    batch_size = z.shape[0]
    t_final = jnp.full((batch_size,), t_final_val, dtype=z.dtype)
    self_cond_cfg_scale_batch = (
        jnp.full((batch_size,), self_cond_cfg_scale, dtype=z.dtype)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = jnp.concatenate([z, jnp.zeros_like(z)], axis=-1) if config.self_cond_prob > 0 else z
    _, decoder_logits = model_apply_fn(
        {"params": model_params}, z_input, t_final,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale_batch,
        decoder_step_active=jnp.array(True),
    )
    return decoder_logits


def _dlm_decode_batch(z, model_params, model_apply_fn, t_final_val, config, self_cond_cfg_scale):
    """Decode z→tokens with the DLM decoder head (argmax). Default/legacy path for
    every caller; signature and behavior unchanged from before the
    `_dlm_decode_logits_batch` refactor above."""
    decoder_logits = _dlm_decode_logits_batch(
        z=z, model_params=model_params, model_apply_fn=model_apply_fn,
        t_final_val=t_final_val, config=config, self_cond_cfg_scale=self_cond_cfg_scale,
    )
    return jnp.argmax(decoder_logits, axis=-1)


def _sample_decoder_tokens(decoder_logits, rng, mode, temperature=1.0, top_p=1.0):
    """OWT-only opt-in decoder sampling (NOT used by `_dlm_decode_batch`/the default
    greedy path, nor by `generate_block_cond`/WMT14/XSum). Caller is expected to
    pre-slice `decoder_logits` to only the positions it actually wants sampled (e.g.
    just this round's newly-committed block) -- this function samples every position
    it's given, independently, via a single `rng` (JAX's categorical/Gumbel-max
    sampling already draws independent noise per position from one key; no manual
    per-position key-splitting is needed for independence).

    mode='categorical': sample from softmax(logits / temperature).
    mode='top_p' (nucleus): sort descending, keep the smallest prefix whose cumulative
    probability >= top_p (always keeping at least the single highest-probability
    token), mask the rest to -inf, renormalize implicitly via categorical sampling over
    the masked logits, then map back to original vocab ids.
    """
    assert mode in ("categorical", "top_p"), f"_sample_decoder_tokens: unsupported mode {mode!r}"
    scaled_logits = decoder_logits / jnp.maximum(temperature, 1e-6)
    if mode == "categorical":
        return jax.random.categorical(rng, scaled_logits, axis=-1)

    # top_p (nucleus) sampling.
    sorted_order = jnp.argsort(scaled_logits, axis=-1)[..., ::-1]  # descending
    sorted_logits = jnp.take_along_axis(scaled_logits, sorted_order, axis=-1)
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumprobs = jnp.cumsum(sorted_probs, axis=-1)
    # Keep position i iff everything strictly before it has cumulative prob < top_p --
    # i.e. position i is needed to cross the top_p threshold. Position 0 (the single
    # highest-probability token) is always kept, even if top_p is tiny.
    keep = (cumprobs - sorted_probs) < top_p
    keep = keep.at[..., 0].set(True)
    neg_inf = jnp.finfo(sorted_logits.dtype).min
    filtered_sorted_logits = jnp.where(keep, sorted_logits, neg_inf)
    sampled_sorted_pos = jax.random.categorical(rng, filtered_sorted_logits, axis=-1)
    return jnp.take_along_axis(sorted_order, sampled_sorted_pos[..., None], axis=-1)[..., 0]


# ============================================
# Shared generation scaffolding
# ============================================
def _make_pmap_pair(model_apply_fn, config, sampling_config, cfg_scale, self_cond_cfg_scale):
    """Build pmapped (generate, decode) pair for a (cfg, sccfg) combo."""
    p_generate = jax.pmap(
        partial(
            _generate_samples_single_batch,
            model_apply_fn=model_apply_fn, config=config, sampling_config=sampling_config,
            cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        ),
        axis_name="batch",
    )
    p_decode_ids = jax.pmap(
        partial(
            _dlm_decode_batch, model_apply_fn=model_apply_fn, config=config,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
    )
    return p_generate, p_decode_ids


def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{sde_str}-{suffix}"


def _shard_timesteps(t_rng, num_local_devices, num_sampling_steps, time_schedule, config):
    t_device_rngs = jax.random.split(t_rng, num_local_devices)
    return jnp.stack([
        get_sampling_steps(
            t_device_rngs[i], n_steps=num_sampling_steps,
            time_schedule=time_schedule, P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        )
        for i in range(num_local_devices)
    ])


def _shard_noise(device_rngs, num_local_devices, per_device, max_length, d_model, noise_scale):
    return jnp.stack([
        jax.random.normal(device_rngs[i], (per_device, max_length, d_model)) * noise_scale
        for i in range(num_local_devices)
    ])


# ============================================
# Block-to-block (fixed-commitment) inference — Adaptive BlockFlow Stage 1, Alg. 2
# ============================================
def pack_source_target_tight(source_ids, committed_target_ids, source_lengths, max_input_length, max_length):
    """Pack `[actual source tokens][committed target tokens][trailing pad]` with NO gap
    between source and target, matching training's `collate_fn` (cond_len varies per
    example, tight-packed against target) -- see Fix A/B history in
    `generate_block_cond`'s docstring. `source_ids` is (B, max_input_length)
    (pad-token-filled beyond each row's `source_lengths[i]`); `committed_target_ids` is
    (B, storage_width) with storage_width a static upper bound (caller's choice, e.g.
    `max_length`). Returns `(full_ids, is_source_full, is_committed_full)`, each
    `(B, max_length)` — `is_committed_full` marks positions < `source_lengths[i] +
    committed_len[i]` implicitly via `committed_target_ids`'s own padding, i.e. the
    caller is still responsible for its own role masks from `committed_len`; this
    function only handles the tight *positional* packing.
    """
    batch_size = source_ids.shape[0]
    pos_full = jnp.arange(max_length)[None, :]
    is_source_full = pos_full < source_lengths[:, None]
    wide = jnp.concatenate([source_ids, committed_target_ids], axis=1)
    gather_idx = jnp.where(
        is_source_full, pos_full, pos_full + (max_input_length - source_lengths[:, None]),
    )
    gather_idx = jnp.clip(gather_idx, 0, wide.shape[1] - 1)
    full_ids = jnp.take_along_axis(wide, gather_idx, axis=1)
    return full_ids, is_source_full


def extract_target_relative(full_tensor, source_lengths, storage_width, max_length):
    """Inverse of `pack_source_target_tight`'s positional packing: extract the
    target-relative region starting at each row's own `source_lengths[i]` (not a fixed
    offset), from a `(B, max_length, ...)` tensor (ids or embeddings). Returns
    `(B, storage_width, ...)`.
    """
    pos_store = jnp.arange(storage_width)[None, :]
    gather_idx = jnp.clip(pos_store + source_lengths[:, None], 0, max_length - 1)
    if full_tensor.ndim == 2:
        return jnp.take_along_axis(full_tensor, gather_idx, axis=1)
    extra = (1,) * (full_tensor.ndim - 2)
    gather_idx = gather_idx.reshape(gather_idx.shape + extra)
    return jnp.take_along_axis(full_tensor, jnp.broadcast_to(gather_idx, gather_idx.shape[:2] + full_tensor.shape[2:]), axis=1)


def generate_block_cond(
    state: Any,
    encoder_params: Dict,
    encoder_apply_fn,
    tokenizer,
    config: Config,
    sampling_config: SamplingConfig,
    source_ids: Array,
    source_lengths: Array,
    rng: PRNGKey,
    use_ema: bool = True,
):
    """Block-to-block conditional generation (single device, eager Python loop).

    Each round re-encodes the growing committed context
    (source + already-committed target) through the frozen T5 encoder and feeds it in
    as the sampler's "clean" cond_seq/cond_seq_mask — the same restore-clean-positions
    mechanism `_generate_samples_single_batch` already uses for the source prefix,
    just extended to cover committed target tokens too. A full, uncached flow solve
    runs every round (the documented M1/M2 baseline; no KV-cache reuse — see
    docs/main.tex Sec. 6). Not pmapped: this is a smoke-test / qualitative-eval
    utility, not a production multi-device eval path.

    Args:
        source_ids: (B, config.max_input_length) int array, right-padded with the
            pad token id (as produced by the existing dataloader's collate_fn).
        source_lengths: (B,) int array, number of real (non-pad) source tokens.

    Returns:
        List of B decoded strings (special tokens stripped).
    """
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id
    block_size = config.block_size
    max_input_length = config.max_input_length
    max_length = config.max_length

    model_params = state.ema_params1 if use_ema else state.params
    model_apply_fn = state.apply_fn
    cfg_scale = sampling_config.cfgs[0]
    self_cond_cfg_scale = sampling_config.self_cond_cfg_scales[0]

    batch_size = source_ids.shape[0]
    pos_full = jnp.arange(max_length)[None, :]      # absolute position in the packed canvas
    storage_width = max_length                      # static upper bound on target_capacity (source_lengths>=0)
    pos_store = jnp.arange(storage_width)[None, :]   # target-relative position within storage
    is_source_full = pos_full < source_lengths[:, None]

    # Fix B: target canvas CAPACITY is reconciled with each example's actual source
    # length (max_length - source_lengths[i]), not a fixed worst-case budget
    # (max_length - max_input_length) -- a shorter source affords strictly more real
    # target room, matching training's tight packing (see below), not less.
    target_capacity = max_length - source_lengths  # (B,), each in [max_length-max_input_length, max_length]

    committed_target_ids = jnp.full((batch_size, storage_width), pad_token_id, dtype=jnp.int32)
    committed_len = jnp.zeros((batch_size,), dtype=jnp.int32)
    finished = jnp.zeros((batch_size,), dtype=bool)

    max_rounds = (storage_width + block_size - 1) // block_size
    for _ in range(max_rounds):
        if bool(jnp.all(finished | (committed_len >= target_capacity))):
            break

        # Fix B: pack [actual source][committed target][trailing pad] with NO gap,
        # matching training's collate_fn (cond_len varies per example, tight-packed
        # against target). The previous version concatenated a FIXED-width
        # (max_input_length-padded) source with the target canvas, so target position 0
        # always sat at the fixed absolute index `max_input_length` regardless of the
        # real source length -- a different (usually larger) source->target relative
        # RoPE distance than any source of that length saw in training, plus an
        # unmasked mid-sequence padding gap that training never has.
        full_ids, _ = pack_source_target_tight(
            source_ids, committed_target_ids, source_lengths, max_input_length, max_length,
        )

        is_committed_full = (pos_full >= source_lengths[:, None]) & (
            pos_full < (source_lengths + committed_len)[:, None]
        )
        is_valid_full = is_source_full | is_committed_full
        encoder_attn, _, clean_mask = build_committed_context(
            is_source_full, is_committed_full, is_valid_full,
        )

        rng, enc_rng, noise_rng, t_rng = jax.random.split(rng, 4)
        cond_seq = encode_text(
            input_ids=full_ids, attention_mask=encoder_attn,
            encoder_apply_fn=encoder_apply_fn, encoder_params=encoder_params,
            latent_mean=config.latent_mean, latent_std=config.latent_std,
        )
        d_model = cond_seq.shape[-1]
        z0 = jax.random.normal(noise_rng, (batch_size, max_length, d_model)) * config.denoiser_noise_scale

        t_steps = get_sampling_steps(
            t_rng, n_steps=sampling_config.num_sampling_steps[0],
            time_schedule=sampling_config.time_schedule,
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        )
        # conditional_gen_fix.md P0 Fix #1: thread the same clean/active role mask
        # training uses (`clean_mask` = source + committed target; "clean" queries
        # never attend to "active" keys) into the denoiser's own self-attention here --
        # previously the sampling call chain passed no attention/role mask at all, so
        # inference silently ran with fully unmasked attention regardless of what
        # training taught the model to expect.
        # P0 Fix #2: CFG's unconditional branch must only drop the SOURCE (matching
        # training's label-drop, which never touches committed target), not the whole
        # source+committed `clean_mask` -- otherwise it erases committed-target context
        # the model was never trained to operate without.
        z_final = _generate_samples_single_batch(
            model_params=model_params, model_apply_fn=model_apply_fn, rng=rng,
            z=z0, t_steps=t_steps, cond_seq=cond_seq, cond_seq_mask=clean_mask,
            config=config, sampling_config=sampling_config,
            cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            role_clean_mask=clean_mask,
            cfg_source_mask=is_source_full.astype(jnp.float32),
        )
        predicted_ids = _dlm_decode_batch(
            z=z_final, model_params=model_params, model_apply_fn=model_apply_fn,
            t_final_val=t_steps[-1], config=config, self_cond_cfg_scale=self_cond_cfg_scale,
        )
        # Target-relative extraction: row i's target starts right after its own real
        # source length (not the fixed max_input_length), inverse of the packing above.
        predicted_target_ids = extract_target_relative(
            predicted_ids, source_lengths, storage_width, max_length,
        )

        # Commit the next `block_size` tokens of the active region (left-to-right).
        commit_upto = jnp.minimum(committed_len + block_size, target_capacity)
        newly_committed = (pos_store >= committed_len[:, None]) & (pos_store < commit_upto[:, None])
        committed_target_ids = jnp.where(newly_committed, predicted_target_ids, committed_target_ids)

        # EOS can only end generation at its own position once everything before it is
        # committed (Sec. 5); since we always commit left-to-right contiguously, the
        # first EOS among the committed-so-far tokens is exactly that position. Restrict
        # the check to positions actually committed by this round (`pos_store < commit_upto`)
        # — with pad_token: eos, pad_token_id == eos_token_id, so the not-yet-committed
        # filler tail would otherwise spuriously register as an EOS hit.
        is_eos = (committed_target_ids == eos_token_id) & (pos_store < commit_upto[:, None])
        has_eos = jnp.any(is_eos, axis=1)
        first_eos_pos = jnp.argmax(is_eos, axis=1)
        committed_len = jnp.where(finished, committed_len, commit_upto)
        committed_len = jnp.where((~finished) & has_eos, first_eos_pos + 1, committed_len)
        finished = finished | has_eos | (committed_len >= target_capacity)
        committed_target_ids = jnp.where(
            pos_store < committed_len[:, None], committed_target_ids, pad_token_id,
        )

    texts = []
    committed_len_np = np.array(committed_len)
    committed_target_ids_np = np.array(committed_target_ids)
    for i in range(batch_size):
        ids = committed_target_ids_np[i, :committed_len_np[i]]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def generate_block_uncond(
    state: Any,
    encoder_params: Dict,
    encoder_apply_fn,
    tokenizer,
    config: Config,
    sampling_config: SamplingConfig,
    batch_size: int,
    rng: PRNGKey,
    use_ema: bool = True,
    decoder_mode: str = "greedy",
    decoder_temperature: float = 1.0,
    decoder_top_p: float = 1.0,
):
    """Block-to-block unconditional generation (single device, eager Python loop).

    decoder_mode/decoder_temperature/decoder_top_p: OWT-only opt-in decoder-sampling
    ablation (see `_sample_decoder_tokens`), investigating OWT's long-canvas repetition
    collapse. Default `decoder_mode="greedy"` is the exact original `_dlm_decode_batch`
    argmax path -- bit-for-bit unchanged, and critically does NOT consume any
    additional `rng` split, so the noise/timestep RNG sequence seen by every other
    caller (and by this function itself in greedy mode) is identical to before this
    parameter existed. Only `decoder_mode in ("categorical", "top_p")` draws an extra
    `rng` split, and only for sampling this round's `block_size`-wide newly-committed
    window (not the full canvas) -- positions outside that window are irrelevant
    (they get overwritten by the existing `jnp.where(newly_committed, ...)` logic
    below regardless of what's sampled there), so restricting the sort/softmax cost to
    that window avoids an unnecessary O(canvas_length) sampling cost per round.
    Never changes `generate_block_cond` (WMT14/XSum) or any other caller.

    Unconditional analogue of `generate_block_cond`: there is no source prefix, so the
    full `config.max_length` canvas is the target set `A_0` on round 0 (matching the
    `cond_len=0` case `sample_block_round`/`train_step.py` already handle during
    training). Each round re-encodes the growing committed-target prefix through the
    frozen T5 encoder (self-attention restricted to committed positions, via
    `build_committed_context` with `is_source` all-False) and feeds it in as the
    sampler's "clean" cond_seq/cond_seq_mask, exactly as `generate_block_cond` does for
    source + committed target. A full, uncached flow solve runs every round (M1/M2
    baseline — see docs/main.tex Sec. 6). Not pmapped: a smoke-test / qualitative-eval
    utility, not a production multi-device eval path.

    Role-aware attention (committed queries restricted to committed-only keys, matching
    `train_step.py`'s fixed-block, non-staggered denoiser objective) is applied whenever
    the checkpoint supports it (`use_block_revelation=True`, `use_staggered_noise=False`)
    AND `config.role_aware_uncond_attention` is True (the default). An A/B comparison
    against real pretrained checkpoints confirmed a clean win on LM1B (Gen. PPL 55.89 ->
    38.16 at 1000 samples, 64 sampling steps, repetition rate essentially unchanged) --
    but on OWT (max_length=1024, 64 commit rounds), role-aware attention was found to
    *increase* repetition-loop collapse despite a lower Gen. PPL (that PPL drop is a
    known GPT-2-scoring artifact of repeated text, not real quality), so OWT's own
    config sets `role_aware_uncond_attention: false` to keep the legacy (fully-unmasked)
    behavior there until the long-canvas regression is understood. When enabled,
    `role_clean_mask` is set to this round's committed-position mask and
    `attention_mask` to an all-valid mask (every position in the unconditional canvas is
    either committed or active -- there is no separate "beyond the canvas" padding
    region here, unlike the conditional task's variable source length).
    `use_staggered_noise=True` checkpoints always fall back to legacy attention
    regardless of the config flag, since the role_clean_mask semantics it relies on
    don't apply to that training regime.

    Returns:
        List of `batch_size` decoded strings (special tokens stripped).
    """
    use_role_aware = (
        config.use_block_revelation
        and not config.use_staggered_noise
        and getattr(config, "role_aware_uncond_attention", True)
    )
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id
    block_size = config.block_size
    canvas_length = config.max_length

    model_params = state.ema_params1 if use_ema else state.params
    model_apply_fn = state.apply_fn
    cfg_scale = sampling_config.cfgs[0]
    self_cond_cfg_scale = sampling_config.self_cond_cfg_scales[0]

    pos_tgt = jnp.arange(canvas_length)[None, :]
    is_source = jnp.zeros((batch_size, canvas_length), dtype=bool)

    committed_target_ids = jnp.full((batch_size, canvas_length), pad_token_id, dtype=jnp.int32)
    committed_len = jnp.zeros((batch_size,), dtype=jnp.int32)
    finished = jnp.zeros((batch_size,), dtype=bool)

    assert decoder_mode in ("greedy", "categorical", "top_p"), (
        f"generate_block_uncond: unsupported decoder_mode={decoder_mode!r}"
    )

    max_rounds = (canvas_length + block_size - 1) // block_size
    for round_idx in range(max_rounds):
        if bool(jnp.all(finished | (committed_len >= canvas_length))):
            break

        is_committed = pos_tgt < committed_len[:, None]
        encoder_attn, _, clean_mask = build_committed_context(
            is_source, is_committed, is_committed,
        )

        rng, enc_rng, noise_rng, t_rng = jax.random.split(rng, 4)
        cond_seq = encode_text(
            input_ids=committed_target_ids, attention_mask=encoder_attn,
            encoder_apply_fn=encoder_apply_fn, encoder_params=encoder_params,
            latent_mean=config.latent_mean, latent_std=config.latent_std,
        )
        d_model = cond_seq.shape[-1]
        z0 = jax.random.normal(noise_rng, (batch_size, canvas_length, d_model)) * config.denoiser_noise_scale

        t_steps = get_sampling_steps(
            t_rng, n_steps=sampling_config.num_sampling_steps[0],
            time_schedule=sampling_config.time_schedule,
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        )
        extra_sampler_kwargs = {}
        if use_role_aware:
            # Every position in the unconditional canvas is either committed or active
            # -- unlike the conditional task's `target_capacity`, there's no separate
            # "beyond the real content" padding region here, so the validity mask is
            # trivially all-ones. Do NOT reuse build_committed_context's discarded
            # `_` (committed-only) return value here: that marks active positions
            # invalid, which would incorrectly block active-to-active attention.
            denoiser_valid_mask = jnp.ones((batch_size, canvas_length), dtype=jnp.float32)
            extra_sampler_kwargs = dict(
                attention_mask=denoiser_valid_mask,
                role_clean_mask=clean_mask,
            )
        z_final = _generate_samples_single_batch(
            model_params=model_params, model_apply_fn=model_apply_fn, rng=rng,
            z=z0, t_steps=t_steps, cond_seq=cond_seq, cond_seq_mask=clean_mask,
            config=config, sampling_config=sampling_config,
            cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            **extra_sampler_kwargs,
        )
        if decoder_mode == "greedy":
            # Exact original path -- no extra rng split consumed, so this branch's RNG
            # stream (and therefore every subsequent round's noise/timestep sampling)
            # is bit-for-bit identical to before decoder_mode existed.
            predicted_ids = _dlm_decode_batch(
                z=z_final, model_params=model_params, model_apply_fn=model_apply_fn,
                t_final_val=t_steps[-1], config=config, self_cond_cfg_scale=self_cond_cfg_scale,
            )
        else:
            # OWT-only opt-in sampling. Restrict the categorical/top-p sort cost to
            # this round's block_size-wide newly-committed window -- positions outside
            # it are discarded by the `jnp.where(newly_committed, ...)` logic below and
            # by the final committed_len cleanup regardless of what's sampled there, so
            # there's no correctness reason to pay the O(canvas_length) sampling cost.
            rng, decode_rng = jax.random.split(rng)
            decoder_logits = _dlm_decode_logits_batch(
                z=z_final, model_params=model_params, model_apply_fn=model_apply_fn,
                t_final_val=t_steps[-1], config=config, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            window_start = round_idx * block_size
            window_end = min(window_start + block_size, canvas_length)
            sampled_window_ids = _sample_decoder_tokens(
                decoder_logits[:, window_start:window_end, :], decode_rng,
                mode=decoder_mode, temperature=decoder_temperature, top_p=decoder_top_p,
            )
            predicted_ids = jnp.zeros((batch_size, canvas_length), dtype=jnp.int32)
            predicted_ids = predicted_ids.at[:, window_start:window_end].set(sampled_window_ids)

        # Commit the next `block_size` tokens of the active region (left-to-right).
        commit_upto = jnp.minimum(committed_len + block_size, canvas_length)
        newly_committed = (pos_tgt >= committed_len[:, None]) & (pos_tgt < commit_upto[:, None])
        committed_target_ids = jnp.where(newly_committed, predicted_ids, committed_target_ids)

        # Same left-to-right-contiguous-commit argument as `generate_block_cond`: the
        # first EOS among this round's newly committed tokens is the true end position.
        is_eos = (committed_target_ids == eos_token_id) & (pos_tgt < commit_upto[:, None])
        has_eos = jnp.any(is_eos, axis=1)
        first_eos_pos = jnp.argmax(is_eos, axis=1)
        committed_len = jnp.where(finished, committed_len, commit_upto)
        committed_len = jnp.where((~finished) & has_eos, first_eos_pos + 1, committed_len)
        finished = finished | has_eos
        committed_target_ids = jnp.where(
            pos_tgt < committed_len[:, None], committed_target_ids, pad_token_id,
        )

    texts = []
    committed_len_np = np.array(committed_len)
    committed_target_ids_np = np.array(committed_target_ids)
    for i in range(batch_size):
        ids = committed_target_ids_np[i, :committed_len_np[i]]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def _setup_generation(state, config, batch_size, header):
    """Shared setup: log header, unreplicate state, build replicated model_params, compute batch sizes."""
    log_for_0("\n" + "=" * 70)
    log_for_0(header)
    log_for_0("=" * 70)

    num_local_devices = jax.local_device_count()
    log_for_0(f"Using {num_local_devices} local devices for generation")

    state_unreplicated = jax_utils.unreplicate(state)
    model_apply_fn = state_unreplicated.apply_fn

    encoder_config, _, _ = get_encoder(config.encoder_model_name, None)
    d_model = encoder_config.d_model

    model_params_replicated = jax_utils.replicate(state_unreplicated.ema_params1)

    per_device_batch = max(1, batch_size // num_local_devices)
    effective_batch_size = per_device_batch * num_local_devices
    log_for_0(f"Per-device batch size: {per_device_batch}, effective batch size: {effective_batch_size}")

    return state_unreplicated, model_apply_fn, model_params_replicated, d_model, num_local_devices, effective_batch_size


