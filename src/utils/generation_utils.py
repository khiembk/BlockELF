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
):
    """Create a scan-compatible step function.

    For method == "sde", `rng` must be provided and the scan carry must include a step index
    (z, x_pred, step_idx); fold_in is done per step. Other methods use a (z, x_pred) carry.
    """
    method = sampling_config.sampling_method
    base_kwargs = dict(
        model_apply_fn=model_apply_fn, model_params=model_params,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
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
) -> Array:
    """Generate samples for a single batch (pmap-compatible, uses lax.scan)."""
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


def _dlm_decode_batch(z, model_params, model_apply_fn, t_final_val, config, self_cond_cfg_scale):
    """Decode z→tokens with the DLM decoder head."""
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
    return jnp.argmax(decoder_logits, axis=-1)


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
        z_final = _generate_samples_single_batch(
            model_params=model_params, model_apply_fn=model_apply_fn, rng=rng,
            z=z0, t_steps=t_steps, cond_seq=cond_seq, cond_seq_mask=clean_mask,
            config=config, sampling_config=sampling_config,
            cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
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
):
    """Block-to-block unconditional generation (single device, eager Python loop).

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

    Returns:
        List of `batch_size` decoded strings (special tokens stripped).
    """
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

    max_rounds = (canvas_length + block_size - 1) // block_size
    for _ in range(max_rounds):
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
        z_final = _generate_samples_single_batch(
            model_params=model_params, model_apply_fn=model_apply_fn, rng=rng,
            z=z0, t_steps=t_steps, cond_seq=cond_seq, cond_seq_mask=clean_mask,
            config=config, sampling_config=sampling_config,
            cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        )
        predicted_ids = _dlm_decode_batch(
            z=z_final, model_params=model_params, model_apply_fn=model_apply_fn,
            t_final_val=t_steps[-1], config=config, self_cond_cfg_scale=self_cond_cfg_scale,
        )

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


