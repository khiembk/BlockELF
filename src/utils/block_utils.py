"""Fixed-size block revelation for Stage-1 block-to-block generation.

On revelation round k, the committed target prefix is {1, ..., min(k*b, L)}; every
remaining ("active") target position is noised and supervised this round, matching
Algorithm 1 / 2 of the Adaptive BlockFlow spec (docs/main.tex). Unlike the full
Gaussian-to-categorical reformulation in that spec, this reuses ELF's existing
continuous-embedding flow: committed positions (source + already-committed target)
are treated exactly like the existing clean "cond" tokens, just extended to include
the committed target prefix.
"""

import jax
import jax.numpy as jnp


def sample_block_round(rng, cond_seq_mask, attention_mask, block_size: int):
    """Sample a per-example committed/active split for one block-revelation round.

    Args:
        rng: JAX PRNGKey.
        cond_seq_mask: (B, S) float/bool, 1 at source ("cond") positions.
        attention_mask: (B, S) float/bool, 1 at any valid (non-pad) position.
        block_size: fixed commit size b (>=1).

    Returns:
        committed_mask: (B, S) float32, 1 at source + committed-target positions
            (i.e. positions that stay clean this round; cond ⊆ committed).
        active_mask: (B, S) float32, 1 at target positions to noise/supervise this
            round. Always disjoint from committed_mask and from padding.
        boundary: (B,) float32, absolute index of the first active position
            (cond_len + k*block_size). Exposed for logging/debugging only.
    """
    cond_seq_mask = cond_seq_mask.astype(jnp.float32)
    attention_mask = attention_mask.astype(jnp.float32)
    is_valid = attention_mask > 0

    cond_len = cond_seq_mask.sum(axis=1)            # (B,)
    total_len = attention_mask.sum(axis=1)           # (B,)
    target_len = jnp.maximum(total_len - cond_len, 0.0)
    target_len_safe = jnp.maximum(target_len, 1.0)

    # K_i = ceil(target_len_i / b) rounds; k sampled uniformly from [0, K_i).
    num_rounds = jnp.ceil(target_len_safe / block_size)
    u = jax.random.uniform(rng, cond_len.shape)
    k = jnp.floor(u * num_rounds)
    committed_target_len = k * block_size            # < target_len_safe whenever target_len > 0

    boundary = cond_len + committed_target_len        # (B,)
    pos = jnp.arange(cond_seq_mask.shape[1], dtype=jnp.float32)[None, :]  # (1, S)

    committed_mask = ((pos < boundary[:, None]) & is_valid).astype(jnp.float32)
    active_mask = ((pos >= boundary[:, None]) & is_valid).astype(jnp.float32)
    return committed_mask, active_mask, boundary


def _staggered_bk(desired_block_size: float, length: float, max_block_size: float = None):
    """(b, k) schedule-shape parameters, ported exactly from the reference `setdlms`
    repo's `StaggeredNoise.init_schedule` (src/noise_schedule/noise_schedules.py),
    `desired_block_size not in [1, max_block_size]` branch — the one actually used by
    its LM1B training config (`configs/noise/staggered.yaml` +
    `bash_scripts/run_train_setdlm_lm1b.sh`, which sets `desired_block_size=4`,
    `max_block_size=2*desired_block_size`, `length=128`). `b` is the window width
    (in timestep units) over which one position transitions from fully-noisy to
    fully-resolved; `k` is the power-law shape of that transition. Both are derived
    deterministically from `desired_block_size`/`length` — there is no free "intensity"
    knob to tune, unlike this repo's first (abandoned) stagger_scale attempt.
    """
    if max_block_size is None:
        max_block_size = 2.0 * desired_block_size
    desired_num_blocks = length / desired_block_size
    desired_area = 1.0 / (2.0 * desired_num_blocks)
    b = max_block_size / (length + max_block_size - 1.0)
    k = desired_area / (b - desired_area)
    return b, k


def compute_staggered_timesteps(t_global, cond_seq_mask, attention_mask, block_size: int, target_capacity: int):
    """Per-position effective timestep: a faithful port of SetDLM's `StaggeredNoise`
    (see `_staggered_bk` and the reference repo's `noise_schedules.py`), re-derived for
    this repo's OPPOSITE time convention (t=0 noise -> t=1 clean; SetDLM's masked
    discrete diffusion uses t=0 clean -> t=1 masked).

    Unlike a discrete per-round reveal (`sample_block_round`), there is no "committed
    boundary" sampled at all: every target position gets its own continuous effective
    timestep, smoothly and monotonically increasing left-to-right across the *entire*
    target region, as a function of (a) its absolute position within that region and
    (b) one globally-sampled `t_global` (same scalar `sample_timesteps` already draws).
    A single training step therefore supervises positions at every stage of
    revelation — from untouched noise far from the start to fully-resolved right at the
    start — simultaneously, which is the actual SetDLM mechanism (their title:
    "interpolating token orderings between autoregression and diffusion"), not a
    localized approximation of it.

    Positions exactly at the start of the target region reach t_pos=1 (fully resolved)
    almost immediately as t_global grows past ~0; positions at the far end only start
    resolving once t_global approaches 1. Source ("cond") positions are unaffected here
    — they're forced clean via the existing `cond_seq_mask` override already passed to
    `add_noise`, same as always.

    v3 fix: the position-rank used to place each position on the [0, 1-b] schedule
    range is normalized by that example's OWN real target length (from
    `attention_mask`), not the fixed `target_capacity`. The reference repo's `length`
    hyperparameter assumes every example's canvas is actually filled with content
    (true for packed/wrapped corpora); LM1B sentences are short (mean ~35 tokens) and
    are tokenized/used one-per-example (not packed) into a 128-token canvas, so most of
    each example's "target region" is pad, not real content. Normalizing by the fixed
    target_capacity crushed a typical short example's entire real content into a tiny
    initial slice of the schedule (e.g. loc in [0, 0.21] for a 35-token example, out of
    the full [0, 0.80] range) while padding occupied the unused rest — and, since the
    crushed fraction depends on each example's own length, different examples
    effectively trained under very different schedules despite sharing the same
    `t_global` distribution. That inconsistency (not just generic exposure bias) is the
    most likely cause of the checkpoint-to-checkpoint quality oscillation seen in the
    first version of this function. `b`/`k` (the schedule's width/shape) are still
    derived from the fixed `target_capacity`, matching the reference repo's intended
    block_size granularity; only the position normalization is now per-example.

    Args:
        t_global: (B,) per-example timestep, as already sampled by sample_timesteps.
        cond_seq_mask: (B, S) float/bool, 1 at source ("cond") positions.
        attention_mask: (B, S) float/bool, 1 at any valid (non-pad) position — used to
            derive each example's real (unpadded) target length.
        block_size: fixed commit size b (>=1) — plays the role of SetDLM's
            `desired_block_size` (their published LM1B config matches its reveal
            granularity to this exact parameter).
        target_capacity: the maximum possible target-region length (e.g.
            `max_length - max_input_length`, or `max_length` when unconditional) —
            plays the role of SetDLM's fixed `length` hyperparameter (used only for the
            `b, k` schedule-shape derivation now, not for position normalization).

    Returns:
        t_pos: (B, S) float32 per-position effective timestep.
    """
    b, k = _staggered_bk(float(block_size), float(target_capacity))

    cond_len = cond_seq_mask.astype(jnp.float32).sum(axis=1)  # (B,)
    total_len = attention_mask.astype(jnp.float32).sum(axis=1)  # (B,)
    target_len = jnp.maximum(total_len - cond_len, 1.0)  # (B,) real (unpadded) target length

    pos = jnp.arange(cond_seq_mask.shape[1], dtype=jnp.float32)[None, :]  # (1, S)
    rank_in_target = pos - cond_len[:, None]  # 0 at first target position, grows rightward
    rank_norm = jnp.clip(rank_in_target / jnp.maximum(target_len[:, None] - 1.0, 1.0), 0.0, 1.0)
    loc = rank_norm * (1.0 - b)

    x = jnp.clip((t_global[:, None] - loc) / b, 0.0, 1.0)
    return x ** k


def build_committed_context(is_source, is_committed_target, is_valid, xp=jnp):
    """Build (encoder_attention_mask, attention_mask, clean_mask) for a fixed committed
    prefix, mirroring `encoder_utils.build_self_attn_cond_masks` but with "clean"
    covering source ∪ committed target instead of just source. Used at inference time
    when re-encoding the growing committed context each revelation round.
    """
    is_clean = is_source | is_committed_target
    encoder_attention_mask = (
        (is_clean[:, :, None] & is_clean[:, None, :]) |
        (~is_clean[:, :, None] & is_valid[:, None, :])
    ).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    clean_mask = is_clean.astype(xp.float32)
    return encoder_attention_mask, attention_mask, clean_mask
