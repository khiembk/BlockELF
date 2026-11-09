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
    """(b, k) schedule-shape parameters, ported from the reference `setdlms` repo's
    `StaggeredNoise.init_schedule` (`desired_block_size not in [1, max_block_size]` branch).
    `b` is the window width (in timestep units) over which one position transitions from
    fully-noisy to fully-resolved; `k` is the power-law shape of that transition. Both are
    derived deterministically from `desired_block_size`/`length` — no free "intensity" knob
    (unlike this repo's first, abandoned `sample_stagger_deltas` attempt, which used a
    discrete per-chunk offset with a free `stagger_strength` knob and caused severe
    training instability when tried on LM1B: 93/100 samples collapsed into single-word
    repetition loops by checkpoint_32000 at stagger_strength=0.5).
    """
    if max_block_size is None:
        max_block_size = 2.0 * desired_block_size
    desired_num_blocks = length / desired_block_size
    desired_area = 1.0 / (2.0 * desired_num_blocks)
    b = max_block_size / (length + max_block_size - 1.0)
    k = desired_area / (b - desired_area)
    return b, k


def compute_staggered_timesteps(t_global, cond_seq_mask, attention_mask, block_size: int, target_capacity: int):
    """Per-position effective-timestep offset from the shared scalar `t_global`: a faithful
    port of SetDLM's `StaggeredNoise` (see `_staggered_bk`), re-derived for this repo's
    OPPOSITE time convention (t=0 noise -> t=1 clean). Every *target* position gets its own
    continuous effective timestep, smoothly and monotonically increasing left-to-right
    across the whole target region, as a function of (a) its absolute rank within that
    region and (b) `t_global`. A single training step therefore supervises positions at
    every stage of revelation simultaneously, instead of needing many separate batches
    (across different sampled rounds) to cover the full revelation spectrum.

    Source ("cond") positions always get delta=0 (i.e. t_position == t_global exactly, the
    same as when staggering is off): unlike LM1B (unconditional, no source tokens at all),
    this repo's WMT task has real source positions that are always clean, so there is no
    "effective resolution" information to add there — forcing delta=0 keeps behavior
    identical to the non-staggered path at every cond position, by construction rather
    than as an accidental side effect of rank-clipping.

    Args:
        t_global: (B,) per-example timestep, as already sampled by sample_timesteps.
        cond_seq_mask: (B, S) float/bool, 1 at source ("cond") positions.
        attention_mask: (B, S) float/bool, 1 at any valid (non-pad) position — used to
            derive each example's real (unpadded) target length.
        block_size: fixed commit size b (>=1) — matches this repo's existing
            use_block_revelation granularity.
        target_capacity: the maximum possible target-region length (max_length -
            max_input_length) — only used for the (b, k) schedule-shape derivation, not
            for position normalization (each example's rank is normalized by its own real
            target length, not this fixed capacity — otherwise short examples get their
            real content crushed into a small initial slice of the schedule while padding
            wastes the rest, and different examples end up training under inconsistent
            schedules; this was v2's bug on the LM1B port of this scheme, fixed in v3).

    Returns:
        delta: (B, S) float32, t_position - t_global. 0 at cond positions and wherever
            t_position would be clipped back to t_global; add to t_global and clip to
            [0, 1] to get the absolute per-position effective timestep.
    """
    b, k = _staggered_bk(float(block_size), float(target_capacity))

    cond_seq_mask = cond_seq_mask.astype(jnp.float32)
    cond_len = cond_seq_mask.sum(axis=1)  # (B,)
    total_len = attention_mask.astype(jnp.float32).sum(axis=1)  # (B,)
    target_len = jnp.maximum(total_len - cond_len, 1.0)  # (B,) real (unpadded) target length

    pos = jnp.arange(cond_seq_mask.shape[1], dtype=jnp.float32)[None, :]  # (1, S)
    rank_in_target = pos - cond_len[:, None]  # 0 at first target position, grows rightward
    rank_norm = jnp.clip(rank_in_target / jnp.maximum(target_len[:, None] - 1.0, 1.0), 0.0, 1.0)
    loc = rank_norm * (1.0 - b)

    x = jnp.clip((t_global[:, None] - loc) / b, 0.0, 1.0)
    t_position = x ** k

    delta = t_position - t_global[:, None]
    return delta * (1.0 - cond_seq_mask)


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
