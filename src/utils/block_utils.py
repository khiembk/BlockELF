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


def compute_staggered_timesteps(t_global, active_mask, boundary, block_size: int, stagger_scale: float):
    """Per-position effective timestep for staggered-noise training (SetDLM-style
    single-pass multi-round curriculum, adapted to continuous flow matching).

    Instead of noising every active position at the same global `t_global` (one
    (committed boundary, noise level) pair per training example — "one round per
    step"), bias positions near the commit boundary toward a *less noisy* (larger)
    effective timestep, decaying back to the plain `t_global` by `block_size`
    positions out. A position right after the boundary (about to be committed next)
    is thus trained more like "nearly resolved", while a position `block_size` or
    more positions out is trained exactly as before — packing signal from many
    conceptual revelation rounds into one forward pass instead of needing a separate
    training step per round. Sampling/inference is unaffected (still one global `t`
    per round, per the Algorithm 2 contract) — this is training-only.

    Args:
        t_global: (B,) per-example timestep, as already sampled by sample_timesteps.
        active_mask: (B, S) float/bool, 1 at target positions to noise/supervise
            this round (the `active_mask_2d` from sample_block_round).
        boundary: (B,) float, absolute index of the first active position
            (sample_block_round's third return value).
        block_size: fixed commit size b (>=1) — the staggering window.
        stagger_scale: max additive timestep boost at the boundary (0 disables
            staggering, recovering the original uniform-t behavior exactly).

    Returns:
        t_pos: (B, S) float32 per-position effective timestep. 1.0 (fully resolved)
            at non-active (source/committed/pad) positions.
    """
    active_mask = active_mask.astype(jnp.float32)
    pos = jnp.arange(active_mask.shape[1], dtype=jnp.float32)[None, :]  # (1, S)
    dist = pos - boundary[:, None]  # (B, S); 0 at first active position, grows rightward
    offset = stagger_scale * jnp.clip(1.0 - dist / block_size, 0.0, 1.0)
    t_pos_active = jnp.clip(t_global[:, None] + offset, 0.0, 1.0)
    return jnp.where(active_mask > 0, t_pos_active, jnp.ones_like(t_pos_active))


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
