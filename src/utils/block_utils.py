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
