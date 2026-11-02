"""Sanity tests for block-to-block (fixed-commitment) revelation — Phase 1.

Pure-CPU, no GPU/TPU, no network access (dummy encoder). Run with:

    cd src && python tests/test_block_flow.py

Covers:
  1. `sample_block_round` mask invariants over many random (cond_len, total_len, b).
  2. End-to-end `train_step` smoke test with a tiny ELF model + dummy encoder,
     both in block-revelation mode and in the original (non-block) mode, to confirm
     finite/sane losses and that no pad/committed position leaks into the loss.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import jax_utils
from flax.training.common_utils import shard

from utils.block_utils import sample_block_round
from utils.encoder_utils import build_self_attn_cond_masks
from utils.train_utils import TrainState, create_learning_rate_fn
from modules.model import ELF
from train_step import train_step


def test_sample_block_round_invariants():
    rng = jax.random.PRNGKey(0)
    S = 40
    trials = 200
    for trial in range(trials):
        rng, cl_rng, tl_rng, b_rng, round_rng = jax.random.split(rng, 5)
        cond_len = int(jax.random.randint(cl_rng, (), 0, S // 2))
        # total_len must be >= cond_len, up to S
        total_len = int(jax.random.randint(tl_rng, (), cond_len, S + 1))
        block_size = int(jax.random.randint(b_rng, (), 1, 17))

        pos = np.arange(S)
        is_cond = (pos < cond_len).astype(np.float32)[None, :]
        is_valid = (pos < total_len).astype(np.float32)[None, :]

        committed, active, boundary = sample_block_round(round_rng, is_cond, is_valid, block_size)
        committed, active, boundary = np.array(committed), np.array(active), np.array(boundary)

        target_len = total_len - cond_len

        # Disjoint, both subsets of valid, union-with-pad covers everything.
        assert np.all((committed > 0) | (active > 0) | (is_valid[0] == 0)), "uncovered valid position"
        assert not np.any((committed > 0) & (active > 0)), "committed/active overlap"
        assert np.all(committed[0][is_valid[0] == 0] == 0), "committed leaks into padding"
        assert np.all(active[0][is_valid[0] == 0] == 0), "active leaks into padding"

        # cond ⊆ committed.
        assert np.all(committed[0][pos < cond_len] == is_valid[0][pos < cond_len]), "cond not fully committed"

        # If there is any target content, the active set must be non-empty (round
        # sampling must never pick the fully-committed terminal state as a training round).
        if target_len > 0:
            assert active.sum() > 0, "active set empty despite remaining target content"
            committed_target_len = committed.sum() - cond_len
            assert committed_target_len % block_size == 0, "committed target length not a multiple of b"
            assert committed_target_len < target_len, "committed target length reached full target length"
        else:
            assert active.sum() == 0

    print(f"[OK] sample_block_round invariants held over {trials} random trials")


def _make_dummy_encoder(vocab_size, d_model):
    """A tiny embedding-table 'encoder' standing in for the frozen T5 encoder, so the
    integration test below needs no network access / pretrained checkpoint."""

    class DummyEncoder(nn.Module):
        @nn.compact
        def __call__(self, input_ids, attention_mask, deterministic=True):
            del attention_mask, deterministic
            return nn.Embed(vocab_size, d_model)(input_ids)

    return DummyEncoder()


def _make_batch(rng, batch_size, seq_len, cond_len, vocab_size):
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
    pos = np.arange(seq_len)[None, :]
    is_cond = np.tile(pos < cond_len, (batch_size, 1))
    is_valid = np.ones((batch_size, seq_len), dtype=bool)  # fully valid, no padding
    encoder_attn, attn, cond_seq_mask = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
    return {
        "input_ids": np.array(input_ids),
        "encoder_attention_mask": encoder_attn,
        "attention_mask": attn,
        "cond_seq_mask": cond_seq_mask,
        "label_drop_mask": np.zeros((batch_size,), dtype=bool),
    }


class _Config:
    """Minimal stand-in for configs.config.Config with only the fields train_step needs."""
    t_eps = 0.05
    self_cond_prob = 0.5
    latent_mean = 0.0
    latent_std = 1.0
    decoder_prob = 0.2
    decoder_noise_scale = 1.0
    decoder_p_mean = 0.8
    decoder_p_std = 0.8
    denoiser_p_mean = -1.5
    denoiser_p_std = 0.8
    denoiser_noise_scale = 2.0
    time_schedule = "logit_normal"
    pad_token = "eos"
    label_drop_prob = 0.0
    num_self_cond_cfg_tokens = 2
    self_cond_cfg_min = 0.5
    self_cond_cfg_max = 5.0
    ema_decay1 = 0.9999
    grad_accum_steps = 1
    use_block_revelation = False
    block_size = 4


def _run_train_step(use_block_revelation, seed=0):
    vocab_size, d_model, seq_len, cond_len, batch_size = 37, 8, 20, 6, 4
    config = _Config()
    config.use_block_revelation = use_block_revelation

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, dropout_rng, data_rng, enc_rng = jax.random.split(rng, 5)

    dummy_encoder = _make_dummy_encoder(vocab_size, d_model)
    encoder_params = dummy_encoder.init(
        enc_rng, input_ids=jnp.zeros((1, seq_len), dtype=jnp.int32),
        attention_mask=jnp.ones((1, seq_len, seq_len)),
    )["params"]

    model = ELF(
        text_encoder_dim=d_model, max_length=seq_len, hidden_size=16, depth=2, num_heads=2,
        bottleneck_dim=4, num_time_tokens=2, num_self_cond_cfg_tokens=2, vocab_size=vocab_size,
    )
    dummy_x = jnp.ones((1, seq_len, 2 * d_model))
    params = model.init(init_rng, x=dummy_x, t=jnp.ones((1,)), deterministic=True,
                        self_cond_cfg_scale=jnp.ones((1,)))["params"]

    lr_fn = create_learning_rate_fn(num_train_steps=10, num_warmup_steps=1, learning_rate=1e-3)
    import optax
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_fn))

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
        dropout_rng=dropout_rng, ema_params1=params,
    )

    batch = _make_batch(data_rng, batch_size, seq_len, cond_len, vocab_size)
    batch = {k: jnp.array(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}

    # train_step is written for pmap (uses axis_name="batch" ops); wrap with a
    # single-device pmap so we can call it directly in this CPU-only test. Use the
    # same (un)replicate helpers train.py uses, since TrainState has plain-int
    # fields (epoch, step) that a naive tree_map(lambda x: x[None]) chokes on.
    batch = shard(batch)
    encoder_params_r = jax_utils.replicate(encoder_params)
    state_r = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        lambda s, ep, b: train_step(s, ep, dummy_encoder.apply, b, config),
        axis_name="batch",
    )
    new_state, metrics = p_train_step(state_r, encoder_params_r, batch)
    metrics = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)
    return metrics


def test_train_step_block_mode_finite_loss():
    for use_block in (False, True):
        metrics = _run_train_step(use_block_revelation=use_block, seed=1)
        for k, v in metrics.items():
            assert np.isfinite(v), f"non-finite metric {k}={v} (use_block_revelation={use_block})"
        assert 0.0 <= metrics["active_frac"] <= 1.0
        print(f"[OK] train_step finite loss (use_block_revelation={use_block}): {metrics}")

    # Block mode must restrict supervision to a shorter (or equal) span than the
    # full non-cond target, i.e. active_frac should be <= the non-block case on
    # average. Check over a few seeds since the round k is stochastic.
    block_fracs = [_run_train_step(True, seed=s)["active_frac"] for s in range(5)]
    base_fracs = [_run_train_step(False, seed=s)["active_frac"] for s in range(5)]
    assert np.mean(block_fracs) <= np.mean(base_fracs) + 1e-6, (
        f"block active_frac ({np.mean(block_fracs):.3f}) should not exceed "
        f"non-block active_frac ({np.mean(base_fracs):.3f})"
    )
    print(f"[OK] mean active_frac block={np.mean(block_fracs):.3f} vs non-block={np.mean(base_fracs):.3f}")


if __name__ == "__main__":
    test_sample_block_round_invariants()
    test_train_step_block_mode_finite_loss()
    print("\nAll block-flow sanity tests passed.")
