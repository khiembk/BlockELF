"""Sanity tests for block-to-block (fixed-commitment) revelation — Phase 1.

Pure-CPU, no GPU/TPU, no network access (dummy encoder). Run with:

    cd src && python tests/test_block_flow.py

Covers:
  1. `sample_block_round` mask invariants over many random (cond_len, total_len, b).
  2. End-to-end `train_step` smoke test with a tiny ELF model + dummy encoder,
     both in block-revelation mode and in the original (non-block) mode, to confirm
     finite/sane losses and that no pad/committed position leaks into the loss.
  3. `compute_block_round_counts` + the per-example K_i-weighted resampling it enables:
     confirms P(K_i=j | round=0 sampled) matches the true marginal P(K_i=j) under the
     weighted scheme, where the naive uniform-per-example scheme is biased toward short
     examples (see train.py's `sample_weights` wiring and data_utils.py docstring).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import collections

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from datasets import Dataset
from flax import jax_utils
from flax.training.common_utils import shard

from utils.block_utils import (
    sample_block_round, compute_staggered_timesteps, _staggered_bk, build_committed_context,
)
from utils.data_utils import compute_block_round_counts
from utils.encoder_utils import build_self_attn_cond_masks, encode_text
from utils.generation_utils import pack_source_target_tight, extract_target_relative
from utils.train_utils import TrainState, create_learning_rate_fn
from modules.model import ELF, ELFBlock
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


def _make_attentive_dummy_encoder(vocab_size, d_model):
    """A tiny single-layer self-attention 'encoder' that actually MIXES information
    across positions via `attention_mask` (unlike `_make_dummy_encoder`, a pure
    per-token embedding lookup that ignores `attention_mask` entirely and so can never
    expose a masking bug). Needed to test Fix A: whether varying token IDs at
    mask-disallowed positions changes the hidden state at other positions."""

    class AttentiveDummyEncoder(nn.Module):
        @nn.compact
        def __call__(self, input_ids, attention_mask, deterministic=True):
            del deterministic
            x = nn.Embed(vocab_size, d_model)(input_ids)
            q = nn.Dense(d_model, name="q")(x)
            k = nn.Dense(d_model, name="k")(x)
            v = nn.Dense(d_model, name="v")(x)
            scores = jnp.einsum("bqd,bkd->bqk", q, k) / jnp.sqrt(d_model)
            # attention_mask here is always the 3-D (B, Q, K) role mask this test
            # constructs via build_committed_context/build_self_attn_cond_masks.
            bias = jnp.where(attention_mask > 0, 0.0, -1e9)
            attn = jax.nn.softmax(scores + bias, axis=-1)
            return jnp.einsum("bqk,bkd->bqd", attn, v)

    return AttentiveDummyEncoder()


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
    use_staggered_noise = False
    max_length = 20
    max_input_length = None


def _run_train_step(use_block_revelation, seed=0, use_staggered_noise=False, return_state=False,
                     decoder_prob=None, num_steps=1):
    vocab_size, d_model, seq_len, cond_len, batch_size = 37, 8, 20, 6, 4
    config = _Config()
    config.use_block_revelation = use_block_revelation
    config.use_staggered_noise = use_staggered_noise
    if decoder_prob is not None:
        config.decoder_prob = decoder_prob

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
    init_kwargs = dict(x=dummy_x, t=jnp.ones((1,)), deterministic=True,
                        self_cond_cfg_scale=jnp.ones((1,)))
    if use_block_revelation and use_staggered_noise:
        init_kwargs["t_pos"] = jnp.ones((1, seq_len))
    params = model.init(init_rng, **init_kwargs)["params"]

    # num_warmup_steps=0: this test only ever takes a single step (step=0), and
    # optax.linear_schedule returns exactly init_value=0.0 at step 0 whenever
    # num_warmup_steps>=1 — i.e. every param (not just the one under test) would see
    # zero effective LR and never move, regardless of gradient correctness.
    lr_fn = create_learning_rate_fn(num_train_steps=10, num_warmup_steps=0, learning_rate=1e-3)
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
    # FinalLayer's output kernel is zero-initialized (see modules/layers.py) — at a
    # freshly-initialized model this makes the Jacobian of the output w.r.t. *any*
    # upstream hidden state exactly zero, so gradient flow into any upstream param
    # (pos_t_embedder, t_embedder, text_proj, ...) is legitimately all-zero on the
    # very first step, regardless of correctness. Only final_layer's own weights move
    # on step 0; take `num_steps` so a gradient-flow check can look at the LAST step
    # (by which point final_layer is no longer all-zero).
    prev_state_r = state_r
    for _ in range(num_steps):
        prev_state_r = state_r
        state_r, metrics = p_train_step(state_r, encoder_params_r, batch)
    new_state = state_r
    metrics = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)
    if return_state:
        return metrics, jax_utils.unreplicate(prev_state_r), jax_utils.unreplicate(new_state)
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


def test_fix_a_no_future_leak_into_committed_encoding():
    """Fix A regression test: the frozen T5 encoder's committed-prefix embeddings must
    never depend on token IDs at as-yet-unrevealed ("active") target positions.

    Before the fix, `train_step.py` called `encode_text` with the *default*
    cond-sees-cond/target-sees-all mask BEFORE `sample_block_round` even ran, so
    "committed" positions' embeddings were contextualized with bidirectional access to
    the entire (including active) target region -- something inference's
    `build_committed_context` (re-encoding only actually-generated tokens each round)
    can never replicate. The fix: sample the committed/active split first and build the
    encoder mask via `build_committed_context` (clean queries see only clean keys).

    Uses `_make_attentive_dummy_encoder` (real masked self-attention, unlike the
    per-token-only `_make_dummy_encoder` used elsewhere) so this test can actually
    distinguish "masking respected" from "masking ignored."
    """
    vocab_size, d_model, seq_len = 37, 8, 20
    rng = jax.random.PRNGKey(0)
    rng, enc_rng = jax.random.split(rng)
    encoder = _make_attentive_dummy_encoder(vocab_size, d_model)
    encoder_params = encoder.init(
        enc_rng, input_ids=jnp.zeros((1, seq_len), dtype=jnp.int32),
        attention_mask=jnp.ones((1, seq_len, seq_len)),
    )["params"]

    def encode(input_ids, is_source, is_committed_target, is_valid):
        encoder_attention_mask, _, _ = build_committed_context(
            is_source, is_committed_target, is_valid, xp=jnp,
        )
        return encode_text(
            input_ids=input_ids, attention_mask=encoder_attention_mask,
            encoder_apply_fn=encoder.apply, encoder_params=encoder_params,
            latent_mean=0.0, latent_std=1.0,
        )

    pos = np.arange(seq_len)
    is_valid = jnp.ones((1, seq_len), dtype=bool)

    # (cond_len, committed_len) cases: LM1B-like (cond_len=0) and WMT14-like
    # (cond_len>0); committed_len=0 ("k=0"), middle, and near-last round.
    cases = [
        (0, 0), (0, 4), (0, 16),      # LM1B-like: no source at all
        (6, 0), (6, 4), (6, 10),      # WMT14-like: source + committed
    ]
    for cond_len, committed_len in cases:
        assert cond_len + committed_len <= seq_len
        is_source = jnp.array(pos < cond_len)[None, :]
        is_committed_target = jnp.array(
            (pos >= cond_len) & (pos < cond_len + committed_len)
        )[None, :]
        is_clean_np = np.array(is_source[0]) | np.array(is_committed_target[0])

        rng, ids_rng, active_a_rng, active_b_rng = jax.random.split(rng, 4)
        base_ids = jax.random.randint(ids_rng, (1, seq_len), 0, vocab_size)

        # Two fillings that agree everywhere EXCEPT the active (uncommitted target)
        # region -- clean positions' resulting embeddings must be identical.
        active_fill_a = jax.random.randint(active_a_rng, (1, seq_len), 0, vocab_size)
        active_fill_b = jax.random.randint(active_b_rng, (1, seq_len), 0, vocab_size)
        is_clean_jnp = jnp.array(is_clean_np)[None, :]
        ids_a = jnp.where(is_clean_jnp, base_ids, active_fill_a)
        ids_b = jnp.where(is_clean_jnp, base_ids, active_fill_b)
        assert not jnp.all(ids_a == ids_b), "test setup bug: active region identical"

        emb_a = encode(ids_a, is_source, is_committed_target, is_valid)
        emb_b = encode(ids_b, is_source, is_committed_target, is_valid)

        clean_idx = np.where(is_clean_np)[0]
        active_idx = np.where(~is_clean_np)[0]
        if len(clean_idx) > 0:
            clean_a = np.array(emb_a[0, clean_idx])
            clean_b = np.array(emb_b[0, clean_idx])
            assert np.allclose(clean_a, clean_b, atol=1e-5), (
                f"Fix A regression: clean (source+committed) embeddings changed when "
                f"only active-region token IDs changed (cond_len={cond_len}, "
                f"committed_len={committed_len})"
            )
        if len(active_idx) > 0:
            # Sanity check the encoder is actually sensitive to its input at all here
            # (rules out a trivially-passing test from an encoder/mask that ignores
            # everything), by requiring at least one active position to visibly differ.
            active_a = np.array(emb_a[0, active_idx])
            active_b = np.array(emb_b[0, active_idx])
            assert not np.allclose(active_a, active_b, atol=1e-5), (
                f"test setup bug: active embeddings identical despite differing input "
                f"IDs (cond_len={cond_len}, committed_len={committed_len})"
            )
    print("[OK] Fix A: committed/source embeddings invariant to active-region token IDs "
          "across k=0/middle/near-last rounds, LM1B- and WMT14-like cond_len")


def test_fix_b_tight_packing_and_target_relative_extraction():
    """Fix B regression test: `generate_block_cond`'s packing must match training's
    tight `[source][target][pad]` layout (no gap), with target position 0 at each
    row's own `source_lengths[i]` (not a fixed `max_input_length` offset), for
    variable source lengths including LM1B-like `source_length=0`.
    """
    max_input_length, max_length = 10, 20
    pad_id = 99
    batch_size = 3
    source_lengths = jnp.array([0, 3, 10])  # zero (LM1B-like), short, and full-length

    # source_ids: real content 1..source_len, padded with pad_id to max_input_length.
    pos_src = np.arange(max_input_length)[None, :]
    source_lengths_np = np.array(source_lengths)
    source_ids = np.where(
        pos_src < source_lengths_np[:, None], pos_src + 1, pad_id,
    ).astype(np.int32)
    source_ids = jnp.array(source_ids)

    # committed_target_ids: real content 100..100+storage_width-1 (so they're
    # trivially distinguishable from source content and from pad_id), storage_width
    # = max_length for this test (matches generate_block_cond's own choice).
    storage_width = max_length
    committed_target_ids = jnp.array(
        np.tile(np.arange(100, 100 + storage_width), (batch_size, 1)).astype(np.int32)
    )

    full_ids, is_source_full = pack_source_target_tight(
        source_ids, committed_target_ids, source_lengths, max_input_length, max_length,
    )
    full_ids_np = np.array(full_ids)
    is_source_full_np = np.array(is_source_full)

    for i in range(batch_size):
        s_len = source_lengths_np[i]
        # No gap: positions [0, s_len) are real source content (1..s_len).
        assert list(full_ids_np[i, :s_len]) == list(range(1, s_len + 1)), (
            f"row {i}: source content not tight-packed at the front"
        )
        # Target position 0 sits EXACTLY at absolute index s_len (not max_input_length
        # unless s_len == max_input_length) -- no unmasked padding gap in between.
        assert full_ids_np[i, s_len] == 100, (
            f"row {i}: target position 0 not at absolute index {s_len} "
            f"(got {full_ids_np[i, s_len]} -- possible stale fixed-offset gap)"
        )
        # The rest of the canvas up to max_length is target content, contiguous.
        remaining = max_length - s_len
        assert list(full_ids_np[i, s_len:max_length]) == list(range(100, 100 + remaining)), (
            f"row {i}: target region not contiguous after source"
        )
        assert is_source_full_np[i, :s_len].all() and not is_source_full_np[i, s_len:].any()

    # Inverse extraction: predicted (full_ids itself, reused as a stand-in "prediction")
    # target-relative slice must recover exactly committed_target_ids's first
    # (max_length - source_lengths[i]) entries for every row, regardless of source_len.
    extracted = extract_target_relative(full_ids, source_lengths, storage_width, max_length)
    extracted_np = np.array(extracted)
    for i in range(batch_size):
        s_len = source_lengths_np[i]
        remaining = max_length - s_len
        assert list(extracted_np[i, :remaining]) == list(range(100, 100 + remaining)), (
            f"row {i}: target-relative extraction did not invert the tight packing"
        )
    print("[OK] Fix B: tight source/target packing + target-relative extraction, "
          "variable source lengths incl. LM1B-like source_length=0")


def test_fix_c_no_active_leak_into_clean_hidden_states():
    """Fix C regression test: under block-revelation's role-based attention mask
    (`role_clean_mask`, modules/model.py), clean (source+committed) query hidden
    states must stay invariant to what's happening at active (noisy) positions, at
    EVERY transformer layer depth -- including through the prepended special tokens
    (time/self-cond-cfg/model-mode tokens), which must not become an indirect leak
    path (active -> special token -> clean). Uses `capture_intermediates` to inspect
    every `ELFBlock`'s raw output directly, sidestepping `FinalLayer`'s zero-initialized
    kernel (which would otherwise make the model's final output trivially all-zero
    regardless of input, masking a real invariance failure as a false pass).
    """
    vocab_size, d_model, seq_len = 37, 8, 16
    batch_size = 2
    model = ELF(
        text_encoder_dim=d_model, max_length=seq_len, hidden_size=16, depth=3, num_heads=2,
        bottleneck_dim=4, num_time_tokens=2, num_self_cond_cfg_tokens=2,
        num_model_mode_tokens=2, vocab_size=vocab_size,
    )
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    t = jnp.full((batch_size,), 0.5)
    self_cond_cfg_scale = jnp.ones((batch_size,))
    dummy_x = jnp.ones((batch_size, seq_len, d_model))
    dummy_clean = jnp.ones((batch_size, seq_len), dtype=bool)
    params = model.init(
        init_rng, x=dummy_x, t=t, deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_step_active=jnp.array(True), role_clean_mask=dummy_clean,
    )["params"]

    cond_len, committed_len = 4, 4
    pos = np.arange(seq_len)
    is_clean_content = pos < (cond_len + committed_len)
    role_clean_mask = jnp.array(np.tile(is_clean_content, (batch_size, 1)))
    # Special tokens (model-mode, time, self-cond-cfg) are prepended in that order,
    # BEFORE content -- see ELF.__call__'s concatenation order.
    num_special = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens
    is_clean_full = np.concatenate([np.ones(num_special, dtype=bool), is_clean_content])
    clean_idx = np.where(is_clean_full)[0]
    active_idx = np.where(~is_clean_full)[0]

    rng, a_rng, b_rng = jax.random.split(rng, 3)
    base_x = jax.random.normal(a_rng, (batch_size, seq_len, d_model))
    active_fill_a = jax.random.normal(a_rng, (batch_size, seq_len, d_model))
    active_fill_b = jax.random.normal(b_rng, (batch_size, seq_len, d_model))
    is_clean_content_b = role_clean_mask[:, :, None]
    x_a = jnp.where(is_clean_content_b, base_x, active_fill_a)
    x_b = jnp.where(is_clean_content_b, base_x, active_fill_b)
    assert not jnp.allclose(x_a, x_b), "test setup bug: active region identical"

    def run(x):
        _, mutated = model.apply(
            {"params": params}, x, t, deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_scale, decoder_step_active=jnp.array(True),
            role_clean_mask=role_clean_mask,
            capture_intermediates=lambda mdl, method_name: isinstance(mdl, ELFBlock),
            mutable=["intermediates"],
        )
        return mutated["intermediates"]

    inter_a = run(x_a)
    inter_b = run(x_b)
    for block_name in inter_a:
        out_a = np.array(inter_a[block_name]["__call__"][0])
        out_b = np.array(inter_b[block_name]["__call__"][0])
        clean_a, clean_b = out_a[:, clean_idx], out_b[:, clean_idx]
        assert np.allclose(clean_a, clean_b, atol=1e-4), (
            f"Fix C regression: clean/special hidden states at {block_name} changed "
            f"when only active-region input changed"
        )
        active_a, active_b = out_a[:, active_idx], out_b[:, active_idx]
        assert not np.allclose(active_a, active_b, atol=1e-5), (
            f"test setup bug: active hidden states at {block_name} identical despite "
            f"differing input"
        )
    print("[OK] Fix C: clean/special hidden states invariant to active-region input "
          f"across all {model.depth} transformer layers")


def test_staggered_timesteps_schedule_shape():
    """Sanity-check the SetDLM-ported (b, k) derivation and the resulting per-position
    schedule, independent of the full train_step machinery.

    Reference numbers hand-derived from the actual `setdlms` repo's LM1B config
    (desired_block_size=4, max_block_size=8, length=128): b≈0.05926, k≈0.35814. Our
    own block_size=16/max_length=128 give different (still valid) numbers — checked
    against an independent hand computation here, not the reference repo's own values.
    """
    b, k = _staggered_bk(desired_block_size=16.0, length=128.0)
    # desired_num_blocks=8, desired_area=1/16=0.0625, max_block_size=32
    # b = 32/(128+32-1) = 32/159; k = desired_area/(b-desired_area)
    expected_b = 32.0 / 159.0
    expected_k = 0.0625 / (expected_b - 0.0625)
    assert abs(b - expected_b) < 1e-6, f"b={b} != expected {expected_b}"
    assert abs(k - expected_k) < 1e-6, f"k={k} != expected {expected_k}"
    assert 0.0 < b <= 1.0
    print(f"[OK] _staggered_bk(16, 128): b={b:.5f}, k={k:.5f}")

    # No cond (unconditional), single example, full 128-length target region
    # (attention_mask all-ones -> real target length == target_capacity, matching the
    # old fixed-target_capacity normalization exactly as a sanity check).
    cond_seq_mask = jnp.zeros((1, 128))
    attention_mask_full = jnp.ones((1, 128))
    for t_global_val in [0.05, 0.3, 0.5, 0.7, 0.95]:
        t_global = jnp.array([t_global_val])
        t_pos = compute_staggered_timesteps(
            t_global, cond_seq_mask, attention_mask_full, block_size=16, target_capacity=128,
        )
        t_pos_np = np.array(t_pos[0])
        assert np.all(t_pos_np >= 0.0) and np.all(t_pos_np <= 1.0), "t_pos out of [0,1]"
        # Monotonically non-increasing left-to-right (earlier positions resolve first).
        assert np.all(np.diff(t_pos_np) <= 1e-6), (
            f"t_pos not monotonically non-increasing at t_global={t_global_val}: {t_pos_np[:10]}..."
        )
        # The very first position should already be close to fully resolved once
        # t_global is comfortably past 0 (b is small relative to the full canvas).
        if t_global_val >= 0.3:
            assert t_pos_np[0] > 0.9, f"t_pos[0]={t_pos_np[0]} should be ~1 at t_global={t_global_val}"
        # The far end of a 128-long region shouldn't have started resolving yet
        # unless t_global is already very large.
        if t_global_val <= 0.5:
            assert t_pos_np[-1] < 0.1, f"t_pos[-1]={t_pos_np[-1]} should be ~0 at t_global={t_global_val}"
    print("[OK] compute_staggered_timesteps: monotonic, bounded, spans full [0,1] range across t_global")

    # Regression check for the per-example-length normalization fix: a SHORT, padded
    # example (35 real target tokens out of a 128 canvas, matching LM1B's mean length)
    # must have its real content span the *full* [0, 1-b] schedule range -- i.e. its
    # last real position should need t_global close to 1 to resolve, same as a
    # full-length example's last position would -- not get crushed into the first ~21%
    # of the range (the bug this test guards against: under the old fixed-
    # target_capacity normalization, a short example's last real position was already
    # FULLY resolved once t_global exceeded ~0.415, far too early).
    real_len = 35
    attention_mask_short = jnp.concatenate(
        [jnp.ones((1, real_len)), jnp.zeros((1, 128 - real_len))], axis=1,
    )
    t_global_mid = jnp.array([0.5])
    t_pos_short = compute_staggered_timesteps(
        t_global_mid, cond_seq_mask, attention_mask_short, block_size=16, target_capacity=128,
    )
    t_pos_short_np = np.array(t_pos_short[0])
    last_real_t_pos = t_pos_short_np[real_len - 1]
    assert last_real_t_pos < 0.1, (
        f"short-example last real position already resolving at t_global=0.5: "
        f"{last_real_t_pos} (should be ~0 -- per-example length normalization not applied?)"
    )
    # Sanity: that same position DOES fully resolve once t_global approaches 1.
    t_pos_short_end = compute_staggered_timesteps(
        jnp.array([1.0]), cond_seq_mask, attention_mask_short, block_size=16, target_capacity=128,
    )
    assert np.array(t_pos_short_end[0])[real_len - 1] > 0.9, (
        "short-example last real position should fully resolve by t_global=1.0"
    )
    print("[OK] compute_staggered_timesteps: short padded example spans the full schedule range")


def test_staggered_noise_finite_loss_and_trains_pos_embedder():
    """Staggered-noise (SetDLM-style single-pass multi-round curriculum) sanity checks.

    1. Finite loss, same active_frac invariants as plain block mode (staggering only
       changes *which timestep* each active position is noised at, not which positions
       are active/committed).
    2. pos_t_embedder actually receives a nonzero gradient update — catches a silent
       no-op if t_pos were accidentally not threaded into the model call.
    3. Disabling staggering (use_staggered_noise=False) must not create any
       pos_t_embedder params at all, preserving exact backward compatibility with
       every existing (non-staggered) config/checkpoint.
    """
    # decoder_prob=0.0 forces the denoiser branch every step (jax.lax.cond only flows
    # gradient through the branch actually taken at runtime) — pos_t_embedder only
    # lives inside the denoiser branch, so leaving decoder_prob at its default would
    # make this check flaky (zero gradient whenever the decoder branch got sampled).
    # num_steps=2: FinalLayer's kernel is zero-initialized, so upstream gradient
    # (pos_t_embedder, and every other pre-final-layer param) is legitimately zero on
    # step 0 regardless of correctness; compare the state before vs. after the *second*
    # step, by which point final_layer has moved off zero.
    metrics, state, new_state = _run_train_step(
        use_block_revelation=True, seed=2, use_staggered_noise=True, return_state=True,
        decoder_prob=0.0, num_steps=2,
    )
    for k, v in metrics.items():
        assert np.isfinite(v), f"non-finite metric {k}={v} (use_staggered_noise=True)"
    assert 0.0 <= metrics["active_frac"] <= 1.0

    assert "pos_t_embedder" in state.params, "pos_t_embedder params missing when use_staggered_noise=True"
    old_leaves = jax.tree_util.tree_leaves(state.params["pos_t_embedder"])
    new_leaves = jax.tree_util.tree_leaves(new_state.params["pos_t_embedder"])
    changed = any(not np.allclose(o, n) for o, n in zip(old_leaves, new_leaves))
    assert changed, "pos_t_embedder params did not change after a training step (gradient not flowing)"
    print(f"[OK] staggered-noise train_step finite loss, pos_t_embedder trains: {metrics}")

    # Backward compatibility: staggering off must not create pos_t_embedder at all.
    _, state_off, _ = _run_train_step(
        use_block_revelation=True, seed=2, use_staggered_noise=False, return_state=True,
    )
    assert "pos_t_embedder" not in state_off.params, (
        "pos_t_embedder params unexpectedly present when use_staggered_noise=False"
    )
    print("[OK] use_staggered_noise=False creates no pos_t_embedder params (backward compatible)")


def test_round_count_reweighting_fixes_bias():
    rng = np.random.default_rng(0)
    n = 20000
    block_size = 16
    max_input_length = 64
    max_length = 128
    cond_lens = rng.integers(1, max_input_length, size=n)
    target_lens = rng.integers(1, 100, size=n)
    ds = Dataset.from_dict({
        "condition_sequence_length": cond_lens.tolist(),
        "sequence_length": target_lens.tolist(),
    })

    K = compute_block_round_counts(ds, block_size, max_length, max_input_length)
    hist = collections.Counter(K.tolist())
    true_marginal = {k: v / n for k, v in hist.items()}

    # Old (biased) scheme: round=0 instance mass per K_i bucket is P(K_i=j) * (1/j).
    old_mass = {j: true_marginal[j] * (1.0 / j) for j in true_marginal}
    old_total = sum(old_mass.values())
    old_posterior = {j: old_mass[j] / old_total for j in old_mass}

    # New scheme: examples drawn with weight K_i, so P(example, round=0) = 1/sum(K_i)
    # for every example regardless of K_i -> P(K_i=j | round=0) should equal true marginal.
    sum_K = K.sum()
    new_mass = {j: hist[j] * (1.0 / sum_K) for j in hist}
    new_total = sum(new_mass.values())
    new_posterior = {j: new_mass[j] / new_total for j in new_mass}

    assert old_posterior[1] > true_marginal[1] * 1.5, "expected old scheme to over-represent K_i=1"
    for j in true_marginal:
        assert abs(new_posterior[j] - true_marginal[j]) < 1e-9, (
            f"K_i={j}: weighted posterior {new_posterior[j]} != true marginal {true_marginal[j]}"
        )
    print(
        f"[OK] K_i=1: true={true_marginal[1]:.3f}, old_biased_posterior={old_posterior[1]:.3f} "
        f"(inflated), new_weighted_posterior={new_posterior[1]:.3f} (matches true)"
    )


if __name__ == "__main__":
    test_sample_block_round_invariants()
    test_train_step_block_mode_finite_loss()
    test_fix_a_no_future_leak_into_committed_encoding()
    test_fix_b_tight_packing_and_target_relative_extraction()
    test_fix_c_no_active_leak_into_clean_hidden_states()
    test_staggered_timesteps_schedule_shape()
    test_staggered_noise_finite_loss_and_trains_pos_embedder()
    test_round_count_reweighting_fixes_bias()
    print("\nAll block-flow sanity tests passed.")
