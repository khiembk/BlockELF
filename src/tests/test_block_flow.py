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
  4. Per-position time conditioning (use_position_time): confirms a model built with it
     enabled produces byte-identical output to one without it when t_position_delta=0
     (zero-init safety — existing checkpoints stay valid), that `compute_staggered_timesteps`
     produces a sane per-position schedule (monotonic, bounded, delta=0 at cond positions),
     and that `merge_pretrained_params` correctly carries over old values while leaving new
     (added) params at their fresh init.
  5. Fix A/B/C (discrete block-revelation path, ported from a parallel LM1B effort, see
     git history): T5 encoder future-leak, train/infer positional layout, and denoiser
     role-based attention masking — plus a local addition (not in the upstream port):
     Fix C's role mask must also exclude padding KEYS for active queries, since the
     upstream version only ANDs in the role split and drops the plain validity mask
     entirely for non-clean queries.
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
from utils.checkpoint_utils import merge_pretrained_params
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
    use_staggered_time = False
    max_length = 20
    max_input_length = None


def _run_train_step(use_block_revelation, seed=0, use_staggered_time=False):
    vocab_size, d_model, seq_len, cond_len, batch_size = 37, 8, 20, 6, 4
    config = _Config()
    config.use_block_revelation = use_block_revelation
    config.use_staggered_time = use_staggered_time
    config.max_length = seq_len
    config.max_input_length = cond_len

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
        use_position_time=use_staggered_time,
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
    path (active -> special token -> clean). Also checks a gap found in the upstream
    (LM1B) port of this fix: active-query rows must still exclude PADDING keys, not
    just respect the clean/active role split -- the upstream version ANDed in only the
    role mask and dropped the plain validity mask entirely for non-clean queries,
    silently re-exposing padding content to every active query. Uses
    `capture_intermediates` to inspect every `ELFBlock`'s raw output directly,
    sidestepping `FinalLayer`'s zero-initialized kernel (which would otherwise make the
    model's final output trivially all-zero regardless of input, masking a real
    invariance failure as a false pass).
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

    # Layout: [0,cond_len)=clean source, [cond_len,cond_len+committed_len)=clean
    # committed target, [.., seq_len-pad_len)=active target, [seq_len-pad_len, seq_len)
    # =padding (not clean, not active content -- excluded from both roles' real content).
    cond_len, committed_len, pad_len = 4, 4, 3
    pos = np.arange(seq_len)
    is_clean_content = pos < (cond_len + committed_len)
    is_pad = pos >= (seq_len - pad_len)
    role_clean_mask = jnp.array(np.tile(is_clean_content, (batch_size, 1)))
    attention_mask = jnp.array(np.tile(~is_pad, (batch_size, 1)).astype(np.float32))
    # Special tokens (model-mode, time, self-cond-cfg) are prepended in that order,
    # BEFORE content -- see ELF.__call__'s concatenation order.
    num_special = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens
    is_clean_full = np.concatenate([np.ones(num_special, dtype=bool), is_clean_content])
    is_pad_full = np.concatenate([np.zeros(num_special, dtype=bool), is_pad])
    clean_idx = np.where(is_clean_full)[0]
    active_idx = np.where(~is_clean_full & ~is_pad_full)[0]

    rng, a_rng, b_rng, pad_a_rng, pad_b_rng = jax.random.split(rng, 5)
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
            attention_mask=attention_mask, role_clean_mask=role_clean_mask,
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

    # Local addition: active queries must also be invariant to PADDING content (the
    # gap the upstream port missed) -- same perturbation trick, this time varying only
    # the padding region's input while holding clean+active content fixed.
    pad_fill_a = jax.random.normal(pad_a_rng, (batch_size, seq_len, d_model))
    pad_fill_b = jax.random.normal(pad_b_rng, (batch_size, seq_len, d_model))
    is_pad_b = jnp.array(np.tile(is_pad, (batch_size, 1)))[:, :, None]
    x_pad_a = jnp.where(is_pad_b, pad_fill_a, base_x)
    x_pad_b = jnp.where(is_pad_b, pad_fill_b, base_x)
    assert not jnp.allclose(x_pad_a, x_pad_b), "test setup bug: padding region identical"

    inter_pad_a = run(x_pad_a)
    inter_pad_b = run(x_pad_b)
    non_pad_idx = np.where(~is_pad_full)[0]
    for block_name in inter_pad_a:
        out_a = np.array(inter_pad_a[block_name]["__call__"][0])
        out_b = np.array(inter_pad_b[block_name]["__call__"][0])
        assert np.allclose(out_a[:, non_pad_idx], out_b[:, non_pad_idx], atol=1e-4), (
            f"padding-leak regression: clean/active hidden states at {block_name} "
            f"changed when only padding-region input changed (active queries must "
            f"exclude padding keys, not just respect the clean/active role split)"
        )
    print("[OK] Fix C (local addition): clean/active hidden states invariant to "
          f"padding-region input across all {model.depth} transformer layers")


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


def test_train_step_staggered_time_finite_loss():
    metrics = _run_train_step(use_block_revelation=True, seed=2, use_staggered_time=True)
    for k, v in metrics.items():
        assert np.isfinite(v), f"non-finite metric {k}={v} with staggered time enabled"
    # use_staggered_time replaces (not layers on top of) the discrete committed/active
    # split: every non-cond position is supervised every step, so active_frac should
    # match the plain non-block case exactly, not the restricted block-mode fraction.
    base_metrics = _run_train_step(use_block_revelation=False, seed=2)
    assert abs(metrics["active_frac"] - base_metrics["active_frac"]) < 1e-6, (
        f"staggered-time active_frac ({metrics['active_frac']}) should match the "
        f"non-block active_frac ({base_metrics['active_frac']}), not a restricted subset"
    )
    print(f"[OK] train_step finite loss (use_block_revelation=True, use_staggered_time=True): {metrics}")


def test_staggered_timesteps_schedule():
    """Sanity-check the SetDLM-ported (b, k) derivation and the resulting per-position
    schedule, independent of the full train_step machinery."""
    b, k = _staggered_bk(desired_block_size=16.0, length=128.0)
    expected_b = 32.0 / 159.0
    expected_k = 0.0625 / (expected_b - 0.0625)
    assert abs(b - expected_b) < 1e-6, f"b={b} != expected {expected_b}"
    assert abs(k - expected_k) < 1e-6, f"k={k} != expected {expected_k}"
    assert 0.0 < b <= 1.0
    print(f"[OK] _staggered_bk(16, 128): b={b:.5f}, k={k:.5f}")

    # Conditional task: real source (cond) positions + a target region. delta must be
    # exactly 0 at every cond position, regardless of t_global (the gap found in the
    # LM1B-only port of this scheme, which never has real cond positions to worry about).
    cond_len, S = 10, 128
    cond_seq_mask = jnp.concatenate(
        [jnp.ones((1, cond_len)), jnp.zeros((1, S - cond_len))], axis=1,
    )
    attention_mask_full = jnp.ones((1, S))
    for t_global_val in [0.05, 0.3, 0.5, 0.7, 0.95]:
        t_global = jnp.array([t_global_val])
        delta = compute_staggered_timesteps(
            t_global, cond_seq_mask, attention_mask_full, block_size=16, target_capacity=128,
        )
        delta_np = np.array(delta[0])
        assert np.all(delta_np[:cond_len] == 0.0), (
            f"expected delta==0 at cond positions, got {delta_np[:cond_len]} at t_global={t_global_val}"
        )
        t_pos_np = t_global_val + delta_np
        assert np.all(t_pos_np >= -1e-6) and np.all(t_pos_np <= 1.0 + 1e-6), "t_position out of [0,1]"
        # Monotonically non-increasing left-to-right within the target region (earlier
        # target positions resolve first).
        assert np.all(np.diff(t_pos_np[cond_len:]) <= 1e-6), (
            f"t_position not monotonically non-increasing at t_global={t_global_val}"
        )
        if t_global_val >= 0.3:
            assert t_pos_np[cond_len] > 0.9, (
                f"t_position[0]={t_pos_np[cond_len]} should be ~1 at t_global={t_global_val}"
            )
        if t_global_val <= 0.5:
            assert t_pos_np[-1] < 0.1, (
                f"t_position[-1]={t_pos_np[-1]} should be ~0 at t_global={t_global_val}"
            )
    print("[OK] compute_staggered_timesteps: delta==0 at cond positions, monotonic/bounded target schedule")

    # Regression check for the per-example-length normalization (v3) fix: a short, padded
    # target region (35 real tokens out of a 118-long target capacity) must have its real
    # content span the *full* [0, 1-b] schedule range, not get crushed into an early slice
    # by the fixed target_capacity.
    real_target_len = 35
    attention_mask_short = jnp.concatenate(
        [jnp.ones((1, cond_len + real_target_len)), jnp.zeros((1, S - cond_len - real_target_len))], axis=1,
    )
    t_global_mid = jnp.array([0.5])
    delta_short = compute_staggered_timesteps(
        t_global_mid, cond_seq_mask, attention_mask_short, block_size=16, target_capacity=128,
    )
    last_real_t_pos = 0.5 + np.array(delta_short[0])[cond_len + real_target_len - 1]
    assert last_real_t_pos < 0.1, (
        f"short-target last real position already resolving at t_global=0.5: {last_real_t_pos} "
        "(should be ~0 -- per-example length normalization not applied?)"
    )
    delta_short_end = compute_staggered_timesteps(
        jnp.array([1.0]), cond_seq_mask, attention_mask_short, block_size=16, target_capacity=128,
    )
    last_real_t_pos_end = 1.0 + np.array(delta_short_end[0])[cond_len + real_target_len - 1]
    assert last_real_t_pos_end > 0.9, "short-target last real position should fully resolve by t_global=1.0"
    print("[OK] compute_staggered_timesteps: short padded target region spans the full schedule range")


def test_position_time_zero_init_matches_baseline():
    """A model built with use_position_time=True, given t_position_delta=0 (or None),
    must produce byte-identical output to the same model with use_position_time=False —
    this is what makes the new pathway safe to add without disturbing existing checkpoints."""
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    text_encoder_dim, hidden_size, seq_len, vocab_size = 8, 16, 12, 20

    base_model = ELF(text_encoder_dim=text_encoder_dim, max_length=seq_len, hidden_size=hidden_size,
                      depth=2, num_heads=2, bottleneck_dim=4, num_time_tokens=2,
                      num_self_cond_cfg_tokens=0, vocab_size=vocab_size, use_position_time=False)
    new_model = ELF(text_encoder_dim=text_encoder_dim, max_length=seq_len, hidden_size=hidden_size,
                     depth=2, num_heads=2, bottleneck_dim=4, num_time_tokens=2,
                     num_self_cond_cfg_tokens=0, vocab_size=vocab_size, use_position_time=True)

    dummy_x = jax.random.normal(rng, (2, seq_len, text_encoder_dim))
    dummy_t = jnp.full((2,), 0.5)
    base_params = base_model.init(init_rng, x=dummy_x, t=dummy_t, deterministic=True)
    new_params = new_model.init(init_rng, x=dummy_x, t=dummy_t, deterministic=True)

    merged, n_matched, n_new = merge_pretrained_params(base_params["params"], new_params["params"])
    base_leaf_count = len(jax.tree_util.tree_leaves(base_params["params"]))
    assert n_matched == base_leaf_count, f"expected all {base_leaf_count} base params matched, got {n_matched}"
    assert n_new > 0, "expected new params from pos_t_embedder/pos_t_proj, got none"
    assert "pos_t_proj" in merged and "pos_t_embedder" in merged, "new submodules missing from merge"
    np.testing.assert_array_equal(
        np.array(merged["pos_t_proj"]["kernel"]), np.array(new_params["params"]["pos_t_proj"]["kernel"]),
    )

    base_out, _ = base_model.apply({"params": base_params["params"]}, x=dummy_x, t=dummy_t, deterministic=True)
    new_out_zero, _ = new_model.apply(
        {"params": merged}, x=dummy_x, t=dummy_t, deterministic=True,
        t_position_delta=jnp.zeros((2, seq_len)),
    )
    new_out_none, _ = new_model.apply({"params": merged}, x=dummy_x, t=dummy_t, deterministic=True)
    np.testing.assert_allclose(np.array(base_out), np.array(new_out_zero), atol=1e-6)
    np.testing.assert_allclose(np.array(base_out), np.array(new_out_none), atol=1e-6)

    # A nonzero delta *should* change the output once pos_t_proj is no longer zero —
    # confirm the new pathway is actually wired up (not a silent no-op). `final_layer`'s
    # kernel is itself zero-initialized (standard diffusion-model practice), so the whole
    # model outputs exactly zero for ANY input at init — perturb it too, so the output
    # actually depends on `x`, isolating whether pos_t_proj's effect propagates.
    perturb_rng, final_rng = jax.random.split(jax.random.PRNGKey(123))
    perturbed_base = jax.tree_util.tree_map(lambda x: x, base_params["params"])
    perturbed_base["final_layer"]["linear"]["kernel"] = 0.1 * jax.random.normal(
        final_rng, perturbed_base["final_layer"]["linear"]["kernel"].shape,
    )
    perturbed_new = jax.tree_util.tree_map(lambda x: x, merged)
    perturbed_new["final_layer"] = perturbed_base["final_layer"]
    perturbed_new["pos_t_proj"]["kernel"] = (
        perturbed_new["pos_t_proj"]["kernel"] + 0.1 * jax.random.normal(perturb_rng, perturbed_new["pos_t_proj"]["kernel"].shape)
    )

    base_out_nz, _ = base_model.apply({"params": perturbed_base}, x=dummy_x, t=dummy_t, deterministic=True)
    new_out_nonzero, _ = new_model.apply(
        {"params": perturbed_new}, x=dummy_x, t=dummy_t, deterministic=True,
        t_position_delta=0.5 * jnp.ones((2, seq_len)),
    )
    assert not np.allclose(np.array(base_out_nz), np.array(new_out_nonzero), atol=1e-4), (
        "expected a nonzero stagger + nonzero pos_t_proj to change the output"
    )
    print(f"[OK] use_position_time zero-init matches baseline exactly; merge added {n_new} new param(s), "
          f"kept {n_matched} matched")


if __name__ == "__main__":
    test_sample_block_round_invariants()
    test_train_step_block_mode_finite_loss()
    test_fix_a_no_future_leak_into_committed_encoding()
    test_fix_b_tight_packing_and_target_relative_extraction()
    test_fix_c_no_active_leak_into_clean_hidden_states()
    test_round_count_reweighting_fixes_bias()
    test_staggered_timesteps_schedule()
    test_position_time_zero_init_matches_baseline()
    test_train_step_staggered_time_finite_loss()
    print("\nAll block-flow sanity tests passed.")
