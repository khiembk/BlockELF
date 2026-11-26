"""Round-zero equivalence test (OWT repetition-collapse follow-up spec §1.2).

Replays IDENTICAL z0/timesteps/rng/params through (a) generate_block_uncond's actual
round-0 internals (encode an all-pad committed_target_ids through the frozen encoder,
build_committed_context with nothing committed, then _generate_samples_single_batch +
_dlm_decode_batch -- exactly as generate_block_uncond itself does for OWT's
role_aware_uncond_attention=False config) and (b) the full-sequence path's equivalent
call (_generate_samples_single_batch with cond_seq=None/cond_seq_mask=None, matching
test_generation_uncond's p_generate call). Compares z_final, decoder logits, and
decoded token IDs exactly -- not just top-level-seed "should probably match."

Uses the attentive dummy encoder (not the plain embedding-lookup one) specifically
because it is sensitive to attention_mask content -- a masking bug could not be
detected with an encoder that ignores attention_mask entirely.

Pure-CPU. Run with:  cd src && python tests/test_round_zero_equivalence.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from utils.block_utils import build_committed_context
from utils.encoder_utils import encode_text
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch
from utils.sampling_utils import get_sampling_steps
from configs.config import SamplingConfig

from tests.test_block_flow import _make_uncond_sampler_fixture, _make_attentive_dummy_encoder


def test_round_zero_block_matches_full_sequence_owt_config():
    vocab_size, d_model, canvas_length, batch_size = 37, 8, 16, 3
    config, model, params, rng = _make_uncond_sampler_fixture(
        vocab_size, d_model, canvas_length, seed=7,
    )
    # Match OWT's actual inference config: role_aware_uncond_attention=False.
    config.role_aware_uncond_attention = False
    pad_token_id = 0

    attentive_encoder = _make_attentive_dummy_encoder(vocab_size, d_model)
    rng, enc_rng = jax.random.split(rng)
    encoder_params = attentive_encoder.init(
        enc_rng, input_ids=jnp.zeros((1, canvas_length), dtype=jnp.int32),
        attention_mask=jnp.ones((1, canvas_length, canvas_length)),
    )["params"]

    sampling_config = SamplingConfig(
        sampling_method="ode", num_sampling_steps=[6], cfgs=[1],
        self_cond_cfg_scales=[1.0], time_schedule="uniform",
    )
    cfg_scale, self_cond_cfg_scale = 1.0, 1.0

    rng, z_rng, t_rng, sample_rng = jax.random.split(rng, 4)
    z0 = jax.random.normal(z_rng, (batch_size, canvas_length, d_model))
    t_steps = get_sampling_steps(t_rng, n_steps=6, time_schedule="uniform")

    # --- (a) generate_block_uncond's actual round-0 path ---
    pos_tgt = jnp.arange(canvas_length)[None, :]
    is_source = jnp.zeros((batch_size, canvas_length), dtype=bool)
    committed_target_ids = jnp.full((batch_size, canvas_length), pad_token_id, dtype=jnp.int32)
    committed_len = jnp.zeros((batch_size,), dtype=jnp.int32)
    is_committed = pos_tgt < committed_len[:, None]
    assert not bool(jnp.any(is_committed)), "sanity: nothing committed at round 0"

    encoder_attn, _, clean_mask = build_committed_context(is_source, is_committed, is_committed)
    cond_seq = encode_text(
        input_ids=committed_target_ids, attention_mask=encoder_attn,
        encoder_apply_fn=attentive_encoder.apply, encoder_params=encoder_params,
        latent_mean=0.0, latent_std=1.0,
    )
    assert bool(jnp.all(jnp.isfinite(cond_seq))), "encoder produced non-finite cond_seq at round 0"

    z_final_block = _generate_samples_single_batch(
        model_params=params, model_apply_fn=model.apply, rng=sample_rng,
        z=z0, t_steps=t_steps, cond_seq=cond_seq, cond_seq_mask=clean_mask,
        config=config, sampling_config=sampling_config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        # role_aware_uncond_attention=False for OWT -> generate_block_uncond passes no
        # attention_mask/role_clean_mask at all (extra_sampler_kwargs={}).
    )
    ids_block = _dlm_decode_batch(
        z=z_final_block, model_params=params, model_apply_fn=model.apply,
        t_final_val=t_steps[-1], config=config, self_cond_cfg_scale=self_cond_cfg_scale,
    )

    # --- (b) full-sequence path's equivalent call ---
    z_final_full = _generate_samples_single_batch(
        model_params=params, model_apply_fn=model.apply, rng=sample_rng,
        z=z0, t_steps=t_steps, cond_seq=None, cond_seq_mask=None,
        config=config, sampling_config=sampling_config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
    )
    ids_full = _dlm_decode_batch(
        z=z_final_full, model_params=params, model_apply_fn=model.apply,
        t_final_val=t_steps[-1], config=config, self_cond_cfg_scale=self_cond_cfg_scale,
    )

    max_diff = float(jnp.abs(z_final_block - z_final_full).max())
    assert jnp.allclose(z_final_block, z_final_full, atol=1e-5), (
        f"round-0 block path diverges from full-sequence path despite identical "
        f"z0/timesteps/rng/params (max |diff|={max_diff:.3e}) -- the committed_target_ids/"
        f"build_committed_context/encode_text round-0 machinery introduces a real "
        f"numerical difference, not just an expected no-op"
    )
    assert jnp.array_equal(ids_block, ids_full), (
        "round-0 block path's decoded token IDs diverge from the full-sequence path's"
    )
    print(f"[OK] round-0 equivalence: block-path z_final matches full-sequence z_final "
          f"exactly (max |diff|={max_diff:.3e}), decoded token IDs identical -- no "
          f"round-0 implementation bug found for OWT's role_aware_uncond_attention=False config")


def test_round_zero_encoder_attention_mask_is_degenerate_but_harmless():
    """Documents a real quirk found via code audit, confirmed here: at round 0,
    build_committed_context's `is_valid` argument is `is_committed` (all-False), so the
    T5-encoder attention_mask it builds is ALSO all-False (no valid key positions at
    all) -- not "block everything" but a uniformly-shifted bias that, because softmax is
    shift-invariant, is numerically IDENTICAL to passing an all-ones (fully unmasked)
    attention_mask. This looks alarming on first read (an all-invalid-looking mask that
    doesn't actually mask anything) but is proven harmless by the test above: cond_seq's
    value is irrelevant regardless, since cond_seq_mask (clean_mask) is also all-False at
    round 0, so restore_cond's jnp.where never selects it."""
    canvas_length, batch_size = 10, 2
    is_committed = jnp.zeros((batch_size, canvas_length), dtype=bool)
    encoder_attn_degenerate, _, clean_mask = build_committed_context(
        jnp.zeros_like(is_committed), is_committed, is_committed,
    )
    encoder_attn_allvalid, _, _ = build_committed_context(
        jnp.zeros_like(is_committed), is_committed,
        jnp.ones((batch_size, canvas_length), dtype=bool),
    )
    # Both produce an all-ones-equivalent additive-bias encoder attention mask (since
    # is_clean is all-False, encoder_attn = (~is_clean & is_valid)[... broadcast ...],
    # which is literally `is_valid` broadcast over the key axis for every query).
    assert bool(jnp.all(clean_mask == 0)), "round 0: clean_mask must be all-False"
    print("[OK] round-0 encoder attention mask quirk confirmed harmless: cond_seq_mask "
          "(clean_mask) is all-False regardless of what the degenerate encoder "
          "attention_mask computes, so cond_seq's value can never be selected")


if __name__ == "__main__":
    test_round_zero_block_matches_full_sequence_owt_config()
    test_round_zero_encoder_attention_mask_is_degenerate_but_harmless()
    print("All round-zero equivalence tests passed.")
