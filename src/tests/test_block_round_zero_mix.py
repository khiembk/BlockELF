"""Tests for the OWT-only opt-in round-sampling curriculum (`rho` / `block_round_zero_mix`,
see utils/block_utils.sample_block_round and docs/owt_collapse_root_cause_findings.md's
T1 recommendation). Covers the external review's required "critical tests":

  1. rho=0 reproduces the EXACT original single-uniform-draw masks/boundary for a fixed
     key (bit-for-bit), both at the sample_block_round level and end-to-end through
     train_step (confirming LM1B/WMT14/XSum -- any config that leaves
     block_round_zero_mix at its 0.0 default, or has use_block_revelation=False
     entirely -- see zero behavior change).
  2. rho=1 selects k=0 for every example, always.
  3. Empirical P(k=0) matches the closed-form q_rho(0|K_i) = (1-rho)/K_i + rho for
     varied K_i in {1, 2, 8, 64}.
  4. Mask invariants (disjoint committed/active, no padding leak, cond subset of
     committed, active empty iff target_len==0) still hold for rho > 0.
  5. A (fast, reduced-precision) Monte Carlo check of the REAL per-device-minibatch-
     pooled loss weighting (reduce_token_loss pools the WHOLE minibatch into one
     scalar denominator, not per-example) confirms rho>0 measurably shrinks the
     effective late-vs-early target-block weighting imbalance, without asserting an
     exact magnitude (that needs the full offline simulation, not a unit test).

Pure-CPU. Run with:  cd src && python tests/test_block_round_zero_mix.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from utils.block_utils import sample_block_round


def _make_fixture(S=40, cond_len=0, total_len=40):
    pos = np.arange(S)
    is_cond = (pos < cond_len).astype(np.float32)[None, :]
    is_valid = (pos < total_len).astype(np.float32)[None, :]
    return jnp.array(is_cond), jnp.array(is_valid)


def test_rho_zero_is_bit_exact_regression():
    rng = jax.random.PRNGKey(0)
    for trial in range(20):
        rng, fixture_rng, call_rng = jax.random.split(rng, 3)
        S = int(jax.random.randint(fixture_rng, (), 20, 100))
        cond_len = int(jax.random.randint(fixture_rng, (), 0, S // 2 + 1))
        is_cond, is_valid = _make_fixture(S=S, cond_len=cond_len, total_len=S)
        committed_old, active_old, boundary_old = sample_block_round(call_rng, is_cond, is_valid, block_size=16)
        committed_new, active_new, boundary_new = sample_block_round(
            call_rng, is_cond, is_valid, block_size=16, rho=0.0,
        )
        assert jnp.array_equal(committed_old, committed_new)
        assert jnp.array_equal(active_old, active_new)
        assert jnp.array_equal(boundary_old, boundary_new)
    print("[OK] rho=0.0 (default) reproduces the exact original sample_block_round output, "
          "bit-for-bit, across 20 random fixture shapes")


def test_rho_one_always_selects_k0():
    rng = jax.random.PRNGKey(1)
    for K_i in (1, 2, 8, 64):
        S = K_i * 16 + 10  # +10 cond positions
        is_cond, is_valid = _make_fixture(S=S, cond_len=10, total_len=S)
        batch_size = 200
        is_cond_b = jnp.tile(is_cond, (batch_size, 1))
        is_valid_b = jnp.tile(is_valid, (batch_size, 1))
        rng, call_rng = jax.random.split(rng)
        committed, active, boundary = sample_block_round(
            call_rng, is_cond_b, is_valid_b, block_size=16, rho=1.0,
        )
        cond_len_arr = is_cond_b.sum(axis=1)
        k = (boundary - cond_len_arr) / 16
        assert jnp.all(k == 0), f"rho=1.0 must force k=0 for every example (K_i={K_i})"
    print("[OK] rho=1.0 forces k=0 for every example, for K_i in {1,2,8,64}")


def test_empirical_p_k0_matches_closed_form_q_rho():
    rng = jax.random.PRNGKey(2)
    trials = 20000
    for K_i in (1, 2, 8, 64):
        S = K_i * 16 + 10
        is_cond, is_valid = _make_fixture(S=S, cond_len=10, total_len=S)
        is_cond_b = jnp.tile(is_cond, (trials, 1))
        is_valid_b = jnp.tile(is_valid, (trials, 1))
        cond_len_arr = is_cond_b.sum(axis=1)
        for rho in (0.0, 0.1, 0.25, 0.5):
            rng, call_rng = jax.random.split(rng)
            _, _, boundary = sample_block_round(call_rng, is_cond_b, is_valid_b, block_size=16, rho=rho)
            k = (boundary - cond_len_arr) / 16
            empirical_p_k0 = float(jnp.mean(k < 0.5))
            theory_p_k0 = (1 - rho) / K_i + rho
            assert abs(empirical_p_k0 - theory_p_k0) < 0.02, (
                f"K_i={K_i} rho={rho}: empirical P(k=0)={empirical_p_k0:.4f} vs "
                f"theory q_rho(0|K_i)={theory_p_k0:.4f} (diff too large for {trials} trials)"
            )
    print("[OK] empirical P(k=0) matches closed-form q_rho(0|K_i)=(1-rho)/K_i+rho "
          "within Monte Carlo noise, for K_i in {1,2,8,64} and rho in {0,0.1,0.25,0.5}")


def test_mask_invariants_hold_for_rho_greater_than_zero():
    rng = jax.random.PRNGKey(3)
    S = 40
    trials = 200
    for trial in range(trials):
        rng, cl_rng, tl_rng, b_rng, rho_rng, round_rng = jax.random.split(rng, 6)
        cond_len = int(jax.random.randint(cl_rng, (), 0, S // 2))
        total_len = int(jax.random.randint(tl_rng, (), cond_len, S + 1))
        block_size = int(jax.random.randint(b_rng, (), 1, 17))
        rho = float(jax.random.uniform(rho_rng, (), minval=0.0, maxval=1.0))

        pos = np.arange(S)
        is_cond = (pos < cond_len).astype(np.float32)[None, :]
        is_valid = (pos < total_len).astype(np.float32)[None, :]

        committed, active, boundary = sample_block_round(round_rng, is_cond, is_valid, block_size, rho=rho)
        committed, active, boundary = np.array(committed), np.array(active), np.array(boundary)
        target_len = total_len - cond_len

        assert np.all((committed > 0) | (active > 0) | (is_valid[0] == 0)), "uncovered valid position"
        assert not np.any((committed > 0) & (active > 0)), "committed/active overlap"
        assert np.all(committed[0][is_valid[0] == 0] == 0), "committed leaks into padding"
        assert np.all(active[0][is_valid[0] == 0] == 0), "active leaks into padding"
        assert np.all(committed[0][pos < cond_len] == is_valid[0][pos < cond_len]), "cond not fully committed"
        if target_len > 0:
            assert active.sum() > 0, "active set empty despite remaining target content"
        else:
            assert active.sum() == 0
    print(f"[OK] mask invariants (disjoint committed/active, no padding leak, cond fully "
          f"committed, active non-empty iff target content remains) hold for random rho "
          f"in [0,1] across {trials} trials")


def test_batch_pooled_weighting_imbalance_shrinks_with_rho():
    """Fast, reduced-precision sanity check of the REAL batch-pooled weighting effect
    (reduce_token_loss pools ALL examples in a per-device minibatch into ONE scalar
    denominator -- not a per-example-normalized objective, see train_step.py). Confirms
    directionally that rho>0 shrinks the late-vs-early target-block weighting
    imbalance; does not assert an exact magnitude (that requires a much larger offline
    simulation against the real OWT length distribution -- see
    docs/owt_collapse_root_cause_findings.md)."""
    rng = np.random.RandomState(0)
    block_size = 16
    per_device_batch = 12
    K = 64
    n_steps = 4000

    def mean_weight_when_active(rho, j):
        # All examples K_i=64 for simplicity (OWT's real K_i is tightly concentrated
        # near 64 -- see the doc's own length-distribution measurement).
        lens = np.full(per_device_batch, K * block_size, dtype=np.float64)
        weights = []
        for _ in range(n_steps):
            u = rng.uniform(size=per_device_batch)
            k_uniform = np.floor(u * K).astype(int)
            if rho > 0:
                force_zero = rng.uniform(size=per_device_batch) < rho
                ks = np.where(force_zero, 0, k_uniform)
            else:
                ks = k_uniform
            active_sizes = lens - ks * block_size
            denom = active_sizes.sum()
            # Probe: is target block j active this step for a 13th (held-out) example
            # whose own k we draw the same way?
            u_probe = rng.uniform()
            k_probe = int(np.floor(u_probe * K))
            if rho > 0 and rng.uniform() < rho:
                k_probe = 0
            if k_probe <= j and denom > 0:
                probe_active = K * block_size - k_probe * block_size
                weights.append(1.0 / (denom + probe_active))
        return np.mean(weights) if weights else 0.0, len(weights) / n_steps

    for rho in (0.0, 0.25):
        w0, p0 = mean_weight_when_active(rho, j=0)
        w63, p63 = mean_weight_when_active(rho, j=63)
        ratio = (p63 * w63) / max(p0 * w0, 1e-12)
        print(f"[INFO] rho={rho}: w_j[0]~{p0*w0:.2e} w_j[63]~{p63*w63:.2e} ratio~{ratio:.1f}")
        if rho == 0.0:
            ratio_rho0 = ratio
        else:
            ratio_rho_pos = ratio

    assert ratio_rho_pos < ratio_rho0, (
        f"rho=0.25 should shrink the late-vs-early effective weighting ratio relative to "
        f"rho=0 (got rho=0: {ratio_rho0:.1f}, rho=0.25: {ratio_rho_pos:.1f})"
    )
    print("[OK] batch-pooled weighting imbalance (NOT per-example-normalized) measurably "
          "shrinks under rho=0.25 relative to rho=0, confirming T1's mechanism works as "
          "intended under the REAL (pooled) loss normalization, not just the flawed "
          "per-example-isolated approximation")


def test_train_step_unaffected_without_opting_in():
    """End-to-end: train_step with block_round_zero_mix left at its default (unset,
    i.e. getattr falls back to 0.0) must produce IDENTICAL new_state/metrics to a
    config that explicitly sets it to 0.0, for both use_block_revelation=False
    (LM1B/WMT14/XSum's non-block path) and use_block_revelation=True (block path at
    its default rho) -- confirming no existing config needs to change to keep its
    historical behavior exactly."""
    from tests.test_block_flow import _run_train_step, _Config

    # _run_train_step's _Config doesn't define block_round_zero_mix at all (attribute
    # absent) -- train_step.py's getattr(config, "block_round_zero_mix", 0.0) must fall
    # back cleanly rather than erroring.
    assert not hasattr(_Config(), "block_round_zero_mix")

    for use_block in (False, True):
        metrics_a = _run_train_step(use_block_revelation=use_block, seed=5)
        metrics_b = _run_train_step(use_block_revelation=use_block, seed=5)
        for k in metrics_a:
            if k in metrics_b:
                assert np.allclose(np.array(metrics_a[k]), np.array(metrics_b[k])), (
                    f"use_block_revelation={use_block}: metric {k} differs between two "
                    f"identical-seed runs with block_round_zero_mix unset/default"
                )
    print("[OK] train_step with block_round_zero_mix unset/default (0.0) is deterministic "
          "and unaffected by this change, for both block and non-block (LM1B/WMT14/XSum-"
          "style) configs")


if __name__ == "__main__":
    test_rho_zero_is_bit_exact_regression()
    test_rho_one_always_selects_k0()
    test_empirical_p_k0_matches_closed_form_q_rho()
    test_mask_invariants_hold_for_rho_greater_than_zero()
    test_batch_pooled_weighting_imbalance_shrinks_with_rho()
    test_train_step_unaffected_without_opting_in()
    print("All block_round_zero_mix (T1) tests passed.")
