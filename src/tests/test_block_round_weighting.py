"""Training-objective positional-weight analysis (OWT repetition-collapse follow-up
spec §1.3).

`sample_block_round` draws k ~ Uniform{0,...,K-1} per example; `reduce_token_loss`
(train_step.py) averages the per-token loss uniformly over that round's active region
(blocks k..K-1). For a fixed target BLOCK j (0-indexed; block j covers absolute target
positions [j*b, (j+1)*b)), block j is only "active" (supervised) in rounds k<=j, and
when active its per-token loss weight for that round is 1/((K-k)*b) (uniform over the
active region's size). The expected per-token loss weight for a token in block j,
marginalizing over the K equally-likely choices of k, is:

    w_j = (1 / (K*b)) * sum_{k=0}^{j} 1/(K-k),   j = 0, ..., K-1

This is an EXPECTED PER-TOKEN OBJECTIVE WEIGHT under the current sampling scheme, not a
claim about actual model accuracy (parameters are shared across all positions/rounds).
At K=64: w_{K-1}/w_0 = K * H_K ~= 303.6 (H_K = the K-th harmonic number) -- the LAST
target block's expected training signal is ~300x the FIRST target block's. This is the
opposite direction of "the model barely learns later blocks" -- it's EARLY target
blocks that are barely ever supervised, because by the time k exceeds a small j, block j
has already been teacher-forced (committed, not noised) for the rest of that round's
possible k values.

**CORRECTION (post-review, see docs/owt_collapse_root_cause_findings.md's T1 section):**
the 303.6x figure above assumes `reduce_token_loss` normalizes PER EXAMPLE. It does not --
`train_step.py::reduce_token_loss` divides by `loss_mask.sum()` with NO axis argument,
pooling the ENTIRE per-device minibatch's active-token count into one shared scalar
denominator. A corrected Monte Carlo simulation using the real OWT per-device batch size
(12, from global_batch_size=24 / 2 local devices) and the real OWT target-length
distribution (sampled directly from the dataset) gives an effective late-vs-early ratio
of **~70x, not ~304x** -- the qualitative imbalance (late blocks dominate) is real and
confirmed, but this file's own per-example-isolated w_j significantly OVERSTATES its
magnitude. See `tests/test_block_round_zero_mix.py`'s
`test_batch_pooled_weighting_imbalance_shrinks_with_rho` for the corrected, batch-pooled
simulation (and confirmation that T1's `rho` mixture still shrinks the REAL imbalance,
just from a smaller starting point than originally claimed).

train.py's `compute_block_round_counts`/`sample_weights` corrects a DIFFERENT, orthogonal
bias (which EXAMPLE gets drawn into a batch, proportional to K_i, canceling the
small-K_i-examples-are-overrepresented-at-k=0 effect across the corpus) -- it does NOT
touch this within-sequence, fixed-K, positional imbalance.

This script only verifies/logs; it does NOT modify `sample_block_round` or the loss.

Pure-CPU. Run with:  cd src && python tests/test_block_round_weighting.py
"""
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from utils.block_utils import sample_block_round


def analytic_weight_profile(K: int, b: int = 1):
    """w_j (unnormalized by b, since b is a constant multiplicative factor -- the
    SHAPE across j is what matters) for j=0..K-1, plus the harmonic-number ratio."""
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, K + 1))])  # H[n] = H_n
    w = np.zeros(K)
    for j in range(K):
        # sum_{k=0}^{j} 1/(K-k) = sum_{m=K-j}^{K} 1/m = H_K - H_{K-j-1}
        w[j] = (H[K] - H[K - j - 1]) / (K * b)
    return w


def test_analytic_weight_formula_matches_direct_double_sum():
    """Closed-form H_K - H_{K-j-1} must equal the direct double sum it derives from."""
    K, b = 16, 4
    w_closed = analytic_weight_profile(K, b)
    w_direct = np.zeros(K)
    for j in range(K):
        w_direct[j] = sum(1.0 / (K - k) for k in range(j + 1)) / (K * b)
    assert np.allclose(w_closed, w_direct, atol=1e-10), "closed-form w_j disagrees with direct double sum"
    print(f"[OK] closed-form w_j (K={K}, b={b}) matches the direct sum_{{k=0}}^{{j}} 1/(K-k) double sum exactly")


def test_k64_ratio_matches_claimed_303x():
    K = 64
    w = analytic_weight_profile(K, b=16)
    ratio = w[-1] / w[0]
    H_K = sum(1.0 / i for i in range(1, K + 1))
    expected_ratio = K * H_K
    assert abs(ratio - expected_ratio) < 1e-6
    assert 290 < ratio < 315, f"K=64 last/first weight ratio {ratio:.2f} far from the expected ~303.6"
    print(f"[OK] K=64 expected per-token weight ratio w_last/w_first = {ratio:.2f} "
          f"(K*H_K={expected_ratio:.2f}) -- matches the task's claimed ~303.6x")


def test_empirical_active_frequency_matches_closed_form():
    """Monte Carlo: run the REAL sample_block_round many times on a synthetic
    full-length (no padding, no cond prefix) example and empirically estimate
    P(target block j is active | k sampled), comparing against the closed-form
    (j+1)/K per-block active-probability (the piece w_j's derivation depends on)."""
    K, b, trials = 16, 4, 20000
    S = K * b
    cond_seq_mask = jnp.zeros((1, S))
    attention_mask = jnp.ones((1, S))
    pos = np.arange(S)

    k_histogram = collections.Counter()
    block_active_count = np.zeros(K)
    rng = jax.random.PRNGKey(0)
    for trial in range(trials):
        rng, round_rng = jax.random.split(rng)
        committed, active, boundary = sample_block_round(round_rng, cond_seq_mask, attention_mask, b)
        k = int(np.array(boundary)[0] / b)
        k_histogram[k] += 1
        active_np = np.array(active)[0]
        for j in range(K):
            block_pos = pos[j * b:(j + 1) * b]
            if active_np[block_pos].all():
                block_active_count[j] += 1
            elif active_np[block_pos].any():
                raise AssertionError(f"block {j} partially active -- committed/active split not block-aligned")

    empirical_active_freq = block_active_count / trials
    closed_form_active_freq = (np.arange(K) + 1) / K

    log_for_0 = print
    log_for_0(f"[LOG] sampled-k histogram over {trials} trials (K={K}): "
              f"{sorted(k_histogram.items())}")
    log_for_0(f"[LOG] per-target-block empirical active frequency (first/mid/last 3 of {K}): "
              f"{np.round(empirical_active_freq[:3], 4)} ... "
              f"{np.round(empirical_active_freq[K // 2 - 1:K // 2 + 2], 4)} ... "
              f"{np.round(empirical_active_freq[-3:], 4)}")

    max_abs_err = float(np.max(np.abs(empirical_active_freq - closed_form_active_freq)))
    assert max_abs_err < 0.02, (
        f"empirical per-block active frequency diverges from the closed-form (j+1)/K "
        f"by {max_abs_err:.4f} (more than Monte Carlo noise at {trials} trials)"
    )
    print(f"[OK] empirical per-target-block active frequency matches closed-form (j+1)/K "
          f"(max abs err={max_abs_err:.4f} over {trials} trials)")

    w = analytic_weight_profile(K, b)
    w_norm = w / w.sum()
    log_for_0(f"[LOG] expected normalized per-block loss-contribution profile w_j/sum(w) "
              f"(first/last 3 of {K}): {np.round(w_norm[:3], 5)} ... {np.round(w_norm[-3:], 5)}")
    log_for_0(f"[LOG] block 0 (earliest target positions) receives "
              f"{100 * w_norm[0] / w_norm.mean():.1f}% of the PER-BLOCK-UNIFORM share; "
              f"block {K-1} (latest) receives {100 * w_norm[-1] / w_norm.mean():.1f}%")


def _try_log_observed_position_entropy_vs_theory():
    """Optional cross-reference (not an assertion): if a previously-saved repetition
    diagnostics report with a position_entropy_profile is available on disk, print it
    next to the theoretical w_j curve so a human can see whether the early-block
    under-weighting hypothesis is directionally consistent with where observed entropy
    actually degrades. Silently skipped if no such file is found (this is NOT a new
    GPU-dependent experiment, purely a convenience cross-reference of already-saved
    diagnostics)."""
    import glob
    import json

    candidates = glob.glob("/tmp/claude-*/**/scratchpad/**/*diagnostics*.json", recursive=True)
    if not candidates:
        print("[INFO] no saved position_entropy_profile diagnostics file found on disk -- skipping cross-reference")
        return
    path = sorted(candidates)[0]
    try:
        with open(path) as f:
            d = json.load(f)
        agg = d.get("aggregate", d)
        profiles_by_window = agg.get("position_entropy_profile")
        if not profiles_by_window:
            print(f"[INFO] {path} has no position_entropy_profile field -- skipping cross-reference")
            return
        window_size, profile = sorted(profiles_by_window.items())[0]
        if isinstance(profile, dict):
            profile = sorted(((int(k), v) for k, v in profile.items()))
        print(f"[LOG] observed window-{window_size} entropy profile from {path} "
              f"([window_start, entropy_bits] pairs), first/last few: "
              f"{profile[:3]} ... {profile[-3:]}")
        print("[LOG] theory predicts EARLY target blocks (low j) are the most under-"
              "supervised (lowest w_j); whether that corresponds to the observed early-"
              "vs-late entropy curve is evidence toward (not proof of) the training-"
              "imbalance hypothesis -- see docs/owt_collapse_root_cause_findings.md "
              "for the actual cross-reference and verdict.")
    except Exception as e:
        print(f"[INFO] could not read {path} for cross-reference: {e}")


if __name__ == "__main__":
    test_analytic_weight_formula_matches_direct_double_sum()
    test_k64_ratio_matches_claimed_303x()
    test_empirical_active_frequency_matches_closed_form()
    _try_log_observed_position_entropy_vs_theory()
    print("All block-round positional-weight tests passed.")
