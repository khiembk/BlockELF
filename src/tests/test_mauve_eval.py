"""Sanity tests for MAUVE evaluation (eval_mauve.py / utils.mauve_eval_utils /
utils.metrics_utils.compute_mauve).

Pure-CPU, no GPU/TPU required. Most tests use fake/stub tokenizers and mock
`mauve.compute_mauve` directly (so they run with or without the real `mauve` package
installed); the two `test_mauve_*_smoke` tests at the bottom do a real, tiny MAUVE
computation (featurize_model_name="gpt2", CPU) and are skipped if `mauve` isn't
installed. Run with:

    cd src && python tests/test_mauve_eval.py

Covers:
  1. Deterministic, duplicate-free reference sampling (same seed -> same indices).
  2. EOS-as-pad decode regression: `sequence_length`-sliced decode never leaks
     padding/EOS content past the real sequence.
  3. duplicate_fraction / text_length_stats correctness on controlled inputs.
  4. library_versions handles "not installed" modules without crashing.
  5. compute_mauve: correct p_text/q_text orientation, single call (not per-batch),
     seed passthrough, and reading `out.mauve` (not frontier_integral or a rescaling).
  6. compute_mauve raises a clear, actionable ImportError when `mauve` isn't installed.
  7. The train-split-as-reference guard (validate_reference_split_args).
  8. Provenance report contains every field the audit's acceptance criteria require.
  9. (smoke, skipped if `mauve` unavailable) identical-distribution MAUVE is high;
     a clearly mismatched distribution scores lower.
"""
import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mauve_eval_utils import (
    build_provenance_report, duplicate_fraction, library_versions,
    sample_reference_texts, text_length_stats, validate_reference_split_args,
)
from utils.metrics_utils import compute_mauve


class _FakeTokenizer:
    """Records exactly which token-id list it was asked to decode (as its own
    string representation), so tests can assert on the precise slice passed in --
    rather than depending on any real tokenizer's decode semantics."""

    def decode(self, ids, skip_special_tokens=True):
        return str(list(ids))


class _FakeJudgeTokenizer:
    """Word-count stand-in for a GPT-2 tokenizer: returns one "token" per
    whitespace-separated word, so length-stat tests don't need a real tokenizer."""

    def __call__(self, text):
        return {"input_ids": text.split()}


class _FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


def test_sample_reference_texts_deterministic_and_unique():
    rows = [{"input_ids": [i, i + 1, 1, 1, 1], "sequence_length": 2} for i in range(50)]
    ds = _FakeDataset(rows)
    tok = _FakeTokenizer()

    texts_a, idx_a = sample_reference_texts(ds, tok, n_samples=10, seed=7)
    texts_b, idx_b = sample_reference_texts(ds, tok, n_samples=10, seed=7)
    assert idx_a == idx_b, "same seed must give the same sampled indices"
    assert texts_a == texts_b, "same seed must give the same decoded texts"
    assert len(set(idx_a)) == 10, "sampling must draw 10 DISTINCT indices (no repeats)"

    texts_c, idx_c = sample_reference_texts(ds, tok, n_samples=10, seed=8)
    assert idx_c != idx_a, "a different seed should (overwhelmingly likely) give different indices"

    # n_samples larger than the dataset must not crash -- clamps to len(dataset).
    texts_all, idx_all = sample_reference_texts(ds, tok, n_samples=10_000, seed=0)
    assert len(idx_all) == len(rows)
    assert len(set(idx_all)) == len(rows)
    print("[OK] sample_reference_texts: deterministic given a seed, indices unique, "
          "clamps cleanly when n_samples > len(dataset)")


def test_sample_reference_texts_eos_as_pad_regression():
    # input_ids = [5, 6, EOS(=1), EOS(=1), EOS(=1)], sequence_length=3: the real
    # content is [5, 6, 1] (one genuine trailing EOS) -- the slice must NOT include
    # the two extra EOS-as-pad positions past sequence_length.
    rows = [{"input_ids": [5, 6, 1, 1, 1], "sequence_length": 3}]
    ds = _FakeDataset(rows)
    tok = _FakeTokenizer()
    texts, _ = sample_reference_texts(ds, tok, n_samples=1, seed=0)
    assert texts[0] == str([5, 6, 1]), (
        f"expected decode to be called with exactly the first 3 ids [5, 6, 1], got {texts[0]!r} "
        "-- EOS-used-as-pad-token content leaked past the recorded sequence_length"
    )

    # No sequence_length column at all: must fall back to the full input_ids length
    # (not crash, not silently truncate to 0).
    rows2 = [{"input_ids": [9, 9, 9]}]
    ds2 = _FakeDataset(rows2)
    texts2, _ = sample_reference_texts(ds2, tok, n_samples=1, seed=0)
    assert texts2[0] == str([9, 9, 9]), "missing sequence_length column must fall back to full length"
    print("[OK] sample_reference_texts: EOS-as-pad-token content never leaks past "
          "sequence_length; missing sequence_length column falls back to full length")


def test_duplicate_fraction():
    assert duplicate_fraction([]) == 0.0
    assert duplicate_fraction(["a", "b", "c"]) == 0.0
    # 5 items, "a" appears 3x (2 duplicates), "b" appears 2x (1 duplicate) -> 3/5.
    assert abs(duplicate_fraction(["a", "a", "a", "b", "b"]) - 3 / 5) < 1e-9
    print("[OK] duplicate_fraction: 0.0 for empty/all-unique, correct fraction with repeats")


def test_text_length_stats():
    texts = ["one two three", "", "a b c d e", "x"]
    stats = text_length_stats(texts, _FakeJudgeTokenizer(), max_text_length=4)
    assert stats["n"] == 4
    assert stats["mean_judge_tokens"] == (3 + 0 + 5 + 1) / 4
    assert stats["empty_fraction"] == 1 / 4, "exactly one of the 4 texts is empty"
    # Only "a b c d e" (5 tokens) exceeds max_text_length=4.
    assert stats["truncated_by_judge_window_fraction"] == 1 / 4
    print("[OK] text_length_stats: mean/empty-fraction/truncation-fraction correct "
          "on a controlled input")


def test_library_versions_handles_missing_module():
    versions = library_versions(module_names=("definitely_not_a_real_package_xyz", "sys"))
    assert versions["definitely_not_a_real_package_xyz"] == "not installed"
    assert versions["sys"] != "not installed"  # sys is always importable
    print("[OK] library_versions: reports 'not installed' for a missing module without crashing")


def test_compute_mauve_calls_api_correctly():
    """Mocks mauve.compute_mauve itself (works whether or not the real package is
    installed) and asserts: called exactly once (not per-batch), p_text=reference
    (human), q_text=generated (model) -- catching a reversed-argument bug -- seed
    passed through, and the return value comes from `.mauve` (not some other field)."""
    fake_mauve_module = MagicMock()
    fake_out = MagicMock()
    fake_out.mauve = 0.4242
    fake_out.frontier_integral = 999.0  # a different field; must NOT be what's returned
    fake_mauve_module.compute_mauve.return_value = fake_out

    with patch.dict(sys.modules, {"mauve": fake_mauve_module}):
        score = compute_mauve(
            generated_texts=["gen1", "gen2"], reference_texts=["ref1", "ref2"],
            featurize_model_name="gpt2-large", seed=7,
        )

    assert score == 0.4242, f"expected the `.mauve` field (0.4242), got {score!r}"
    assert fake_mauve_module.compute_mauve.call_count == 1, (
        "compute_mauve must be invoked exactly once on the full collected sample, "
        "not once per batch with scores averaged"
    )
    _, kwargs = fake_mauve_module.compute_mauve.call_args
    assert kwargs["p_text"] == ["ref1", "ref2"], "p_text must be the REFERENCE (human) texts"
    assert kwargs["q_text"] == ["gen1", "gen2"], "q_text must be the GENERATED (model) texts"
    assert kwargs["seed"] == 7, "seed must be passed through to MAUVE's own k-means step"
    assert kwargs["featurize_model_name"] == "gpt2-large"
    print("[OK] compute_mauve: p_text=reference/q_text=generated (not reversed), called "
          "exactly once, seed passed through, reads `.mauve` (not frontier_integral)")


def test_compute_mauve_raises_clear_error_when_mauve_not_installed():
    # sys.modules['mauve'] = None is the standard way to make `import mauve` raise
    # ImportError regardless of whether the real package is actually installed.
    with patch.dict(sys.modules, {"mauve": None}):
        try:
            compute_mauve(generated_texts=["a"], reference_texts=["b"])
            raised = False
        except ImportError as e:
            raised = True
            msg = str(e)
    assert raised, "compute_mauve must raise (not silently return a bad score) when mauve is missing"
    assert "pip install mauve-text" in msg, (
        f"error message should give an actionable install instruction, got: {msg!r}"
    )
    print("[OK] compute_mauve: raises a clear, actionable ImportError when `mauve` is not installed")


def test_validate_reference_split_args():
    # train + not allowed -> must raise.
    try:
        validate_reference_split_args("train", allow_train_split_as_reference=False)
        raised = False
    except ValueError:
        raised = True
    assert raised, "reference_split_label='train' without the explicit opt-in flag must raise"

    # train + explicitly allowed -> must NOT raise.
    validate_reference_split_args("train", allow_train_split_as_reference=True)

    # any non-train label -> must NOT raise regardless of the flag.
    validate_reference_split_args("validation", allow_train_split_as_reference=False)
    validate_reference_split_args("test", allow_train_split_as_reference=False)
    print("[OK] validate_reference_split_args: train split requires the explicit opt-in flag; "
          "validation/test splits never require it")


def test_provenance_report_contains_required_fields():
    generated = ["gen one", "gen two", "gen two"]  # one duplicate, on purpose
    reference = ["ref alpha", "ref beta"]
    report = build_provenance_report(
        generated_path="/fake/path/gen.jsonl", generated_texts=generated,
        reference_dataset_path="fake/dataset", reference_texts=reference,
        reference_indices=[5, 2], reference_split_label="test",
        judge_tokenizer=_FakeJudgeTokenizer(), featurize_model_name="gpt2-large",
        max_text_length=1024, seed=42, mauve_score=0.123,
    )

    assert report["mauve_score"] == 0.123
    assert report["protocol"]["featurize_model_name"] == "gpt2-large"
    assert report["protocol"]["max_text_length"] == 1024
    assert report["protocol"]["mauve_seed"] == 42

    gen = report["generated"]
    assert gen["path"] == "/fake/path/gen.jsonl"
    assert gen["n"] == 3
    assert abs(gen["duplicate_fraction"] - 1 / 3) < 1e-9
    assert "mean_judge_tokens" in gen and "empty_fraction" in gen

    ref = report["reference"]
    assert ref["dataset"] == "fake/dataset"
    assert ref["split_label"] == "test"
    assert ref["n"] == 2
    assert ref["sampled_indices"] == [2, 5]
    assert "duplicate_fraction" in ref and "mean_judge_tokens" in ref

    assert "library_versions" in report
    # Must round-trip through json.dumps (this is what eval_mauve.py --report_path writes).
    json.dumps(report)
    print("[OK] build_provenance_report: contains every field the audit's acceptance "
          "criteria require (N, split label, sampled indices, length stats, "
          "empty/duplicate rates, protocol, library versions), and is JSON-serializable")


def test_mauve_truncation_window_bias_regression():
    """Regression test for the measurement bug found and fixed in this session
    (compute_mauve's max_text_length default changed 1024 -> 768): mauve-text
    featurizes each text by its LAST token's hidden state only, so a window AT OR
    ABOVE the natural document-length distribution means almost nothing gets
    truncated and each text's "last token" lands at a different, data-dependent
    position -- this alone was measured to collapse real OWT MAUVE from ~0.8 to
    ~0.2-0.3, independent of generation quality (see commit history / sibling
    CorrFlow repo's b5b1d8f). This test isolates the mechanism synthetically: two
    text samples built from the IDENTICAL style/content (same base passage repeated
    a variable number of times), differing ONLY in their length *profile*
    (reference: wide length variance, like real documents; "generated": narrow,
    consistently near the top of the range, like a model filling its canvas). Since
    content/style is indistinguishable by construction, any MAUVE difference between
    the two window sizes below is attributable PURELY to the truncation-window
    mechanism, not a real quality gap.
    """
    try:
        import mauve  # noqa: F401
    except ImportError:
        print("[SKIP] mauve not installed -- skipping truncation-window regression test")
        return

    import random
    base = ("The quick brown fox jumps over the lazy dog near the river bank. "
            "Scientists announced a new discovery in the field of astrophysics today. ")
    rng = random.Random(0)
    reference_texts = [base * rng.randint(1, 20) for _ in range(100)]   # wide length variance
    generated_texts = [base * rng.randint(18, 20) for _ in range(100)]  # narrow, near-max length

    score_no_truncation = compute_mauve(
        generated_texts=generated_texts, reference_texts=reference_texts,
        featurize_model_name="gpt2", max_text_length=600, device_id=-1, seed=42,
    )
    score_consistent_truncation = compute_mauve(
        generated_texts=generated_texts, reference_texts=reference_texts,
        featurize_model_name="gpt2", max_text_length=150, device_id=-1, seed=42,
    )
    print(f"[INFO] same content/style, max_text_length=600 (no truncation): "
          f"MAUVE={score_no_truncation:.4f}")
    print(f"[INFO] same content/style, max_text_length=150 (consistent truncation): "
          f"MAUVE={score_consistent_truncation:.4f}")
    assert score_consistent_truncation > score_no_truncation + 0.2, (
        "a smaller max_text_length that forces consistent truncation should score "
        f"substantially higher than one that doesn't, on IDENTICAL-style text -- got "
        f"{score_consistent_truncation:.4f} (K=150) vs {score_no_truncation:.4f} (K=600); "
        "if this fails, the max_text_length default may need re-validating, not "
        "just this test's thresholds"
    )
    print("[OK] truncation-window regression: a too-large max_text_length measurably "
          "collapses MAUVE on same-style text purely via the last-token-position "
          "mechanism, confirming why the default was changed from 1024 to 768")


def test_mauve_identical_distribution_smoke():
    try:
        import mauve  # noqa: F401
    except ImportError:
        print("[SKIP] mauve not installed -- skipping real MAUVE smoke test")
        return

    texts = [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Scientists announced a new discovery in the field of astrophysics today.",
        "The local bakery introduced a new recipe for sourdough bread this week.",
        "A gentle breeze moved through the trees as the sun began to set.",
        "The committee will meet next Tuesday to discuss the proposed budget.",
        "Her latest novel explores themes of memory and loss across generations.",
        "The stadium was filled with fans cheering for the home team's victory.",
        "Researchers published a study on the effects of sleep on memory retention.",
        "The museum unveiled a new exhibit featuring ancient Roman artifacts.",
        "He spent the afternoon repairing the old wooden fence in the backyard.",
        "The city council approved funding for three new public parks this year.",
        "A sudden storm rolled in, forcing the outdoor concert to be postponed.",
        "The chef prepared a seven-course meal using only locally sourced ingredients.",
        "Volunteers gathered at dawn to clean up litter along the coastline.",
        "The orchestra performed a moving tribute to the late composer's work.",
    ]
    score_identical = compute_mauve(
        generated_texts=texts, reference_texts=texts,
        featurize_model_name="gpt2", max_text_length=256, device_id=-1, seed=42,
    )
    print(f"[INFO] MAUVE(p=q, identical collection, n={len(texts)}): {score_identical:.4f}")
    assert score_identical > 0.5, (
        f"MAUVE of a collection against itself should be high, got {score_identical:.4f}"
    )

    mismatched = ["1 2 3 4 5", "6 7 8 9 10", "11 12 13 14 15"] * 5
    score_mismatched = compute_mauve(
        generated_texts=mismatched, reference_texts=texts,
        featurize_model_name="gpt2", max_text_length=256, device_id=-1, seed=42,
    )
    print(f"[INFO] MAUVE(p=prose, q=numeric strings, n={len(texts)}): {score_mismatched:.4f}")
    assert score_mismatched < score_identical, (
        "a clearly mismatched distribution (numeric strings vs. prose) should score "
        f"lower than the identical-distribution case ({score_mismatched:.4f} vs "
        f"{score_identical:.4f})"
    )
    print("[OK] real MAUVE smoke test: identical distribution scores high, a clearly "
          "mismatched distribution scores lower")


if __name__ == "__main__":
    test_sample_reference_texts_deterministic_and_unique()
    test_sample_reference_texts_eos_as_pad_regression()
    test_duplicate_fraction()
    test_text_length_stats()
    test_library_versions_handles_missing_module()
    test_compute_mauve_calls_api_correctly()
    test_compute_mauve_raises_clear_error_when_mauve_not_installed()
    test_validate_reference_split_args()
    test_provenance_report_contains_required_fields()
    test_mauve_truncation_window_bias_regression()
    test_mauve_identical_distribution_smoke()
    print("\nAll MAUVE eval sanity tests passed.")
