"""Diagnostics and provenance helpers for MAUVE evaluation (see eval_mauve.py).

Kept separate from metrics_utils.py's `compute_mauve` (the raw score wrapper around the
`mauve` package) so these plain, CPU-only, dependency-light functions can be unit
tested without needing `mauve`/GPU/network -- only `transformers`' tokenizer and the
stdlib are required.
"""
import random
from collections import Counter


def sample_reference_texts(dataset, tokenizer, n_samples, seed):
    """Deterministically sample `n_samples` distinct rows from `dataset` (sampling
    WITHOUT replacement via `random.Random(seed).sample`, so no index -- and therefore
    no reference text -- can repeat within one call), and decode each via its own
    recorded `sequence_length` column (falling back to the full `input_ids` length if
    the column is absent).

    This `ids[:seq_len]` slice-before-decode is what makes EOS-used-as-pad-token safe:
    `pad_token: eos` (BlockELF's LM1B/OWT configs) means any position past the real
    content is EOS, identical to a real end-of-sequence token. Slicing to the recorded
    true length *before* decoding guarantees only the real content is kept, independent
    of `skip_special_tokens` -- decoding the full, unsliced array and relying on
    `skip_special_tokens=True` to strip trailing EOS/pad would be the fragile version
    of this (it works for simple trailing-pad cases but is not what this function
    does; it is called out here because it's the anti-pattern this guards against).

    Returns (texts, indices) -- `indices` is exposed so callers can fingerprint/report
    exactly which rows were sampled (reproducibility / duplicate-across-runs checks).
    """
    n = min(n_samples, len(dataset))
    indices = random.Random(seed).sample(range(len(dataset)), n)
    texts = []
    for i in indices:
        row = dataset[i]
        seq_len = row.get("sequence_length", len(row["input_ids"]))
        ids = row["input_ids"][:seq_len]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts, indices


def text_length_stats(texts, judge_tokenizer, max_text_length):
    """Length distribution of `texts` under the MAUVE *judge* tokenizer (GPT-2 by
    default) -- deliberately NOT the T5 tokenizer BlockELF trains/decodes with, since
    that's the tokenization MAUVE's featurizer actually sees and truncates by. Also
    reports the empty-text fraction and how many texts would be truncated by
    `max_text_length` (the window actually passed to `mauve.compute_mauve`).
    """
    lengths = [len(judge_tokenizer(t)["input_ids"]) for t in texts]
    n = len(texts)
    lengths_sorted = sorted(lengths)

    def pct(p):
        if not lengths_sorted:
            return 0
        idx = min(len(lengths_sorted) - 1, int(p * len(lengths_sorted)))
        return lengths_sorted[idx]

    empty = sum(1 for t in texts if not t.strip())
    truncated = sum(1 for L in lengths if L > max_text_length)
    return {
        "n": n,
        "mean_judge_tokens": (sum(lengths) / n) if n else 0.0,
        "median_judge_tokens": pct(0.5),
        "p10_judge_tokens": pct(0.10),
        "p90_judge_tokens": pct(0.90),
        "empty_fraction": (empty / n) if n else 0.0,
        "truncated_by_judge_window_fraction": (truncated / n) if n else 0.0,
    }


def duplicate_fraction(texts):
    """Fraction of `texts` that are exact-string duplicates of an earlier entry in the
    same list (0.0 if every text is unique). A nonzero value for a freshly-sampled
    reference set (sampled without replacement, per `sample_reference_texts`) would
    mean the underlying dataset itself contains duplicate rows, not a sampling bug.
    """
    n = len(texts)
    if n == 0:
        return 0.0
    counts = Counter(texts)
    duplicates = sum(c - 1 for c in counts.values() if c > 1)
    return duplicates / n


def library_versions(module_names=("mauve", "torch", "transformers", "faiss")):
    versions = {}
    for mod_name in module_names:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not installed"
    return versions


def validate_reference_split_args(reference_split_label, allow_train_split_as_reference):
    """Raises ValueError unless using the TRAIN split as the MAUVE human-reference
    distribution was an explicit, documented opt-in -- some BlockELF datasets (e.g. the
    cached embedded-language-flows/openwebtext-t5) have no separate held-out split
    available, and silently treating train data as "real/held-out" text would hide a
    real protocol limitation (soft risk of comparing the model to data it may have
    trained on) rather than exposing it. No-op for any other split label.
    """
    if reference_split_label == "train" and not allow_train_split_as_reference:
        raise ValueError(
            "reference_split_label='train' requires allow_train_split_as_reference=True -- "
            "using training data as the human-reference distribution is a real protocol "
            "limitation that must be a documented, opt-in choice, not a silent default. If a "
            "held-out validation/test split is available for this dataset, use that instead."
        )


def build_provenance_report(
    *, generated_path, generated_texts, reference_dataset_path, reference_texts,
    reference_indices, reference_split_label, judge_tokenizer, featurize_model_name,
    max_text_length, seed, mauve_score,
):
    """Assembles the full protocol/provenance record the audit requires: exact N,
    reference split label + sampled-row indices, measured length distributions, empty
    and duplicate rates, judge protocol (model/window/seed), and installed package
    versions -- everything needed to tell whether a *later* run is actually comparable
    to this one, or to catch a suspiciously high/low score before trusting it.
    """
    gen_stats = text_length_stats(generated_texts, judge_tokenizer, max_text_length)
    ref_stats = text_length_stats(reference_texts, judge_tokenizer, max_text_length)
    return {
        "mauve_score": mauve_score,
        "protocol": {
            "featurize_model_name": featurize_model_name,
            "max_text_length": max_text_length,
            "mauve_seed": seed,
            "num_buckets": "auto",
        },
        "generated": {
            "path": generated_path,
            "n": len(generated_texts),
            "duplicate_fraction": duplicate_fraction(generated_texts),
            **gen_stats,
        },
        "reference": {
            "dataset": reference_dataset_path,
            "split_label": reference_split_label,
            "n": len(reference_texts),
            "sampled_indices": sorted(reference_indices),
            "duplicate_fraction": duplicate_fraction(reference_texts),
            **ref_stats,
        },
        "library_versions": library_versions(),
    }
