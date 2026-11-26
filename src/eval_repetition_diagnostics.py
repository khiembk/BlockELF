#!/usr/bin/env python
"""Offline repetition/quality diagnostics for already-saved unconditional-generation
JSONL files (e.g. eval_block_uncond.py's --out_path). Does NOT run the flow model or
any GPU inference -- reads exactly the same generated text that was/will be passed to
eval_mauve.py, so these numbers are always about the exact same sample, never a
re-generated or filtered one.

All n-gram statistics use the GPT-2 ("gpt2", small -- the tokenizer is identical across
all GPT-2 sizes) tokenization, for consistency with MAUVE's own featurizer tokenizer --
NOT a word-level or T5 tokenization. No sample is ever dropped, deduplicated, or edited
before being counted; empty strings are reported, not excluded.

Metric definitions (exact, to avoid ambiguity):
  seq-rep-N: for a single sample, the fraction of its N-gram OCCURRENCES (not distinct
    N-grams) that are a repeat of an earlier occurrence of the identical N-gram further
    left in the SAME sample. Equivalently: 1 - (distinct N-grams / total N-gram
    occurrences). Denominator = max(0, num_tokens - N + 1). A sample with fewer than N
    tokens contributes no N-gram occurrences (excluded from that metric's mean, counted
    in the "too-short-for-metric" diagnostic instead).
  distinct-N: unique N-grams / total N-gram occurrences, per sample (not pooled across
    the corpus) -- the complement of seq-rep-N, reported separately since different
    papers/conventions read one or the other more naturally.
  "truncated-to-canvas" rate: an ESTIMATE, not exact -- eval_block_uncond.py's saved
    JSONL stores only the final decoded string (`skip_special_tokens=True`), so whether
    a sample stopped via EOS or hit the model's max_length canvas limit is not directly
    recoverable from the text alone (EOS is stripped either way). This script estimates
    it via token length: a sample within `--canvas_length_margin` GPT-2 tokens of
    `--canvas_length` is counted as likely canvas-truncated. This is a proxy; exact
    EOS-vs-canvas accounting would require the generator to additionally export
    committed_len/has_eos per sample, which this task intentionally does not add (would
    require changing generate_block_uncond's return contract, used by other callers).

Usage:
    cd src && python eval_repetition_diagnostics.py \
        --generated_path /path/to/all_generated.jsonl \
        --window_sizes 128 256 \
        --looping_thresholds 0.3 0.5 0.7 \
        --report_path results.json
"""
import argparse
import json
import logging
import math
import sys
from collections import Counter

REPO_ROOT = __file__.rsplit("/", 1)[0]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
log = logging.getLogger(__name__)


def load_generated_texts(path, text_field="generated"):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_field not in obj:
                raise KeyError(f"'{text_field}' not found in JSONL row keys: {list(obj.keys())}")
            texts.append(obj[text_field])
    return texts


def ngram_occurrence_stats(token_ids, n):
    """Returns (seq_rep_n, distinct_n, num_occurrences) for one sample's token_ids, per
    the exact definitions in this module's docstring. (None, None, 0) if the sample has
    fewer than n tokens (no n-gram occurrences at all)."""
    num_occurrences = len(token_ids) - n + 1
    if num_occurrences <= 0:
        return None, None, 0
    ngrams = [tuple(token_ids[i:i + n]) for i in range(num_occurrences)]
    counts = Counter(ngrams)
    distinct_n = len(counts) / num_occurrences
    seq_rep_n = 1.0 - distinct_n
    return seq_rep_n, distinct_n, num_occurrences


def sliding_window_entropy(token_ids, window_size, stride=None):
    """Shannon entropy (bits) of the token-frequency distribution within each
    (optionally overlapping, stride<window_size) window, in sequence order. Returns a
    list of (window_start_position, entropy) -- lets callers see WHERE entropy
    collapses relative to the sample's start, not just a single aggregate number."""
    stride = stride or window_size
    results = []
    for start in range(0, max(1, len(token_ids) - window_size + 1), stride):
        window = token_ids[start:start + window_size]
        if not window:
            continue
        counts = Counter(window)
        n = len(window)
        ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
        results.append((start, ent))
    return results


def analyze_sample(text, gpt2_tokenizer, ngram_sizes, window_sizes, canvas_length, canvas_length_margin):
    is_empty = not text.strip()
    token_ids = gpt2_tokenizer.encode(text) if not is_empty else []
    length = len(token_ids)
    record = {
        "empty": is_empty,
        "length_tokens": length,
        "truncated_to_canvas_estimate": (canvas_length - length) <= canvas_length_margin,
    }
    for n in ngram_sizes:
        seq_rep_n, distinct_n, num_occ = ngram_occurrence_stats(token_ids, n)
        record[f"seq_rep_{n}"] = seq_rep_n
        record[f"distinct_{n}"] = distinct_n
        record[f"num_{n}gram_occurrences"] = num_occ
    record["window_entropy"] = {
        w: sliding_window_entropy(token_ids, w) for w in window_sizes
    }
    return record


def _mean_ignore_none(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, int(p * len(vals)))
    return vals[idx]


def aggregate(records, ngram_sizes, looping_thresholds, looping_metric="seq_rep_4"):
    n = len(records)
    empty_n = sum(1 for r in records if r["empty"])
    lengths = [r["length_tokens"] for r in records]
    duplicate_n = len(records) - len({r.get("_text_for_dup", i) for i, r in enumerate(records)})

    agg = {
        "n_samples": n,
        "empty_count": empty_n,
        "empty_fraction": empty_n / n if n else 0.0,
        "length_tokens": {
            "mean": _mean_ignore_none(lengths),
            "median": _percentile(lengths, 0.5),
            "p5": _percentile(lengths, 0.05),
            "p95": _percentile(lengths, 0.95),
        },
        "truncated_to_canvas_fraction": sum(
            1 for r in records if r["truncated_to_canvas_estimate"]
        ) / n if n else 0.0,
    }
    for ngram_n in ngram_sizes:
        seq_rep_vals = [r[f"seq_rep_{ngram_n}"] for r in records]
        distinct_vals = [r[f"distinct_{ngram_n}"] for r in records]
        too_short = sum(1 for v in seq_rep_vals if v is None)
        agg[f"seq_rep_{ngram_n}"] = {
            "mean": _mean_ignore_none(seq_rep_vals),
            "median": _percentile([v for v in seq_rep_vals if v is not None], 0.5),
            "too_short_for_metric_count": too_short,
        }
        agg[f"distinct_{ngram_n}"] = {
            "mean": _mean_ignore_none(distinct_vals),
            "median": _percentile([v for v in distinct_vals if v is not None], 0.5),
        }
    agg["looping_metric"] = looping_metric
    agg["looping_fraction_by_threshold"] = {}
    for thr in looping_thresholds:
        vals = [r[looping_metric] for r in records if r[looping_metric] is not None]
        looping = sum(1 for v in vals if v > thr)
        agg["looping_fraction_by_threshold"][thr] = looping / len(vals) if vals else None
    return agg


def position_profile(records, window_size):
    """Pools window-start -> mean-entropy across ALL samples (for the window_size that
    was computed), to see where entropy tends to collapse relative to sample start --
    answers "is this visible early (round 0-ish) or does it emerge later" without
    needing to re-run generation."""
    by_start = {}
    for r in records:
        for start, ent in r["window_entropy"].get(window_size, []):
            by_start.setdefault(start, []).append(ent)
    return {start: sum(vals) / len(vals) for start, vals in sorted(by_start.items())}


def main():
    ap = argparse.ArgumentParser(description="Offline repetition/quality diagnostics (no GPU, no regeneration)")
    ap.add_argument("--generated_path", required=True)
    ap.add_argument("--text_field", default="generated")
    ap.add_argument("--tokenizer_name", default="gpt2",
                    help="Tokenizer for n-gram/entropy stats -- 'gpt2' (shared vocab with gpt2-large) "
                         "for consistency with the MAUVE featurizer, NOT the T5 tokenizer.")
    ap.add_argument("--ngram_sizes", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--window_sizes", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--looping_thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--looping_metric", default="seq_rep_4")
    ap.add_argument("--canvas_length", type=int, default=1024)
    ap.add_argument("--canvas_length_margin", type=int, default=16,
                    help="A sample within this many GPT-2 tokens of --canvas_length is counted "
                         "as likely canvas-truncated (estimate; see module docstring).")
    ap.add_argument("--print_examples", type=int, default=3)
    ap.add_argument("--report_path", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    gpt2_tok = AutoTokenizer.from_pretrained(args.tokenizer_name)

    texts = load_generated_texts(args.generated_path, args.text_field)
    log.info(f"Loaded {len(texts)} generated texts from {args.generated_path}")

    duplicate_count = len(texts) - len(set(texts))
    records = []
    for text in texts:
        rec = analyze_sample(
            text, gpt2_tok, args.ngram_sizes, args.window_sizes,
            args.canvas_length, args.canvas_length_margin,
        )
        rec["_text_for_dup"] = text
        records.append(rec)

    agg = aggregate(records, args.ngram_sizes, args.looping_thresholds, args.looping_metric)
    agg["duplicate_count"] = duplicate_count
    agg["duplicate_fraction"] = duplicate_count / len(texts) if texts else 0.0
    agg["generated_path"] = args.generated_path
    agg["tokenizer_name"] = args.tokenizer_name

    position_profiles = {w: position_profile(records, w) for w in args.window_sizes}
    agg["position_entropy_profile"] = position_profiles

    log.info("=" * 60)
    log.info(f"N samples: {agg['n_samples']}  empty: {agg['empty_count']} ({agg['empty_fraction']:.1%})  "
              f"duplicate: {agg['duplicate_count']} ({agg['duplicate_fraction']:.1%})")
    log.info(f"Length (GPT-2 tokens): mean={agg['length_tokens']['mean']:.1f} "
              f"median={agg['length_tokens']['median']} p5={agg['length_tokens']['p5']} "
              f"p95={agg['length_tokens']['p95']}")
    log.info(f"Truncated-to-canvas (estimate): {agg['truncated_to_canvas_fraction']:.1%}")
    for n in args.ngram_sizes:
        sr = agg[f"seq_rep_{n}"]
        log.info(f"seq-rep-{n}: mean={sr['mean']:.4f} median={sr['median']:.4f} "
                  f"(too-short-for-metric: {sr['too_short_for_metric_count']})")
    for thr, frac in agg["looping_fraction_by_threshold"].items():
        log.info(f"Looping fraction ({agg['looping_metric']} > {thr}): "
                  f"{frac:.1%}" if frac is not None else f"Looping fraction ({agg['looping_metric']} > {thr}): n/a")
    for w, profile in position_profiles.items():
        if not profile:
            continue
        starts = sorted(profile)
        first_e, last_e = profile[starts[0]], profile[starts[-1]]
        log.info(f"Window-{w} entropy: start={first_e:.3f} bits, end={last_e:.3f} bits "
                  f"({'collapses toward the end' if last_e < first_e - 0.3 else 'roughly flat / no clear collapse'})")
    log.info("=" * 60)

    printed = 0
    for i, text in enumerate(texts):
        if printed >= args.print_examples:
            break
        snippet_len = 200
        log.info(f"[{i}] seq_rep_4={records[i]['seq_rep_4']} length={records[i]['length_tokens']}")
        log.info(f"    BEGIN: {text[:snippet_len]!r}")
        mid = len(text) // 2
        log.info(f"    MID:   {text[mid:mid + snippet_len]!r}")
        log.info(f"    END:   {text[-snippet_len:]!r}")
        printed += 1

    if args.report_path:
        # Drop the raw per-sample text we stashed only for duplicate-counting before
        # serializing (keeps the report focused on aggregate/diagnostic numbers).
        for r in records:
            r.pop("_text_for_dup", None)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump({"aggregate": agg, "per_sample": records}, f, indent=2)
        log.info(f"Saved full diagnostics report to {args.report_path}")


if __name__ == "__main__":
    main()
