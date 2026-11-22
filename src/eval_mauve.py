#!/usr/bin/env python
"""MAUVE evaluation for unconditional generation (OWT/LM1B): compares the distribution
of already-generated text (from eval_block_uncond.py's --out_path) against a sample of
real text from the task's own dataset. Unlike BLEU/ROUGE (used for the conditional
tasks), MAUVE needs no per-example reference -- it scores the generated SAMPLE as a
whole against a real-text SAMPLE, which is the right comparison for unconditional
generation (there's no ground-truth continuation per generated example).

Does not run any generation itself -- point --generated_path at an existing
eval_block_uncond.py --out_path JSONL (one {"generated": ...} object per line). Does
not touch train.py, train_step.py, generation.py, or eval_block_uncond.py; this is a
read-only offline scorer over already-produced artifacts.

Requires the optional `mauve-text` package (see requirements-eval-mauve.txt) -- NOT a
dependency of training/inference/other eval paths.

Every run prints and (optionally) saves a full provenance report: exact N, reference
split/row indices, measured length distributions under the MAUVE judge tokenizer,
empty/duplicate rates, judge protocol (model/window/seed), and installed package
versions -- so a later run can be checked for comparability before trusting a score
delta, and a suspicious score can be audited instead of accepted at face value.
"""
import argparse
import json
import logging
import sys

REPO_ROOT = __file__.rsplit("/", 1)[0]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoTokenizer

from utils.data_utils import load_dataset_split
from utils.mauve_eval_utils import (
    build_provenance_report, sample_reference_texts, validate_reference_split_args,
)
from utils.metrics_utils import compute_mauve
from utils.logging_utils import log_for_0

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)


def parse_args():
    p = argparse.ArgumentParser(description="MAUVE score for an unconditional generation sample")
    p.add_argument("--generated_path", type=str, required=True,
                   help="JSONL from eval_block_uncond.py's --out_path (one {'generated': ...} per line).")
    p.add_argument("--reference_dataset", type=str, required=True,
                   help="HF repo id or local load_from_disk path for the real-text reference sample "
                        "(e.g. the task's own data_path/eval_data_path -- must be the SAME task as "
                        "--generated_path: never compare OWT generations to an LM1B reference or "
                        "vice versa, and never use WMT14/XSum data here).")
    p.add_argument("--reference_split_label", type=str, required=True,
                   choices=["train", "validation", "test", "other"],
                   help="What split --reference_dataset actually is. Required (not auto-detected) "
                        "so this is a documented, explicit protocol choice rather than a silent one. "
                        "If 'train', you must also pass --allow_train_split_as_reference.")
    p.add_argument("--allow_train_split_as_reference", action="store_true", default=False,
                   help="Required alongside --reference_split_label train. Some BlockELF datasets "
                        "(e.g. the cached embedded-language-flows/openwebtext-t5) have no separate "
                        "held-out split available in this environment -- using train data as the "
                        "human-reference distribution is a real limitation (soft risk of comparing "
                        "the model to data it may have trained on), not a silently-fine default.")
    p.add_argument("--tokenizer_name", type=str, default="t5-small",
                   help="Tokenizer the reference dataset's input_ids were encoded with (NOT the MAUVE "
                        "judge tokenizer, which is always GPT-2/GPT-2-large per --featurize_model_name).")
    p.add_argument("--num_reference_samples", type=int, default=None,
                   help="How many real-text examples to sample as the reference set. "
                        "Defaults to the generated sample's size (recommended: equal sizes).")
    p.add_argument("--featurize_model_name", type=str, default="gpt2-large")
    p.add_argument("--max_text_length", type=int, default=1024,
                   help="MAUVE judge truncation window, measured in the featurizer's OWN "
                        "(GPT-2) tokenization -- not T5 tokens. Must match across any scores "
                        "being compared to each other.")
    p.add_argument("--device_id", type=int, default=0,
                   help="GPU device index for MAUVE's GPT-2 featurizer; -1 for CPU.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seeds both the reference-row sampling and MAUVE's own k-means step.")
    p.add_argument("--report_path", type=str, default=None,
                   help="Optional path to save the full provenance report as JSON.")
    return p.parse_args()


def main():
    args = parse_args()

    validate_reference_split_args(args.reference_split_label, args.allow_train_split_as_reference)

    generated_texts = []
    with open(args.generated_path, "r", encoding="utf-8") as f:
        for line in f:
            generated_texts.append(json.loads(line)["generated"])
    log_for_0(f"Loaded {len(generated_texts)} generated texts from {args.generated_path}")

    n_ref = args.num_reference_samples or len(generated_texts)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    ds = load_dataset_split(args.reference_dataset)
    log_for_0(f"Reference dataset {args.reference_dataset} "
              f"(split_label={args.reference_split_label}): {len(ds)} rows")
    if args.reference_split_label == "train":
        log_for_0(
            "WARNING: using the TRAIN split as the MAUVE human-reference distribution "
            "(--allow_train_split_as_reference was set). This is a real protocol limitation, "
            "not a clean held-out comparison -- recorded in the provenance report."
        )

    reference_texts, reference_indices = sample_reference_texts(ds, tokenizer, n_ref, args.seed)
    log_for_0(f"Sampled {len(reference_texts)} real reference texts (seed={args.seed})")

    if len(reference_texts) != len(generated_texts):
        log_for_0(
            f"WARNING: reference sample size ({len(reference_texts)}) != generated sample size "
            f"({len(generated_texts)}) -- the reference dataset had fewer than "
            f"{len(generated_texts)} rows available. Recorded in the provenance report; "
            f"consider this before comparing scores to a matched-size run."
        )

    mauve_score = compute_mauve(
        generated_texts=generated_texts, reference_texts=reference_texts,
        featurize_model_name=args.featurize_model_name, max_text_length=args.max_text_length,
        device_id=args.device_id, verbose=True, seed=args.seed,
    )

    judge_tokenizer_name = args.featurize_model_name if "gpt2" in args.featurize_model_name else "gpt2"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_tokenizer_name)
    report = build_provenance_report(
        generated_path=args.generated_path, generated_texts=generated_texts,
        reference_dataset_path=args.reference_dataset, reference_texts=reference_texts,
        reference_indices=reference_indices, reference_split_label=args.reference_split_label,
        judge_tokenizer=judge_tokenizer, featurize_model_name=args.featurize_model_name,
        max_text_length=args.max_text_length, seed=args.seed, mauve_score=mauve_score,
    )

    log_for_0("=" * 60)
    log_for_0(f"Generated sample: {args.generated_path} (n={len(generated_texts)})")
    log_for_0(f"Reference sample: {args.reference_dataset} "
              f"(split_label={args.reference_split_label}, n={len(reference_texts)})")
    log_for_0(f"MAUVE: {mauve_score:.4f}")
    log_for_0(f"Generated length (judge tokens): mean={report['generated']['mean_judge_tokens']:.1f} "
              f"median={report['generated']['median_judge_tokens']} "
              f"p10={report['generated']['p10_judge_tokens']} p90={report['generated']['p90_judge_tokens']}")
    log_for_0(f"Reference length (judge tokens): mean={report['reference']['mean_judge_tokens']:.1f} "
              f"median={report['reference']['median_judge_tokens']} "
              f"p10={report['reference']['p10_judge_tokens']} p90={report['reference']['p90_judge_tokens']}")
    log_for_0(f"Generated empty/duplicate fraction: "
              f"{report['generated']['empty_fraction']:.4f} / {report['generated']['duplicate_fraction']:.4f}")
    log_for_0(f"Reference empty/duplicate fraction: "
              f"{report['reference']['empty_fraction']:.4f} / {report['reference']['duplicate_fraction']:.4f}")
    log_for_0("=" * 60)

    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log_for_0(f"Saved full provenance report to {args.report_path}")


if __name__ == "__main__":
    main()
