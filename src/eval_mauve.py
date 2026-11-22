#!/usr/bin/env python
"""MAUVE evaluation for unconditional generation (OWT/LM1B): compares the distribution
of already-generated text (from eval_block_uncond.py's --out_path) against a sample of
real text from the task's own dataset. Unlike BLEU/ROUGE (used for the conditional
tasks), MAUVE needs no per-example reference -- it scores the generated SAMPLE as a
whole against a real-text SAMPLE, which is the right comparison for unconditional
generation (there's no ground-truth continuation per generated example).

Does not run any generation itself -- point --generated_path at an existing
eval_block_uncond.py --out_path JSONL (one {"generated": ...} object per line).
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
                        "(e.g. the task's own data_path).")
    p.add_argument("--tokenizer_name", type=str, default="t5-small")
    p.add_argument("--num_reference_samples", type=int, default=None,
                   help="How many real-text examples to sample as the reference set. "
                        "Defaults to the generated sample's size (recommended: equal sizes).")
    p.add_argument("--featurize_model_name", type=str, default="gpt2-large")
    p.add_argument("--max_text_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    generated_texts = []
    with open(args.generated_path, "r", encoding="utf-8") as f:
        for line in f:
            generated_texts.append(json.loads(line)["generated"])
    log_for_0(f"Loaded {len(generated_texts)} generated texts from {args.generated_path}")

    n_ref = args.num_reference_samples or len(generated_texts)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    ds = load_dataset_split(args.reference_dataset)
    log_for_0(f"Reference dataset {args.reference_dataset}: {len(ds)} rows")

    import random
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), min(n_ref, len(ds)))
    reference_texts = []
    for i in indices:
        row = ds[i]
        seq_len = row.get("sequence_length", len(row["input_ids"]))
        ids = row["input_ids"][:seq_len]
        reference_texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    log_for_0(f"Sampled {len(reference_texts)} real reference texts (seed={args.seed})")

    mauve_score = compute_mauve(
        generated_texts=generated_texts, reference_texts=reference_texts,
        featurize_model_name=args.featurize_model_name, max_text_length=args.max_text_length,
        device_id=0, verbose=True, seed=args.seed,
    )

    log_for_0("=" * 60)
    log_for_0(f"Generated sample: {args.generated_path} (n={len(generated_texts)})")
    log_for_0(f"Reference sample: {args.reference_dataset} (n={len(reference_texts)})")
    log_for_0(f"MAUVE: {mauve_score:.4f}")
    log_for_0("=" * 60)


if __name__ == "__main__":
    main()
