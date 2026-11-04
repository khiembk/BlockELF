#!/usr/bin/env python
"""Tokenize the One Billion Word Benchmark (LM1B) for unconditional ELF/BlockELF training.

LM1B examples are already short, single sentences (unlike OpenWebText's long documents),
so — matching how this corpus is treated elsewhere in the diffusion-LM literature
(e.g. SEDD/MDLM use `wrap=False` for lm1b but `wrap=True` for owt/wikitext) — each
sentence becomes its own example: T5-tokenized with an explicit EOS appended
(`add_special_tokens=True`), truncated to `--max_length`, no packing across sentences.

This mirrors the schema of this repo's other pretokenized datasets (see data_utils.py /
README "Prepare your own data"): each saved example has
  - input_ids: T5 token ids (unconditional — no condition_input_ids column)
  - sequence_length: token count after truncation (used by
    utils.data_utils.compute_block_round_counts for block-revelation round reweighting)

Usage:
    cd src && python scripts/prepare_lm1b.py \
        --output_dir /scratch/<user>/datasets/lm1b_t5 \
        --max_length 128
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import load_dataset
from transformers import AutoTokenizer

# Tried in order; the legacy `lm1b` HF script dataset has been superseded by
# Parquet mirrors that modern `datasets` versions (dropped loading-script support)
# can read directly with no `trust_remote_code`.
LM1B_HF_CANDIDATES = ["dvruette/lm1b", "billion-word-benchmark/lm1b", "lm1b"]
SPLIT_NAMES = ("train", "test")


def parse_args():
    p = argparse.ArgumentParser(description="Tokenize LM1B with a T5 tokenizer for BlockELF.")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Root dir to write; splits are saved to <output_dir>/{train,test}.")
    p.add_argument("--tokenizer_name", type=str, default="t5-small")
    p.add_argument("--max_length", type=int, default=128,
                    help="Per-sentence token cap (truncation), including the appended EOS.")
    p.add_argument("--splits", type=str, default="train,test",
                    help="Comma-separated subset of {train, test} to process.")
    p.add_argument("--hf_dataset_name", type=str, default=None,
                    help="Override the HF dataset id instead of trying LM1B_HF_CANDIDATES in order.")
    p.add_argument("--cache_dir", type=str, default=None, help="HF datasets cache dir.")
    p.add_argument("--num_proc", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--batch_size", type=int, default=1000, help="Map() batch size for tokenization.")
    p.add_argument("--limit", type=int, default=None,
                    help="Optional cap on examples per split, for a fast smoke test.")
    return p.parse_args()


def load_lm1b(hf_dataset_name, cache_dir):
    candidates = [hf_dataset_name] if hf_dataset_name else LM1B_HF_CANDIDATES
    last_err = None
    for name in candidates:
        try:
            ds = load_dataset(name, cache_dir=cache_dir)
            print(f"Loaded LM1B from '{name}': {ds}")
            return ds
        except Exception as e:  # noqa: BLE001 - want to try the next candidate
            print(f"Could not load '{name}' ({e.__class__.__name__}: {e}); trying next candidate...")
            last_err = e
    raise RuntimeError(f"Failed to load LM1B from any of {candidates}") from last_err


def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in SPLIT_NAMES:
            raise ValueError(f"Unknown split '{s}', expected one of {SPLIT_NAMES}")

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    raw = load_lm1b(args.hf_dataset_name, args.cache_dir)

    def tokenize_batch(examples):
        enc = tokenizer(
            examples["text"],
            add_special_tokens=True,  # appends EOS, matching the pad_token: eos convention
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = enc["input_ids"]
        return {
            "input_ids": input_ids,
            "sequence_length": [len(ids) for ids in input_ids],
        }

    os.makedirs(args.output_dir, exist_ok=True)
    for split in splits:
        ds = raw[split]
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))
        print(f"[{split}] tokenizing {len(ds)} examples...")

        ds = ds.map(
            tokenize_batch,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=ds.column_names,
            desc=f"Tokenizing LM1B/{split}",
        )
        before = len(ds)
        ds = ds.filter(
            lambda seq_len: seq_len > 0,
            input_columns=["sequence_length"],
            num_proc=args.num_proc,
            desc=f"Dropping empty LM1B/{split} examples",
        )
        dropped = before - len(ds)
        if dropped:
            print(f"[{split}] dropped {dropped} empty example(s)")

        out_path = os.path.join(args.output_dir, split)
        ds.save_to_disk(out_path)
        print(f"[{split}] saved {len(ds)} examples to {out_path}")
        # Stats from a prefix sample only: a full-column pull (ds[col]) is a row-by-row
        # Python iteration and takes ~2 minutes at 30M+ rows for what's just a log line.
        sample_n = min(len(ds), 200_000)
        seq_lengths = np.asarray(ds[:sample_n]["sequence_length"])
        print(
            f"[{split}] sequence_length over first {sample_n} examples: "
            f"mean={seq_lengths.mean():.2f}, max={int(seq_lengths.max())}, min={int(seq_lengths.min())}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
