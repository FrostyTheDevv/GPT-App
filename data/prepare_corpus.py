#!/usr/bin/env python3
"""
Downloads WikiText-2 and writes out a single-line-per-document
corpus to data/corpus.txt for training the GPT model.
"""

import argparse
from datasets import load_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        default="data/corpus.txt",
        help="Path to write the combined corpus (one line per example)"
    )
    p.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Which splits of WikiText-2 to include"
    )
    args = p.parse_args()

    # Load WikiText-2
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split if args.split!="all" else ["train", "validation", "test"])
    # If ds is a list of splits, concatenate
    if isinstance(ds, list):
        from datasets import concatenate_datasets
        ds = concatenate_datasets(ds)

    print(f"[prepare_corpus] Loaded {len(ds)} lines from WikiText-2 ({args.split})")

    # Each entry has a 'text' field
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in ds:
            line = ex["text"].strip()
            # skip empty lines
            if not line:
                continue
            # replace newlines inside each article with space
            line = line.replace("\n", " ")
            f.write(line + "\n")

    print(f"[prepare_corpus] Wrote corpus to {args.output}")

if __name__ == "__main__":
    main()