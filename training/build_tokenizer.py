#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
import torch
from src.tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Build a BPE tokenizer from a text corpus"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to your newline-separated text corpus"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30000,
        help="Desired number of BPE tokens"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to save the tokenizer (e.g. models/tokenizer.pt)"
    )
    args = parser.parse_args()

    # 1) Load corpus
    print(f"[build_tokenizer] Loading corpus from {args.input}")
    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)

    print(f"[build_tokenizer] {len(texts)} lines loaded. Building BPE vocab...")
    tok = BPETokenizer(vocab_size=args.vocab_size)
    tok.get_vocab(texts)

    print(f"[build_tokenizer] Vocab size={len(tok.encoder)}. Saving to {args.output}")
    # Save just the encoder+decoder dicts
    torch.save((tok.encoder, tok.decoder), args.output)
    print("[build_tokenizer] Done.")

if __name__ == "__main__":
    main()