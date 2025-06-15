# training/train.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))


import os
import time
import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from src.tokenizer import BPETokenizer
from src.model     import GPT, GPTConfig

class TextDataset(Dataset):
    """Converts a flat list of token IDs into (input, target) sequence pairs."""
    def __init__(self, token_ids, block_size):
        self.ids = token_ids
        self.bs  = block_size

    def __len__(self):
        return len(self.ids) - self.bs

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+self.bs], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+1+self.bs], dtype=torch.long)
        return x, y

def load_or_build_tokens(corpus_path: str, tokenizer: BPETokenizer, cache_path: str):
    """
    Encode the corpus into a flat list of token IDs.
    Caches to disk so you don’t re-tokenize every run.
    """
    cache = Path(cache_path)
    if cache.exists():
        print(f"[train] Loading token IDs from cache: {cache}")
        return torch.load(cache)

    print(f"[train] Tokenizing corpus: {corpus_path}")
    ids = []
    eos_id = tokenizer.encoder.get('</w>', None)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.extend(tokenizer.encode(line))
            if eos_id is not None:
                ids.append(eos_id)
    print(f"[train] {len(ids)} tokens encoded; caching to {cache}")
    torch.save(ids, cache)
    return ids

def save_checkpoint(model, optimizer, scheduler, scaler, step, ckpt_dir):
    ckpt = {
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict() if scheduler else None,
        'scaler':     scaler.state_dict(),
        'step':       step
    }
    path = os.path.join(ckpt_dir, f'ckpt_step_{step}.pt')
    torch.save(ckpt, path)
    print(f"[train] Saved checkpoint: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',     required=True,  help="Raw text corpus (one line per document)")
    parser.add_argument('--tokenizer',  required=True,  help="Path to tokenizer.pt from build_tokenizer")
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--model_size', type=int, default=256, help="Embedding dimension (n_embd)")
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accum_steps',type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--epochs',     type=int, default=3)
    parser.add_argument('--warmup_steps',type=int,default=1000)
    parser.add_argument('--max_steps',  type=int, default=None, help="Stop after this many updates")
    parser.add_argument('--save_steps', type=int, default=500, help="Checkpoint every N steps")
    parser.add_argument('--out_dir',    default='models', help="Where to save model and logs")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'runs'))

    # 1) Load tokenizer & corpus tokens
    enc, dec = torch.load(args.tokenizer, map_location='cpu')
    tokenizer = BPETokenizer()
    tokenizer.encoder, tokenizer.decoder = enc, dec
    tokens_cache = os.path.join(args.out_dir, 'tokens.pt')
    token_ids = load_or_build_tokens(args.corpus, tokenizer, tokens_cache)

    # 2) Prepare dataset & dataloader
    dataset = TextDataset(token_ids, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 3) Build model
    cfg = GPTConfig(
        vocab_size = args.vocab_size,
        n_embd     = args.model_size,
        n_layer    = 12,
        n_head     = max(1, args.model_size // 64),
        block_size = args.block_size
    )
    model = GPT(cfg).to(device)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[train] Using {torch.cuda.device_count()} GPUs")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=1e-1)
    total_updates = ((len(dataloader) * args.epochs) // args.accum_steps) + args.warmup_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_updates,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    scaler = GradScaler()

    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast():
                logits = model(xb)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    yb.view(-1)
                ) / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if step % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"[train] E{epoch} S{step:06d} loss={loss.item()*args.accum_steps:.4f} lr={lr:.2e}")
                writer.add_scalar('loss/train', loss.item()*args.accum_steps, step)
                writer.add_scalar('lr', lr, step)

            if args.save_steps and step and step % args.save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, step, args.out_dir)

            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    save_checkpoint(model, optimizer, scheduler, scaler, step, args.out_dir)
    writer.close()
    print("[train] Training complete.")

if __name__ == "__main__":
    main()