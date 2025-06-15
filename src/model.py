# src/model.py
import torch
import torch.nn as nn
import math

class GPTConfig:
    def __init__(self, vocab_size, n_embd=256, n_layer=6, n_head=4, block_size=512):
        self.vocab_size = vocab_size
        self.n_embd     = n_embd
        self.n_layer    = n_layer
        self.n_head     = n_head
        self.block_size = block_size

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb   = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop      = nn.Dropout(0.1)
        self.blocks    = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f      = nn.LayerNorm(config.n_embd)
        self.head      = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, "Sequence length exceeds block size"
        tok_emb = self.token_emb(idx)               # (B, T, n_embd)
        pos_emb = self.pos_emb[:, :T, :]            # (1, T, n_embd)
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                       # (B, T, vocab_size)
        return logits