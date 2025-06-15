# src/generate.py
import argparse
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import BPETokenizer

def sample_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    Apply temperature scaling, top-k, and/or top-p (nucleus) filtering to logits,
    then return the normalized probabilities.
    """
    # Temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, top_k)
        min_topk = topk_vals[-1]
        logits = torch.where(logits < min_topk, torch.full_like(logits, float('-inf')), logits)

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)

        # Identify tokens to remove
        sorted_indices_to_remove = cum_probs > top_p
        # Shift right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0]  = False

        # Mask out those tokens
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

    # Return probabilities
    return F.softmax(logits, dim=-1)

def generate(
    prompt: str,
    model: GPT,
    tokenizer: BPETokenizer,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token: int = None
) -> str:
    """
    Generate text autoregressively given a prompt.
    """
    model.eval()
    token_ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids)        # (1, seq_len, vocab_size)
        next_logits = logits[0, -1]          # (vocab_size,)

        probs = sample_logits(next_logits, temperature, top_k, top_p)
        next_id = int(torch.multinomial(probs, num_samples=1).item())

        token_ids.append(next_id)
        if eos_token is not None and next_id == eos_token:
            break

    return tokenizer.decode(token_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Text Generation")
    parser.add_argument("prompt", type=str, help="Input prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature",      type=float, default=1.0, help="Sampling temperature (>0)")
    parser.add_argument("--top_k",           type=int,   default=50,   help="Top-k filtering (0 = disabled)")
    parser.add_argument("--top_p",           type=float, default=0.9,  help="Top-p (nucleus) filtering (0-1)")
    args = parser.parse_args()

    # 1) Load tokenizer
    enc, dec = torch.load("models/tokenizer.pt", map_location="cpu")
    tokenizer = BPETokenizer()
    tokenizer.encoder, tokenizer.decoder = enc, dec

    # 2) Instantiate model with matching architecture
    vocab_size = len(tokenizer.encoder)
    cfg = GPTConfig(
        vocab_size=vocab_size,
        n_embd=512,       # match your training --model_size
        n_layer=12,       # match your training n_layer
        n_head=8,         # match your training n_head
        block_size=1024   # match your training block_size
    )
    model = GPT(cfg)
    state_dict = torch.load("models/final_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    # 3) Determine end-of-sequence token if present
    eos_token = tokenizer.encoder.get("</w>", None)

    # 4) Generate and print
    output = generate(
        args.prompt,
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token=eos_token
    )
    print(output)