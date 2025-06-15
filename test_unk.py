# test_unk.py
import torch, sys, os
sys.path.insert(0, os.getcwd())
from src.tokenizer import BPETokenizer

enc, dec = torch.load("models/tokenizer.pt", map_location="cpu")
tok = BPETokenizer()
tok.encoder, tok.decoder = enc, dec
print("<unk> in vocab?", "<unk>" in tok.encoder)