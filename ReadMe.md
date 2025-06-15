# 🧠 my_local_gpt

A from-scratch GPT-style language model implementation in Python.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Overview

This project provides:

- ✅ Byte-Pair Encoding (BPE) tokenizer  
- ✅ GPT-style Transformer decoder in PyTorch  
- ✅ Training pipeline with:
  - Mixed-precision (AMP)
  - Multi-GPU support
  - Gradient accumulation
  - One-Cycle LR schedule
  - TensorBoard logging
  - Checkpointing
- ✅ CLI inference script (`generate.py`) for text generation  
- ✅ Guidelines for FastAPI or Discord bot integration  

---

## 📁 Project Layout
my_local_gpt/
├── data/
│ └── corpus.txt # Raw text data, one line per example
├── models/
│ ├── tokenizer.pt # Saved BPE tokenizer (encoder & decoder)
│ ├── tokens.pt # Cached token IDs (optional)
│ ├── ckpt_step_*.pt # Training checkpoints
│ └── final_model.pt # Final model weights
├── src/
│ ├── tokenizer.py # BPETokenizer implementation
│ ├── model.py # GPT model definition
│ └── generate.py # CLI for text generation
├── training/
│ ├── build_tokenizer.py # Script to build BPE tokenizer
│ └── train.py # Full-scale training pipeline
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
├── LICENSE # Open-source license
├── TOS.md # Terms of Service
└── README.md # This file

---

## 🧪 Installation

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/my_local_gpt.git
cd my_local_gpt
```

🧱 Building the Tokenizer
```bash
python training/build_tokenizer.py \
  --input data/corpus.txt \
  --vocab_size 30000 \
  --output models/tokenizer.pt
  ```

🏋️‍♀️ Training the Model
  ```bash
  python training/train.py \
  --corpus data/corpus.txt \
  --tokenizer models/tokenizer.pt \
  --vocab_size 30000 \
  --model_size 512 \
  --block_size 1024 \
  --batch_size 8 \
  --accum_steps 4 \
  --lr 5e-4 \
  --epochs 5 \
  --save_steps 1000 \
  --out_dir models
   ```

✨ Generating Text
```bash
python src/generate.py "Your prompt here"
 ```

