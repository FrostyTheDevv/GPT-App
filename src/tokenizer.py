# src/tokenizer.py
import re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.bpe_ranks = {}
        self.encoder = {}
        self.decoder = {}
        self.unk_token = "<unk>"

    def get_vocab(self, texts):
        # 1) Build initial character-level vocab
        tokens = [tuple(text) + ('</w>',) for text in texts]
        vocab = Counter(tokens)

        # 2) Learn merges
        for _ in range(self.vocab_size - len(self.encoder)):
            pairs = Counter()
            for word, freq in vocab.items():
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.bpe_ranks[best] = len(self.bpe_ranks)
            new_vocab = {}
            pattern = re.escape(' '.join(best))
            for word, freq in vocab.items():
                w = ' '.join(word)
                w_new = re.sub(pattern, ''.join(best), w)
                new_vocab[tuple(w_new.split(' '))] = freq
            vocab = new_vocab

        # 3) Build encoder/decoder
        idx = 0
        for word in vocab:
            token = ''.join(word).replace('</w>', '')
            if token not in self.encoder:
                self.encoder[token] = idx
                self.decoder[idx] = token
                idx += 1

        # 4) Add unknown token
        if self.unk_token not in self.encoder:
            self.encoder[self.unk_token] = idx
            self.decoder[idx] = self.unk_token

    def encode(self, text):
        # naive longest-match BPE with unk fallback
        tokens = []
        i = 0
        unk_id = self.encoder[self.unk_token]
        while i < len(text):
            match = None
            # try the longest possible substring
            for length in range(len(text) - i, 0, -1):
                sub = text[i:i + length]
                if sub in self.encoder:
                    match = sub
                    break
            if match is None:
                # no match → treat as unknown single char
                tokens.append(unk_id)
                i += 1
            else:
                tokens.append(self.encoder[match])
                i += len(match)
        return tokens

    def decode(self, ids):
        # turn token IDs back into text
        pieces = []
        for idx in ids:
            pieces.append(self.decoder.get(idx, self.unk_token))
        return ''.join(pieces)