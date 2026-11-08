"""Lightweight deterministic Tokenizer.

Provides deterministic string encoding and decoding for auto-regressive
generation without requiring multi-gigabyte external model weights.
"""

import re
from typing import Dict, List, Optional


class SimpleTokenizer:
    """Fast, deterministic word-and-character tokenizer for lightweight inference."""

    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    UNK_TOKEN = "<|unk|>"

    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    UNK_TOKEN_ID = 3

    def __init__(self, vocab_size: int = 1000) -> None:
        self.vocab_size: int = max(vocab_size, 256)
        self.token_to_id: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_TOKEN_ID,
            self.BOS_TOKEN: self.BOS_TOKEN_ID,
            self.EOS_TOKEN: self.EOS_TOKEN_ID,
            self.UNK_TOKEN: self.UNK_TOKEN_ID,
        }
        self.id_to_token: Dict[int, str] = {
            v: k for k, v in self.token_to_id.items()
        }

        # Seed vocabulary with common LLM keywords, system terms, and punctuation
        seed_words = [
            " ", "\n", ".", ",", "!", "?", ":", ";", "-", "_", "(", ")", "[", "]", "{", "}",
            "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was", "for",
            "on", "are", "as", "with", "his", "they", "I", "at", "be", "this", "have", "from",
            "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when",
            "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how",
            "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these",
            "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look",
            "two", "more", "write", "go", "see", "number", "no", "way", "could", "people", "my",
            "than", "first", "water", "been", "call", "who", "oil", "its", "now", "find", "long",
            "down", "day", "did", "get", "come", "made", "may", "part", "LLM", "inference",
            "batch", "serving", "engine", "paged", "cache", "block", "memory", "latency", "throughput",
            "continuous", "token", "scheduler", "attention", "request", "stream", "FastAPI",
            "allocation", "physical", "logical", "table", "preemption", "zero", "fragmentation",
            "dynamic", "performance", "system", "parallel", "generation", "distributed", "model",
        ]

        curr_id = 4
        for word in seed_words:
            if word not in self.token_to_id and curr_id < self.vocab_size:
                self.token_to_id[word] = curr_id
                self.id_to_token[curr_id] = word
                curr_id += 1

        # Fill remaining slots with character/byte tokens to ensure zero OOV
        for code in range(32, 127):
            char = chr(code)
            if char not in self.token_to_id and curr_id < self.vocab_size:
                self.token_to_id[char] = curr_id
                self.id_to_token[curr_id] = char
                curr_id += 1

        # Fallback tokens
        while curr_id < self.vocab_size:
            placeholder = f"<|tok_{curr_id}|>"
            self.token_to_id[placeholder] = curr_id
            self.id_to_token[curr_id] = placeholder
            curr_id += 1

    def encode(self, text: str) -> List[int]:
        """Encodes an input string into a list of token IDs."""
        if not text:
            return []

        # Tokenize by words, punctuation, whitespace chunks
        tokens: List[int] = []
        words = re.findall(r"\w+|[^\w\s]|\s+", text, re.UNICODE)

        for w in words:
            if w in self.token_to_id:
                tokens.append(self.token_to_id[w])
            else:
                # Fallback to character decomposition
                for ch in w:
                    tokens.append(self.token_to_id.get(ch, self.UNK_TOKEN_ID))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back into a human-readable text string."""
        pieces: List[str] = []
        for tid in token_ids:
            if tid == self.EOS_TOKEN_ID:
                break
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if not tok.startswith("<|"):
                    pieces.append(tok)
            else:
                pieces.append("")
        return "".join(pieces)

    def decode_single(self, token_id: int) -> str:
        """Decodes a single token ID into its text representation."""
        if token_id == self.EOS_TOKEN_ID:
            return ""
        tok = self.id_to_token.get(token_id, "")
        return "" if tok.startswith("<|") else tok


# Default tokenizer instance
tokenizer = SimpleTokenizer()

