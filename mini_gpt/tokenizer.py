"""Tokenizers: turning text into lists of integers (token IDs) and back.

A neural network only understands numbers, so the first step of any language
model is to chop text into pieces called *tokens* and give each piece an ID.

Two tokenizers are implemented here, both from scratch:

* :class:`CharTokenizer` - one token per character. Simple, tiny vocabulary.
* :class:`BPETokenizer`  - byte-pair encoding. Starts from single bytes and
  repeatedly glues the most frequent neighbouring pair into a new token, so
  common chunks such as ``"the"`` or ``" and"`` become single tokens.

Both expose the same interface: ``encode``, ``decode``, ``vocab_size``,
``token_to_str``, ``save`` and ``load``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Character-level tokenizer
# --------------------------------------------------------------------------- #


class CharTokenizer:
    """One token per unique character in the training text."""

    def __init__(self, chars: list[str]) -> None:
        """
        Args:
            chars: the vocabulary - every distinct character, in a fixed order.
                   The position of a character in this list is its token ID.
        """
        self.chars = chars
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}  # string -> integer
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}  # integer -> string

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        """Build the vocabulary: the sorted set of characters that appear in ``text``."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens."""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Text -> list of token IDs. Raises if a character was never seen in training."""
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"Characters not in the vocabulary: {unknown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """List of token IDs -> text."""
        return "".join(self.itos[i] for i in ids)

    def token_to_str(self, token_id: int) -> str:
        """Human-readable form of a single token (used to label plots)."""
        return self.itos[token_id]

    def to_dict(self) -> dict[str, Any]:
        """Everything needed to rebuild this tokenizer, as JSON-friendly data."""
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "CharTokenizer":
        return cls(list(state["chars"]))

    def save(self, path: str | Path) -> None:
        """Write the vocabulary to a JSON file."""
        _save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Read a vocabulary written by :meth:`save`."""
        return cls.from_dict(_load_json(path))


def _save_json(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
