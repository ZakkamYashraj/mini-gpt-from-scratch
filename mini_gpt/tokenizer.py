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
import re
from collections import Counter
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


# --------------------------------------------------------------------------- #
# Byte-pair encoding (BPE) tokenizer
# --------------------------------------------------------------------------- #

# Split text into "chunks": a word together with the single whitespace character
# in front of it (e.g. " king"), or a run of whitespace. Merges never cross a
# chunk boundary, which keeps tokens word-like and makes training fast, because
# we only need to process each *distinct* chunk once.
_CHUNK_PATTERN = re.compile(r"\s?\S+|\s+")


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with the single token ``new_id``.

    Example: ``_merge_pair([1, 2, 3, 1, 2], (1, 2), 9) -> [9, 3, 9]``
    """
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2  # skip both members of the pair
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BPETokenizer:
    """A small, readable byte-level BPE tokenizer.

    * The base vocabulary is the 256 possible byte values, so *any* text in
      *any* language (it is all UTF-8 bytes underneath) can be encoded.
    * Training learns an ordered list of merges ``(a, b) -> new_id``.
    * Encoding replays those merges in the order they were learned.
    """

    def __init__(self, merges: dict[tuple[int, int], int]) -> None:
        """
        Args:
            merges: maps a pair of token IDs to the ID of the token they merge
                    into. Insertion order is the order the merges were learned.
        """
        self.merges = merges
        # vocab maps token ID -> the raw bytes that token stands for.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}  # chunk -> token IDs (speeds up encoding)

    # ------------------------------ training ------------------------------ #

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Learn ``vocab_size - 256`` merges from ``text``.

        The algorithm, repeated until the vocabulary is big enough:
            1. count how often every adjacent pair of tokens occurs
            2. pick the most frequent pair
            3. give it a new token ID and replace the pair everywhere
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the number of byte values)")

        # Count each distinct chunk once: {"the": 3000, " king": 120, ...}
        chunk_counts = Counter(_CHUNK_PATTERN.findall(text))
        # Represent every chunk as a list of byte values (ints 0..255).
        words: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        counts: list[int] = list(chunk_counts.values())

        merges: dict[tuple[int, int], int] = {}
        for new_id in range(256, vocab_size):
            # 1. count adjacent pairs, weighted by how often the chunk occurs
            pair_counts: Counter[tuple[int, int]] = Counter()
            for ids, count in zip(words, counts):
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                break  # every chunk is already a single token: nothing left to merge

            # 2. the most frequent pair (ties broken by smallest IDs, so training is deterministic)
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))

            # 3. record the merge and apply it to every chunk
            merges[best] = new_id
            words = [_merge_pair(ids, best, new_id) for ids in words]

            if verbose and (new_id - 255) % 50 == 0:
                print(f"  BPE merge {new_id - 255}/{vocab_size - 256}: {best} -> {new_id}")

        return cls(merges)

    # --------------------------- encode / decode --------------------------- #

    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens (256 bytes + number of merges)."""
        return len(self.vocab)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one chunk by replaying the learned merges, earliest-learned first."""
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Among the pairs present, find the one that was learned earliest.
            # Pairs that were never learned get rank "infinity".
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no more merges apply
            ids = _merge_pair(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        """Text -> list of token IDs."""
        ids: list[int] = []
        for chunk in _CHUNK_PATTERN.findall(text):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(chunk)
            ids.extend(self._cache[chunk])
        return ids

    def decode(self, ids: list[int]) -> str:
        """List of token IDs -> text.

        A model can generate a sequence that stops halfway through a multi-byte
        character; ``errors="replace"`` turns such broken bytes into the
        replacement character instead of crashing.
        """
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def token_to_str(self, token_id: int) -> str:
        """Human-readable form of a single token (used to label plots)."""
        return self.vocab[token_id].decode("utf-8", errors="replace")

    # ------------------------------ save / load ----------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Everything needed to rebuild this tokenizer, as JSON-friendly data."""
        return {"type": "bpe", "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()]}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "BPETokenizer":
        return cls({(a, b): new_id for a, b, new_id in state["merges"]})

    def save(self, path: str | Path) -> None:
        """Write the merge list to a JSON file."""
        _save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Read a merge list written by :meth:`save`."""
        return cls.from_dict(_load_json(path))


# --------------------------------------------------------------------------- #
# Helpers shared by both tokenizers
# --------------------------------------------------------------------------- #

Tokenizer = CharTokenizer | BPETokenizer


def train_tokenizer(kind: str, text: str, bpe_vocab_size: int = 512) -> Tokenizer:
    """Train the tokenizer named by ``kind`` (``"char"`` or ``"bpe"``) on ``text``."""
    if kind == "char":
        return CharTokenizer.train(text)
    if kind == "bpe":
        return BPETokenizer.train(text, bpe_vocab_size)
    raise ValueError(f"Unknown tokenizer {kind!r}; expected 'char' or 'bpe'")


def tokenizer_from_dict(state: dict[str, Any]) -> Tokenizer:
    """Rebuild a tokenizer from the dict produced by its ``to_dict`` method."""
    if state["type"] == "char":
        return CharTokenizer.from_dict(state)
    if state["type"] == "bpe":
        return BPETokenizer.from_dict(state)
    raise ValueError(f"Unknown tokenizer type {state['type']!r}")


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load whichever tokenizer was saved at ``path``."""
    return tokenizer_from_dict(_load_json(path))


def _save_json(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
