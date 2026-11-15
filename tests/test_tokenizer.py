"""Tests for the character-level and BPE tokenizers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_gpt.tokenizer import BPETokenizer, CharTokenizer, _merge_pair, load_tokenizer

TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n"
) * 20


# ------------------------------ character level ------------------------------ #


def test_char_round_trip() -> None:
    tok = CharTokenizer.train(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_char_vocab_is_the_set_of_characters() -> None:
    tok = CharTokenizer.train("hello")
    assert tok.chars == ["e", "h", "l", "o"]
    assert tok.vocab_size == 4
    assert tok.encode("hello") == [1, 0, 2, 2, 3]


def test_char_unknown_character_raises() -> None:
    tok = CharTokenizer.train("abc")
    with pytest.raises(ValueError):
        tok.encode("abcd")


def test_char_save_load(tmp_path: Path) -> None:
    tok = CharTokenizer.train(TEXT)
    tok.save(tmp_path / "tok.json")
    loaded = load_tokenizer(tmp_path / "tok.json")
    assert isinstance(loaded, CharTokenizer)
    assert loaded.encode(TEXT) == tok.encode(TEXT)


# ------------------------------------ BPE ------------------------------------ #


def test_merge_pair() -> None:
    assert _merge_pair([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert _merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]  # pairs do not overlap
    assert _merge_pair([5], (1, 2), 9) == [5]


def test_bpe_round_trip() -> None:
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_bpe_round_trip_on_unseen_and_non_english_text() -> None:
    """Byte-level BPE can encode ANY string, even characters it never saw in training."""
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    for text in ["", "zebra quiz!", "naïve café", "తెలుగు భాష", "tabs\tand\n\nnewlines  "]:
        assert tok.decode(tok.encode(text)) == text


def test_bpe_compresses_text() -> None:
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    assert tok.vocab_size == 300
    n_bytes = len(TEXT.encode("utf-8"))
    assert len(tok.encode(TEXT)) < n_bytes / 2  # merges make sequences much shorter


def test_bpe_first_merge_is_the_most_frequent_pair() -> None:
    # In "aaab aaab aaab" the pair ("a", "a") is the most common adjacent pair.
    tok = BPETokenizer.train("aaab aaab aaab", vocab_size=257)
    assert list(tok.merges) == [(ord("a"), ord("a"))]
    assert tok.vocab[256] == b"aa"


def test_bpe_training_is_deterministic() -> None:
    a = BPETokenizer.train(TEXT, vocab_size=280)
    b = BPETokenizer.train(TEXT, vocab_size=280)
    assert a.merges == b.merges


def test_bpe_save_load(tmp_path: Path) -> None:
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    tok.save(tmp_path / "bpe.json")
    loaded = load_tokenizer(tmp_path / "bpe.json")
    assert isinstance(loaded, BPETokenizer)
    assert loaded.merges == tok.merges
    assert loaded.encode(TEXT) == tok.encode(TEXT)
