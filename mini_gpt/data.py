"""Data pipeline: raw text file -> token-ID arrays -> random training batches.

Steps performed by :func:`prepare_dataset`:

1. read the text file (downloading Tiny Shakespeare if it is the default and missing)
2. train a tokenizer on it (character-level or BPE)
3. encode the whole text into one long array of token IDs
4. split it 90% / 10% into train and validation arrays
5. save everything under ``data/processed/<dataset>_<tokenizer>/``
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from mini_gpt.config import DataConfig
from mini_gpt.tokenizer import Tokenizer, load_tokenizer, train_tokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DEFAULT_INPUT = "data/tinyshakespeare.txt"
PROCESSED_ROOT = Path("data/processed")


def download_tiny_shakespeare(path: str | Path = DEFAULT_INPUT) -> Path:
    """Download the ~1 MB Tiny Shakespeare text file (if it is not already there)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path


def processed_dir(cfg: DataConfig) -> Path:
    """Folder holding the tokenized arrays for this (text file, tokenizer) combination."""
    tag = "char" if cfg.tokenizer == "char" else f"bpe{cfg.bpe_vocab_size}"
    return PROCESSED_ROOT / f"{Path(cfg.input_file).stem}_{tag}"


def _read_text(cfg: DataConfig) -> str:
    """Read the input text, fetching Tiny Shakespeare on first use."""
    path = Path(cfg.input_file)
    if not path.exists():
        if path.as_posix() == DEFAULT_INPUT:
            download_tiny_shakespeare(path)
        else:
            raise FileNotFoundError(f"Input text file not found: {path}")
    return path.read_text(encoding="utf-8")


def prepare_dataset(cfg: DataConfig, force: bool = False) -> Path:
    """Tokenize the text file and save train/val token-ID arrays.

    The work is skipped when the arrays already exist *and* were made from the
    same text (checked with a hash), unless ``force`` is True.

    Returns:
        The folder containing ``train.npy``, ``val.npy``, ``tokenizer.json``, ``meta.json``.
    """
    out_dir = processed_dir(cfg)
    text = _read_text(cfg)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("text_hash") == text_hash and meta.get("val_fraction") == cfg.val_fraction:
            return out_dir  # already prepared from this exact text

    print(f"Preparing dataset from {cfg.input_file} with the {cfg.tokenizer!r} tokenizer ...")
    tokenizer = train_tokenizer(cfg.tokenizer, text, cfg.bpe_vocab_size)
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)  # uint16 holds IDs up to 65535

    # The split is by position, not random: the model is validated on a part of
    # the text it has never seen, which is what "generalisation" means.
    n_train = int(len(ids) * (1.0 - cfg.val_fraction))
    train_ids, val_ids = ids[:n_train], ids[n_train:]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", train_ids)
    np.save(out_dir / "val.npy", val_ids)
    tokenizer.save(out_dir / "tokenizer.json")
    meta = {
        "input_file": cfg.input_file,
        "tokenizer": cfg.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "n_chars": len(text),
        "n_tokens": int(len(ids)),
        "n_train_tokens": int(len(train_ids)),
        "n_val_tokens": int(len(val_ids)),
        "chars_per_token": len(text) / len(ids),
        "val_fraction": cfg.val_fraction,
        "text_hash": text_hash,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"  {meta['n_chars']:,} characters -> {meta['n_tokens']:,} tokens "
        f"(vocab size {meta['vocab_size']}), train {len(train_ids):,} / val {len(val_ids):,}"
    )
    return out_dir


def load_dataset(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, Tokenizer, dict]:
    """Prepare (if needed) and load the dataset.

    Returns:
        ``(train_ids, val_ids, tokenizer, meta)`` where the IDs are 1-D int64 tensors.
    """
    folder = prepare_dataset(cfg)
    # Embedding layers expect int64 ("long") indices, so convert from uint16.
    train_ids = torch.from_numpy(np.load(folder / "train.npy").astype(np.int64))
    val_ids = torch.from_numpy(np.load(folder / "val.npy").astype(np.int64))
    tokenizer = load_tokenizer(folder / "tokenizer.json")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return train_ids, val_ids, tokenizer, meta


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences.

    For each sequence we pick a random starting point ``s`` in the data:

        x = data[s     : s + block_size]      the tokens the model sees
        y = data[s + 1 : s + block_size + 1]  the same tokens shifted left by one

    so ``y[t]`` is the token that comes right after ``x[t]``: the "correct
    answer" for the prediction made at position ``t``.

    Returns:
        ``x`` and ``y``, both of shape (batch_size, block_size), on ``device``.
    """
    if len(data) <= block_size:
        raise ValueError(
            f"Dataset split has only {len(data)} tokens but block_size is {block_size}; "
            "use a longer text or a smaller block_size."
        )
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)
