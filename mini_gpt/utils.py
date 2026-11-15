"""Small helpers shared by the training, generation and experiment scripts."""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def fix_console_encoding() -> None:
    """Make ``print`` safe for any Unicode text (needed on Windows terminals).

    The default Windows console encoding cannot print every character (for
    example Telugu script). Switching stdout to UTF-8 avoids UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def set_seed(seed: int) -> None:
    """Seed every random number generator we use, so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    """Pick the device to run on.

    ``"auto"`` chooses the GPU when PyTorch can see one, otherwise the CPU.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Learning rate at a given step: linear warmup, then cosine decay.

    1. Warmup (step < warmup_steps): the rate climbs in a straight line from
       ~0 up to ``max_lr``. Early gradients are noisy because the weights are
       random, so we start with small steps.
    2. Cosine decay: the rate follows half a cosine wave from ``max_lr`` down
       to ``min_lr``. Big steps first (explore), small steps last (fine-tune).
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # progress goes from 0.0 (end of warmup) to 1.0 (end of training)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    # cos(0) = 1 and cos(pi) = -1, so this coefficient slides smoothly from 1 to 0
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coefficient * (max_lr - min_lr)


def configure_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
) -> torch.optim.AdamW:
    """Create AdamW with weight decay applied only to weight matrices.

    Weight decay gently pulls weights toward zero, which discourages the model
    from relying on a few huge weights (a form of regularisation). It is
    standard practice to decay only tensors with 2+ dimensions (the weight
    matrices of Linear layers and embeddings) and to leave 1-D tensors (biases
    and LayerNorm gains) alone.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)


class CSVLogger:
    """Append rows of metrics to a CSV file (one row per evaluation)."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict[str, Any]) -> None:
        """Write one row; keys must match the fieldnames given at construction."""
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


def read_csv_log(path: str | Path) -> dict[str, list[float]]:
    """Read a CSV written by :class:`CSVLogger` into ``{column: [values...]}``."""
    columns: dict[str, list[float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))
    return columns


def save_checkpoint(path: str | Path, **contents: Any) -> None:
    """Save a checkpoint: a dict with the model weights and everything needed to reuse them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(contents, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a checkpoint saved by :func:`save_checkpoint` onto ``device``."""
    return torch.load(path, map_location=device, weights_only=True)
