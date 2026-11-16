"""Learning Guide, Chapter 14: temperature, top-k and top-p on a toy distribution.

Run it from the repository root:  python examples/ch14_sampling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mini_gpt.model import sample_next_token, top_k_filter, top_p_filter  # noqa: E402

torch.manual_seed(0)
tokens = ["e", "a", "o", "i", "u", "z"]
logits = torch.tensor([[3.0, 2.5, 2.0, 1.0, 0.0, -2.0]])  # the model's scores for 6 tokens


def show(label: str, probs: torch.Tensor) -> None:
    """Print one row of probabilities next to its label."""
    cells = "  ".join(f"{t}:{p:.3f}" for t, p in zip(tokens, probs[0].tolist()))
    print(f"{label:<22}{cells}")


# --- 1. Temperature ---
print("1. Temperature: divide the logits by T before the softmax")
for T in [0.5, 1.0, 2.0]:
    show(f"T = {T}", F.softmax(logits / T, dim=-1))

# --- 2. Top-k ---
print("\n2. Top-k: keep only the k best tokens")
show("top_k = 3", F.softmax(top_k_filter(logits, 3), dim=-1))

# --- 3. Top-p ---
print("\n3. Top-p: keep the smallest set of tokens whose probabilities reach p")
show("top_p = 0.9", F.softmax(top_p_filter(logits, 0.9), dim=-1))
show("top_p = 0.5", F.softmax(top_p_filter(logits, 0.5), dim=-1))

# --- 4. Sampling many times ---
print("\n4. Draw 1000 samples and count them")
for label, kwargs in [
    ("T = 1.0", dict(temperature=1.0)),
    ("T = 0.5", dict(temperature=0.5)),
    ("T = 1.0, top_k = 3", dict(temperature=1.0, top_k=3)),
    ("T = 0 (greedy)", dict(temperature=0.0)),
]:
    draws = sample_next_token(logits.repeat(1000, 1), **kwargs).flatten()
    counts = torch.bincount(draws, minlength=len(tokens))
    cells = "  ".join(f"{t}:{c:4d}" for t, c in zip(tokens, counts.tolist()))
    print(f"{label:<22}{cells}")
