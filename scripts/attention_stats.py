"""Measure, in numbers, where each attention head looks.

A heatmap of one sentence is an anecdote. This script averages over many
validation sequences and reports, for every head:

* how much weight goes to the token itself, the previous token, two back, ...
* the mean distance (in tokens) between a token and what it attends to
* the "effective number of tokens" attended to, exp(entropy): 1 means the head
  looks at a single token, 40 means its weight is spread over about 40 tokens

Usage (from the repository root):

    python scripts/attention_stats.py                  # writes results/attention_stats.md
    python scripts/attention_stats.py --head_removal   # also switch heads off and measure the loss
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`, `import generate`

import torch  # noqa: E402

from generate import load_model  # noqa: E402
from mini_gpt.config import Config  # noqa: E402
from mini_gpt.data import get_batch, load_dataset  # noqa: E402
from mini_gpt.utils import fix_console_encoding, get_device, set_seed  # noqa: E402


def head_statistics(weights: torch.Tensor, skip: int = 16) -> list[dict[str, float]]:
    """Statistics for every head of one layer.

    Args:
        weights: attention weights of shape (B, n_head, T, T).
        skip: ignore the first ``skip`` rows; early tokens have few tokens to look at.
    """
    B, n_head, T, _ = weights.shape
    positions = torch.arange(T)
    distance = (positions[:, None] - positions[None, :]).clamp(min=0).float()  # (T, T): i - j
    rows = weights[:, :, skip:, :]  # (B, n_head, T - skip, T)

    stats = []
    for h in range(n_head):
        mean_map = weights[:, h].mean(dim=0)  # (T, T), averaged over the batch
        # weight at offset k = the k-th diagonal below the main one
        at_offset = [torch.diagonal(mean_map, -k)[max(0, skip - k):].mean().item() for k in range(4)]
        mean_distance = (rows[:, h] * distance[skip:]).sum(dim=-1).mean().item()
        entropy = -(rows[:, h] * (rows[:, h] + 1e-12).log()).sum(dim=-1).mean().item()
        stats.append({
            "self": at_offset[0], "prev": at_offset[1], "two_back": at_offset[2], "three_back": at_offset[3],
            "distance": mean_distance, "effective_tokens": float(torch.tensor(entropy).exp()),
        })
    return stats


def zero_head(model: torch.nn.Module, layer: int, head: int) -> Callable[[], None]:
    """Make one attention head output zeros. Returns a function that undoes the change."""
    module = model.blocks[layer].attn.heads[head]
    original = module.forward

    def forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The head's forward pass with its output replaced by zeros."""
        out, weights = original(x)
        return torch.zeros_like(out), weights  # the head now contributes nothing

    module.forward = forward
    return lambda: setattr(module, "forward", original)


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module, val_ids: torch.Tensor, cfg: Config, device: torch.device, batches: int = 40
) -> float:
    """Average loss over the SAME validation batches every time it is called."""
    torch.manual_seed(0)
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(val_ids, cfg.model.block_size, cfg.train.batch_size, device)
        total += model(x, y)[1].item()
    return total / batches


def head_removal_table(
    model: torch.nn.Module, val_ids: torch.Tensor, cfg: Config, device: torch.device
) -> list[str]:
    """Switch off heads of layer 0, one at a time and in one pair, and measure the damage.

    Looking at where a head attends only describes it. Removing it and measuring the
    loss tests whether the model actually needs it.
    """
    intact = validation_loss(model, val_ids, cfg, device)
    lines = ["", "## Removing heads of layer 0", "",
             "Validation loss of the trained model when the output of a head is set to zero "
             "(same 40 batches each time).", "",
             "| Heads removed | Validation loss | Change |", "|---|---:|---:|",
             f"| none | {intact:.4f} | |"]
    n_head = len(model.blocks[0].attn.heads)
    for heads in [[h] for h in range(n_head)] + [[0, 4]]:
        undo = [zero_head(model, 0, h) for h in heads]
        loss = validation_loss(model, val_ids, cfg, device)
        for restore in undo:
            restore()
        lines.append(f"| {' and '.join(map(str, heads))} | {loss:.4f} | {loss - intact:+.4f} |")
    return lines


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Attention statistics per head.")
    parser.add_argument("--checkpoint", default="checkpoints/small/best.pt")
    parser.add_argument("--out", default="results/attention_stats.md")
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--head_removal", action="store_true",
                        help="also measure the loss with heads of layer 0 switched off (slow on a CPU)")
    args = parser.parse_args()
    fix_console_encoding()
    set_seed(0)

    device = get_device("auto")
    model, _, cfg = load_model(args.checkpoint, device)
    _, val_ids, _, _ = load_dataset(cfg.data)
    length = min(args.length, cfg.model.block_size)
    x, _ = get_batch(val_ids, length, args.sequences, device)
    maps = [m.float().cpu() for m in model.attention_maps(x)]  # one (B, n_head, T, T) per layer

    lines = [
        "# Attention statistics",
        "",
        f"Averaged over {args.sequences} validation sequences of {length} tokens "
        f"(`python scripts/attention_stats.py`). *self / prev / 2 back / 3 back*: average weight "
        "a token puts on itself and on the tokens 1, 2 and 3 positions before it. *Distance*: "
        "average distance in tokens to what is attended. *Eff. tokens*: exp(entropy), roughly "
        "how many tokens the weight is spread over.",
        "",
        "## Every head",
        "",
        "| Layer | Head | self | prev | 2 back | 3 back | Distance | Eff. tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = []
    for layer, weights in enumerate(maps):
        stats = head_statistics(weights)
        for h, s in enumerate(stats):
            lines.append(f"| {layer} | {h} | {s['self']:.2f} | {s['prev']:.2f} | {s['two_back']:.2f} | "
                         f"{s['three_back']:.2f} | {s['distance']:.1f} | {s['effective_tokens']:.1f} |")
        summary.append((layer, sum(s["distance"] for s in stats) / len(stats),
                        sum(s["effective_tokens"] for s in stats) / len(stats)))
    lines += ["", "## Average per layer", "", "| Layer | Mean distance | Mean eff. tokens |", "|---:|---:|---:|"]
    lines += [f"| {layer} | {distance:.1f} | {effective:.1f} |" for layer, distance, effective in summary]
    if args.head_removal:
        lines += head_removal_table(model, val_ids, cfg, device)
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[8:]))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
