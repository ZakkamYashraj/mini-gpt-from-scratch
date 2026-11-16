"""Plot attention heatmaps of a trained model for a sample sentence.

Usage (from the repository root):

    python scripts/visualize_attention.py
    python scripts/visualize_attention.py --text "ROMEO:\nO, speak again!" --layers 0 3 5

For every requested layer this saves ``results/attention_layer<L>.png``: one
heatmap per head. Row i of a heatmap shows where token i "looks": cell (i, j)
is the weight token i puts on token j. Everything above the diagonal is zero
because of the causal mask. It also saves ``results/attention_closeup.png``,
a single large head with every token labelled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`, `import generate`

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from plot_style import MUTED, SEQUENTIAL, apply_style  # noqa: E402

from generate import load_model  # noqa: E402
from mini_gpt.utils import fix_console_encoding, get_device  # noqa: E402

DEFAULT_TEXT = "First Citizen:\nBefore we proceed, hear me speak."


def printable(token: str) -> str:
    """Make whitespace tokens visible in axis labels."""
    return token.replace("\n", "\\n").replace(" ", "␣")  # newline -> \n, space -> open-box


def plot_layer(weights: torch.Tensor, labels: list[str], layer: int, path: Path) -> None:
    """One small heatmap per head. ``weights`` has shape (n_head, T, T)."""
    n_head = weights.shape[0]
    cols = min(n_head, 3)
    rows = (n_head + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for h, ax in enumerate(axes.flat):
        if h >= n_head:
            ax.axis("off")
            continue
        ax.imshow(weights[h], cmap=SEQUENTIAL, vmin=0.0, vmax=1.0)
        ax.set_title(f"layer {layer}, head {h}", fontsize=10)
        ax.set_xticks(range(len(labels)), labels, fontsize=5)
        ax.set_yticks(range(len(labels)), labels, fontsize=5)
        ax.tick_params(length=0)
        ax.grid(False)
    fig.supxlabel("token being attended TO (key)", color=MUTED, fontsize=10)
    fig.supylabel("token doing the attending (query)", color=MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_closeup(weights: torch.Tensor, labels: list[str], title: str, path: Path) -> None:
    """One large heatmap with a colour bar. ``weights`` has shape (T, T)."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(weights, cmap=SEQUENTIAL, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.set_xlabel("token being attended TO (key)")
    ax.set_ylabel("token doing the attending (query)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="attention weight")
    fig.savefig(path)
    plt.close(fig)


def most_focused_head(maps: list[torch.Tensor]) -> tuple[int, int]:
    """Find the (layer, head) whose rows are most concentrated on tokens other than themselves.

    This picks a visually interesting head for the close-up: one that clearly
    "points" at specific earlier tokens rather than spreading its attention evenly.
    """
    best, best_score = (0, 0), -1.0
    for layer, weights in enumerate(maps):
        w = weights[0]  # (n_head, T, T)
        off_diagonal = w - torch.diag_embed(torch.diagonal(w, dim1=-2, dim2=-1))
        score = off_diagonal.max(dim=-1).values[:, 2:].mean(dim=-1)  # skip the first rows (few choices)
        head = int(score.argmax())
        if score[head] > best_score:
            best, best_score = (layer, head), float(score[head])
    return best


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Plot attention heatmaps.")
    parser.add_argument("--checkpoint", default="checkpoints/small/best.pt")
    parser.add_argument("--text", default=DEFAULT_TEXT, help=r"sentence to visualise (use \n for newline)")
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="default: first and last")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    fix_console_encoding()
    apply_style()

    device = get_device("auto")
    model, tokenizer, cfg = load_model(args.checkpoint, device)

    text = args.text.replace("\\n", "\n")
    ids = tokenizer.encode(text)[: cfg.model.block_size]
    labels = [printable(tokenizer.token_to_str(i)) for i in ids]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # One tensor per layer, each (1, n_head, T, T).
    maps = [m.float().cpu() for m in model.attention_maps(idx)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = args.layers if args.layers is not None else sorted({0, cfg.model.n_layer - 1})
    for layer in layers:
        path = out_dir / f"attention_layer{layer}.png"
        plot_layer(maps[layer][0], labels, layer, path)
        print(f"Saved {path}")

    layer, head = most_focused_head(maps)
    path = out_dir / "attention_closeup.png"
    plot_closeup(maps[layer][0, head], labels, f"Attention weights: layer {layer}, head {head}", path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
