"""Plot training vs validation loss from a run's log.csv.

Usage (from the repository root):

    python scripts/plot_loss.py                                  # checkpoints/small -> results/loss_curve.png
    python scripts/plot_loss.py --run checkpoints/tiny_cpu --out results/loss_tiny.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`

import matplotlib.pyplot as plt  # noqa: E402
from plot_style import BLUE, MUTED, ORANGE, TEXT, apply_style  # noqa: E402

from mini_gpt.utils import read_csv_log  # noqa: E402


def plot_loss(run_dir: str | Path, out_path: str | Path) -> Path:
    """Draw the train and validation loss curves of one run and save the figure."""
    log = read_csv_log(Path(run_dir) / "log.csv")
    steps, train, val = log["step"], log["train_loss"], log["val_loss"]
    best = min(range(len(val)), key=lambda i: val[i])  # index of the lowest validation loss

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, train, color=BLUE, label="train")
    ax.plot(steps, val, color=ORANGE, label="validation")

    # Label the lines directly at their right-hand ends (in ink, not in the series colour).
    ax.annotate(f"train {train[-1]:.2f}", (steps[-1], train[-1]), xytext=(6, -3),
                textcoords="offset points", color=TEXT, fontsize=9, va="top")
    ax.annotate(f"validation {val[-1]:.2f}", (steps[-1], val[-1]), xytext=(6, 3),
                textcoords="offset points", color=TEXT, fontsize=9, va="bottom")

    # Mark the checkpoint that train.py kept (lowest validation loss).
    ax.plot(steps[best], val[best], "o", color=ORANGE, markersize=8,
            markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax.annotate(f"best val {val[best]:.3f}\n(step {int(steps[best])})", (steps[best], val[best]),
                xytext=(0, 22), textcoords="offset points", ha="center", color=MUTED, fontsize=9)

    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats per token)")
    ax.set_title(f"Training vs validation loss ({Path(run_dir).name})")
    ax.set_ylim(bottom=max(0.0, min(train + val) - 0.25), top=min(max(train + val), 4.5) + 0.1)
    ax.margins(x=0.02)
    ax.legend(loc="upper right")
    fig.subplots_adjust(right=0.84)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train vs validation loss.")
    parser.add_argument("--run", default="checkpoints/small", help="folder containing log.csv")
    parser.add_argument("--out", default="results/loss_curve.png")
    args = parser.parse_args()
    print(f"Saved {plot_loss(args.run, args.out)}")


if __name__ == "__main__":
    main()
