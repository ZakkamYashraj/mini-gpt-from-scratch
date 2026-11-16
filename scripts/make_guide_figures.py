"""Generate the figures of the learning guide into docs/figures/.

Usage (from the repository root):

    python scripts/make_guide_figures.py                 # all figures
    python scripts/make_guide_figures.py --only pipeline temperature

Two figures (``next_token`` and ``pos_emb``) need a trained checkpoint
(default checkpoints/small/best.pt); they are skipped if it does not exist.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import Colormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from plot_style import (AQUA, BLUE, DIVERGING, GRID, MUTED, ORANGE, SEQUENTIAL, TEXT,  # noqa: E402
                        apply_style)

from mini_gpt.utils import get_lr  # noqa: E402

OUT = Path("docs/figures")
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_AQUA, LIGHT_GREY = "#dbe9fb", "#fde3d6", "#d6f1e6", "#eeeeea"


# --------------------------------------------------------------------------- #
# Small drawing helpers for the diagrams
# --------------------------------------------------------------------------- #


def blank_axes(
    width: float, height: float, xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[Figure, Axes]:
    """A figure with one axes and nothing drawn on it (for box-and-arrow diagrams)."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def box(
    ax: Axes, x: float, y: float, w: float, h: float, text: str, fill: str = LIGHT_BLUE, edge: str = BLUE,
    fontsize: float = 9, weight: str = "normal", color: str = TEXT,
) -> None:
    """A rounded rectangle centred at (x, y) with text inside."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=color)


def arrow(
    ax: Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED,
    style: str = "-|>", width: float = 1.2, curve: float = 0.0,
) -> None:
    """An arrow from ``start`` to ``end`` (both (x, y) tuples)."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11, color=color,
                                 linewidth=width, connectionstyle=f"arc3,rad={curve}",
                                 shrinkA=0, shrinkB=0))


def cell_grid(
    ax: Axes, x0: float, y0: float, values: np.ndarray | list, fill: str, cell: float = 0.6,
    fontsize: float = 9, edge: str = "white", fmt: str = "{:g}", color: str = TEXT,
) -> None:
    """Draw a 2-D array as a grid of coloured cells with the numbers written inside."""
    values = np.atleast_2d(values)
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            ax.add_patch(Rectangle((x0 + c * cell, y0 - (r + 1) * cell), cell, cell,
                                   facecolor=fill, edgecolor=edge, linewidth=2))
            ax.text(x0 + (c + 0.5) * cell, y0 - (r + 0.5) * cell, fmt.format(value),
                    ha="center", va="center", fontsize=fontsize, color=color)


def save(fig: Figure, name: str) -> None:
    """Write the figure to docs/figures/<name>.png and close it."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  saved {OUT / name}.png")


# --------------------------------------------------------------------------- #
# Chapter 1
# --------------------------------------------------------------------------- #


def fig_pipeline() -> None:
    """Chapter 1: text -> tokenizer -> model -> probabilities -> sample, as a loop."""
    fig, ax = blank_axes(9.0, 2.9, (0, 10), (0, 3.2))
    y = 2.2
    box(ax, 0.9, y, 1.5, 0.8, '"First Citi"', LIGHT_GREY, MUTED)
    box(ax, 3.0, y, 1.5, 0.8, "Tokenizer", LIGHT_BLUE, BLUE, weight="bold")
    box(ax, 5.1, y, 1.6, 0.8, "[18, 47, 56, ...]", LIGHT_GREY, MUTED, fontsize=8)
    box(ax, 7.2, y, 1.5, 0.8, "GPT model", LIGHT_BLUE, BLUE, weight="bold")
    box(ax, 9.15, y, 1.5, 0.8, "probabilities\nfor next token", LIGHT_GREY, MUTED, fontsize=8)
    for a, b in [(1.65, 2.25), (3.75, 4.3), (5.9, 6.45), (7.95, 8.4)]:
        arrow(ax, (a, y), (b, y))
    for x, label in [(0.9, "text"), (5.1, "token IDs"), (9.15, "one number per\nvocabulary entry")]:
        ax.text(x, y + 0.62, label, ha="center", va="bottom", fontsize=8, color=MUTED)
    # the generation loop back to the text
    box(ax, 5.1, 0.6, 2.3, 0.7, 'sample one token: "z"', LIGHT_ORANGE, ORANGE, fontsize=8.5)
    arrow(ax, (9.15, y - 0.4), (6.27, 0.6), color=ORANGE, curve=-0.25)
    arrow(ax, (3.93, 0.6), (0.9, y - 0.4), color=ORANGE, curve=-0.25)
    ax.text(2.0, 0.55, "append to the text,\nthen repeat", ha="center", va="top", fontsize=8, color=MUTED)
    save(fig, "pipeline")


def fig_next_token(checkpoint: Path) -> None:
    """Chapter 1: the trained model's top predictions for the next character."""
    from generate import load_model

    model, tokenizer, _ = load_model(checkpoint, torch.device("cpu"))
    context = "First Citi"
    idx = torch.tensor([tokenizer.encode(context)])
    with torch.no_grad():
        logits, _ = model(idx)
    probs = F.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, 8)
    labels = [repr(tokenizer.token_to_str(i))[1:-1].replace(" ", "space") for i in top.indices.tolist()]
    values = top.values.tolist()

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.bar(labels, values, color=BLUE, width=0.6)
    for label, value in zip(labels, values):
        ax.text(label, value + 0.015, f"{value:.3f}" if value >= 0.0005 else "<0.001",
                ha="center", fontsize=8.5, color=TEXT)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("probability")
    ax.set_xlabel("candidate next character")
    ax.set_title(f'What comes after "{context}"?')
    ax.grid(axis="x", visible=False)
    save(fig, "next_token")


# --------------------------------------------------------------------------- #
# Chapter 2
# --------------------------------------------------------------------------- #


def fig_broadcasting() -> None:
    """Chapter 2: two examples of broadcasting drawn as grids of cells."""
    fig, ax = blank_axes(8.8, 1.9, (0, 14.9), (0.55, 3.45))
    c = 0.62
    ax.text(0.0, 3.1, "M   shape (2, 3)", fontsize=9, color=MUTED)
    cell_grid(ax, 0.0, 2.8, [[1, 2, 3], [4, 5, 6]], LIGHT_BLUE, c)
    ax.text(2.35, 2.15, "+", fontsize=16, ha="center", va="center")
    ax.text(2.9, 3.1, "row   shape (3,)", fontsize=9, color=MUTED)
    cell_grid(ax, 2.9, 2.8, [[10, 20, 30]], LIGHT_ORANGE, c)
    cell_grid(ax, 2.9, 2.8 - c, [[10, 20, 30]], LIGHT_GREY, c, color=MUTED)
    ax.text(3.83, 1.25, "repeated copy", fontsize=7.5, color=MUTED, ha="center")
    ax.text(5.3, 2.15, "=", fontsize=16, ha="center", va="center")
    ax.text(5.85, 3.1, "result   shape (2, 3)", fontsize=9, color=MUTED)
    cell_grid(ax, 5.85, 2.8, [[11, 22, 33], [14, 25, 36]], LIGHT_AQUA, c)

    x0 = 9.5
    ax.text(x0, 3.1, "row (3,)", fontsize=9, color=MUTED)
    cell_grid(ax, x0, 2.8, [[10, 20, 30]], LIGHT_ORANGE, c)
    cell_grid(ax, x0, 2.8 - c, [[10, 20, 30]], LIGHT_GREY, c, color=MUTED)
    ax.text(x0 + 2.25, 2.15, "+", fontsize=16, ha="center", va="center")
    ax.text(x0 + 2.7, 3.1, "col (2, 1)", fontsize=9, color=MUTED)
    cell_grid(ax, x0 + 2.7, 2.8, [[100], [200]], LIGHT_ORANGE, c)
    cell_grid(ax, x0 + 2.7 + c, 2.8, [[100, 100], [200, 200]], LIGHT_GREY, c, color=MUTED, fontsize=8)
    ax.text(x0 + 0.95, 0.9, "both arrays are stretched to (2, 3)", fontsize=8, color=MUTED, ha="left")
    ax.plot([8.75, 8.75], [0.7, 3.3], color=GRID, linewidth=1)
    save(fig, "broadcasting")


def fig_matmul() -> None:
    """Chapter 2: a worked matrix multiplication and the shape rule."""
    fig, ax = blank_axes(8.6, 2.2, (0, 14.2), (0.2, 3.6))
    c = 0.55
    A = [[1, 2], [3, 4], [5, 6]]
    B = [[1, 0, 2, 0], [0, 1, 0, 2]]
    C = [[1, 2, 2, 4], [3, 4, 6, 8], [5, 6, 10, 12]]
    ax.text(0.2, 3.25, "A   (3, 2)", fontsize=9, color=MUTED)
    cell_grid(ax, 0.2, 3.0, A, LIGHT_BLUE, c)
    ax.add_patch(Rectangle((0.2, 3.0 - 3 * c), 2 * c, c, fill=False, edgecolor=BLUE, linewidth=2))
    ax.text(1.75, 2.2, "@", fontsize=14, ha="center", va="center")
    ax.text(2.3, 3.25, "B   (2, 4)", fontsize=9, color=MUTED)
    cell_grid(ax, 2.3, 3.0, B, LIGHT_ORANGE, c)
    ax.add_patch(Rectangle((2.3 + 3 * c, 3.0 - 2 * c), c, 2 * c, fill=False, edgecolor=ORANGE, linewidth=2))
    ax.text(4.95, 2.2, "=", fontsize=14, ha="center", va="center")
    ax.text(5.4, 3.25, "C   (3, 4)", fontsize=9, color=MUTED)
    cell_grid(ax, 5.4, 3.0, C, LIGHT_AQUA, c)
    ax.add_patch(Rectangle((5.4 + 3 * c, 3.0 - 3 * c), c, c, fill=False, edgecolor=TEXT, linewidth=2))
    ax.text(0.2, 0.55, "C[2, 3]  =  row 2 of A  ·  column 3 of B  =  5·0 + 6·2  =  12", fontsize=9, color=TEXT)
    # the shape rule
    ax.text(11.4, 2.6, "( n , k )  @  ( k , m )", fontsize=12, ha="center", family="DejaVu Sans Mono")
    ax.text(11.4, 1.2, "( n , m )", fontsize=12, ha="center", family="DejaVu Sans Mono", weight="bold")
    arrow(ax, (11.4, 2.25), (11.4, 1.6))
    ax.text(11.4, 3.15, "inner sizes must match", fontsize=8, color=MUTED, ha="center")
    ax.text(11.4, 0.7, "outer sizes remain", fontsize=8, color=MUTED, ha="center")
    ax.plot([8.9, 8.9], [0.4, 3.4], color=GRID, linewidth=1)
    save(fig, "matmul")


# --------------------------------------------------------------------------- #
# Chapters 3 and 4
# --------------------------------------------------------------------------- #


def fig_comp_graph() -> None:
    """Chapter 3: the computation graph of f = x^2 * y + 3y, forward and backward."""
    fig, ax = blank_axes(8.4, 2.9, (0, 12), (0, 4))
    nodes = {
        "x": (0.8, 3.1, "x = 2", LIGHT_GREY), "y": (0.8, 0.9, "y = 5", LIGHT_GREY),
        "sq": (3.3, 3.1, "a = x²\n= 4", LIGHT_BLUE), "mul": (5.9, 3.1, "b = a·y\n= 20", LIGHT_BLUE),
        "three": (5.9, 0.9, "c = 3·y\n= 15", LIGHT_BLUE), "f": (8.8, 2.0, "f = b + c\n= 35", LIGHT_AQUA),
    }
    for x, y, text, fill in nodes.values():
        box(ax, x, y, 1.5, 0.95, text, fill, BLUE if fill != LIGHT_GREY else MUTED)
    forward = [("x", "sq"), ("sq", "mul"), ("y", "mul"), ("y", "three"), ("mul", "f"), ("three", "f")]
    for a, b in forward:
        (xa, ya, *_), (xb, yb, *_) = nodes[a], nodes[b]
        dx = 0.78
        arrow(ax, (xa + dx, ya + (0.15 if yb > ya else -0.15 if yb < ya else 0)),
              (xb - dx, yb + (-0.15 if yb > ya else 0.15 if yb < ya else 0)), color=TEXT)
    backward = [(7.4, 2.95, "∂f/∂b = 1"), (7.4, 1.0, "∂f/∂c = 1"), (4.6, 3.75, "∂f/∂a = y = 5"),
                (2.05, 3.75, "∂f/∂x = 5·2x = 20"), (3.0, 0.3, "∂f/∂y = a + 3 = 7")]
    for x, y, text in backward:
        ax.text(x, y, text, fontsize=8.5, color=ORANGE, ha="center", weight="bold")
    ax.text(10.9, 3.5, "forward:\nvalues", fontsize=8.5, color=TEXT, ha="center")
    ax.text(10.9, 0.6, "backward:\nderivatives", fontsize=8.5, color=ORANGE, ha="center", weight="bold")
    arrow(ax, (10.3, 2.9), (11.5, 2.9), color=TEXT)
    arrow(ax, (11.5, 1.3), (10.3, 1.3), color=ORANGE)
    save(fig, "comp_graph")


def fig_network() -> None:
    """Chapter 4: the 1 -> 16 -> 1 network as circles and lines."""
    fig, ax = blank_axes(6.6, 3.3, (0, 10), (0, 5))
    hidden_y = np.linspace(0.5, 4.5, 8)
    for y in hidden_y:
        ax.plot([1.6, 5.0], [2.5, y], color=BLUE, linewidth=0.7, alpha=0.6)
        ax.plot([5.0, 8.4], [y, 2.5], color=ORANGE, linewidth=0.7, alpha=0.6)
    for y in hidden_y:
        ax.add_patch(plt.Circle((5.0, y), 0.22, facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.2, zorder=3))
    ax.text(5.0, 2.5, "...", ha="center", va="center", fontsize=9, color=MUTED, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    for x, label in [(1.6, "x"), (8.4, "pred")]:
        ax.add_patch(plt.Circle((x, 2.5), 0.36, facecolor=LIGHT_GREY, edgecolor=MUTED, linewidth=1.2, zorder=3))
        ax.text(x, 2.5, label, ha="center", va="center", fontsize=8.5, zorder=4)
    ax.text(1.6, 1.75, "input\n(1 number)", ha="center", va="top", fontsize=8, color=MUTED)
    ax.text(5.0, 4.95, "hidden layer: 16 neurons, each computes tanh(w·x + b)", ha="center",
            fontsize=8, color=MUTED)
    ax.text(8.4, 1.75, "output\n(1 number)", ha="center", va="top", fontsize=8, color=MUTED)
    ax.text(3.2, 0.25, "W1 (1, 16), b1 (16,)", ha="center", fontsize=8, color=BLUE)
    ax.text(6.8, 0.25, "W2 (16, 1), b2 (1,)", ha="center", fontsize=8, color=ORANGE)
    save(fig, "network")


def fig_activations() -> None:
    """Chapter 4: the ReLU, tanh and GELU curves."""
    z = np.linspace(-4, 4, 400)
    gelu = np.array([0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in z])
    curves = [("ReLU:  max(0, z)", np.maximum(0, z), BLUE), ("tanh(z)", np.tanh(z), ORANGE), ("GELU(z)", gelu, AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True)
    for ax, (title, values, color) in zip(axes, curves):
        ax.axhline(0, color=GRID, linewidth=1)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.plot(z, values, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("z")
        ax.set_ylim(-1.5, 4.2)
    axes[0].set_ylabel("output")
    save(fig, "activations")


def fig_gradient_descent() -> None:
    """Chapter 4: gradient-descent paths for three learning rates on a 2-D bowl."""
    def loss(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """A bowl-shaped loss that is steeper along parameter 2."""
        return 0.5 * w1**2 + 2.5 * w2**2

    def path(lr: float, steps: int = 18) -> np.ndarray:
        """The points visited by gradient descent with learning rate ``lr``."""
        w = np.array([-3.5, 1.4])
        points = [w.copy()]
        for _ in range(steps):
            w = w - lr * np.array([w[0], 5 * w[1]])  # gradient of the loss above
            points.append(w.copy())
        return np.array(points)

    g1, g2 = np.meshgrid(np.linspace(-4.2, 4.2, 200), np.linspace(-2.4, 2.4, 200))
    settings = [(0.1, "learning rate 0.1: steady"), (0.37, "learning rate 0.37: zig-zags"),
                (0.42, "learning rate 0.42: diverges")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), sharey=True)
    for ax, (lr, title) in zip(axes, settings):
        ax.contour(g1, g2, loss(g1, g2), levels=[0.2, 1, 2.5, 5, 8, 12], colors=GRID, linewidths=1)
        p = path(lr, steps=7 if lr > 0.4 else 18)
        ax.plot(p[:, 0], p[:, 1], "-o", color=ORANGE, markersize=3.5, linewidth=1.3)
        ax.plot(p[0, 0], p[0, 1], "o", color=TEXT, markersize=6)
        ax.plot(0, 0, "*", color=BLUE, markersize=12)
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-2.4, 2.4)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("parameter 1")
        ax.grid(False)
    axes[0].set_ylabel("parameter 2")
    axes[0].annotate("start", (-3.5, 1.4), xytext=(-3.9, 1.95), fontsize=8, color=MUTED)
    axes[0].annotate("minimum", (0, 0), xytext=(0.4, -0.75), fontsize=8, color=MUTED)
    save(fig, "gradient_descent")


def fig_tiny_network() -> None:
    """Chapter 4: the tiny network's fit to sin(x) during training, and its loss curve."""
    rng = np.random.default_rng(0)
    X = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    Y = np.sin(X)
    H = 16
    W1, b1 = rng.normal(0.0, 1.0, size=(1, H)), np.zeros(H)
    W2, b2 = rng.normal(0.0, 0.1, size=(H, 1)), np.zeros(1)
    snapshots, history = {}, []
    for step in range(5001):  # the same training as examples/ch04_tiny_network.py
        h = np.tanh(X @ W1 + b1)
        pred = h @ W2 + b2
        history.append(float(((pred - Y) ** 2).mean()))
        if step in (0, 200, 5000):
            snapshots[step] = pred.copy()
        d_pred = 2 * (pred - Y) / len(X)
        d_h = d_pred @ W2.T
        d_z1 = d_h * (1 - h**2)
        W2 -= 0.05 * (h.T @ d_pred)
        b2 -= 0.05 * d_pred.sum(axis=0)
        W1 -= 0.05 * (X.T @ d_z1)
        b1 -= 0.05 * d_z1.sum(axis=0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    left.plot(X, Y, color=TEXT, linewidth=1.2, linestyle="--", label="sin(x)  (target)")
    for (step, pred), color in zip(snapshots.items(), [AQUA, ORANGE, BLUE]):
        left.plot(X, pred, color=color, label=f"network after {step} steps")
    left.set_xlabel("x")
    left.set_ylabel("y")
    left.set_title("The network's curve during training", fontsize=10)
    left.legend(fontsize=7.5, loc="upper left")
    right.plot(history, color=BLUE)
    right.set_yscale("log")
    right.set_xlabel("training step")
    right.set_ylabel("loss (mean squared error)")
    right.set_title("Loss", fontsize=10)
    save(fig, "tiny_network_fit")


# --------------------------------------------------------------------------- #
# Chapter 5
# --------------------------------------------------------------------------- #


def fig_overfitting() -> None:
    """Chapter 5: an illustrative pair of train/validation curves showing overfitting."""
    steps = np.linspace(0, 1, 200)
    train = 0.35 + 3.8 * np.exp(-6 * steps) - 0.25 * steps
    val = 1.45 + 2.7 * np.exp(-7 * steps) + 0.9 * np.clip(steps - 0.42, 0, None) ** 1.6
    best = int(np.argmin(val))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(steps, train, color=BLUE, label="training loss")
    ax.plot(steps, val, color=ORANGE, label="validation loss")
    ax.axvline(steps[best], color=MUTED, linewidth=1, linestyle=":")
    ax.plot(steps[best], val[best], "o", color=ORANGE, markeredgecolor="white", markeredgewidth=2, markersize=9)
    ax.annotate("best checkpoint:\nlowest validation loss", (steps[best], val[best]), xytext=(0.2, 2.6),
                fontsize=8.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.text(0.17, 0.5, "still learning\n(both fall)", fontsize=8.5, color=MUTED, ha="center")
    ax.text(0.74, 2.35, "overfitting: training loss keeps\nfalling, validation loss rises", fontsize=8.5,
            color=MUTED, ha="center")
    ax.annotate("", (0.99, val[-1]), (0.99, train[-1]), arrowprops=dict(arrowstyle="<->", color=MUTED))
    ax.text(0.97, (val[-1] + train[-1]) / 2, "generalisation gap", fontsize=8, color=MUTED, ha="right", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("training time")
    ax.set_ylabel("loss")
    ax.set_title("Illustration: how to recognise overfitting")
    ax.legend(loc="upper right")
    ax.grid(False)
    save(fig, "overfitting")


def fig_batch() -> None:
    """Chapter 5: one training sequence and its targets (shifted by one)."""
    text = "First Citizen:"
    fig, ax = blank_axes(8.2, 2.1, (0, 15.2), (0, 3.6))
    c = 0.9
    for i, ch in enumerate(text[:9]):
        in_x, in_y = i < 8, 1 <= i
        ax.add_patch(Rectangle((1.9 + i * c, 2.3), c, c * 0.9, facecolor=LIGHT_BLUE if in_x else LIGHT_GREY,
                               edgecolor="white", linewidth=2))
        ax.text(1.9 + (i + 0.5) * c, 2.3 + 0.4, ch if ch != " " else "␣", ha="center", va="center", fontsize=10,
                family="DejaVu Sans Mono")
        if in_y:
            ax.add_patch(Rectangle((1.9 + (i - 1) * c, 0.7), c, c * 0.9, facecolor=LIGHT_ORANGE,
                                   edgecolor="white", linewidth=2))
            ax.text(1.9 + (i - 0.5) * c, 0.7 + 0.4, ch if ch != " " else "␣", ha="center", va="center",
                    fontsize=10, family="DejaVu Sans Mono")
            arrow(ax, (1.9 + (i - 0.5) * c, 2.25), (1.9 + (i - 0.5) * c, 1.6), color=MUTED, width=0.8)
    ax.text(1.7, 2.7, "x (input)", ha="right", va="center", fontsize=9, color=BLUE)
    ax.text(1.7, 1.1, "y (target)", ha="right", va="center", fontsize=9, color=ORANGE)
    ax.text(10.6, 2.7, "8 tokens taken from the text", fontsize=8.5, color=MUTED, va="center")
    ax.text(10.6, 1.1, "the same tokens, shifted\none place to the left", fontsize=8.5, color=MUTED, va="center")
    ax.text(1.9, 0.2, "position t of y is the correct answer for the prediction made at position t of x",
            fontsize=8.5, color=MUTED)
    save(fig, "batch_xy")


# --------------------------------------------------------------------------- #
# Chapter 6
# --------------------------------------------------------------------------- #


def fig_bpe_compression() -> None:
    """Chapter 6: length of Tiny Shakespeare as BPE merges are added (re-runs the training loop)."""
    from mini_gpt.tokenizer import _CHUNK_PATTERN, _merge_pair

    text = Path("data/tinyshakespeare.txt").read_text(encoding="utf-8")
    chunk_counts = Counter(_CHUNK_PATTERN.findall(text))
    words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
    counts = list(chunk_counts.values())
    lengths = [sum(len(w) * n for w, n in zip(words, counts))]
    for new_id in range(256, 512):  # the training loop of BPETokenizer.train, recording the length
        pairs: Counter = Counter()
        for ids, count in zip(words, counts):
            for pair in zip(ids, ids[1:]):
                pairs[pair] += count
        best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))
        words = [_merge_pair(ids, best, new_id) for ids in words]
        lengths.append(sum(len(w) * n for w, n in zip(words, counts)))

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(range(len(lengths)), np.array(lengths) / 1e6, color=BLUE)
    ax.set_xlabel("number of merges learned")
    ax.set_ylabel("length of Tiny Shakespeare\n(millions of tokens)")
    ax.set_ylim(0, 1.2)
    ax.set_title("Each BPE merge makes the text shorter")
    ax.annotate(f"0 merges: {lengths[0] / 1e6:.2f}M tokens\n(one per byte)", (0, lengths[0] / 1e6),
                xytext=(26, 1.0), fontsize=8.5, color=MUTED)
    ax.annotate(f"256 merges: {lengths[-1] / 1e6:.2f}M tokens", (256, lengths[-1] / 1e6), xytext=(165, 0.68),
                fontsize=8.5, color=MUTED)
    save(fig, "bpe_compression")


# --------------------------------------------------------------------------- #
# Chapter 7
# --------------------------------------------------------------------------- #


def fig_embedding_lookup() -> None:
    """Chapter 7: an embedding lookup as rows copied out of a table."""
    rng = np.random.default_rng(3)
    table = np.round(rng.normal(0, 1, size=(5, 4)), 1)
    fig, ax = blank_axes(8.2, 2.9, (0, 14), (-0.3, 4.6))
    c = 0.62
    ax.text(0.2, 4.25, "token IDs", fontsize=9, color=MUTED)
    cell_grid(ax, 0.2, 3.9, [[3], [1], [3]], LIGHT_GREY, c)
    for label, y in zip(["'l'", "'e'", "'l'"], [3.6, 2.98, 2.36]):
        ax.text(1.0, y, label, fontsize=8, color=MUTED, va="center")
    ax.text(4.2, 4.25, "embedding table  (V, C) = (5, 4)", fontsize=9, color=MUTED)
    cell_grid(ax, 4.2, 3.9, table, LIGHT_BLUE, c, fontsize=8, fmt="{:.1f}")
    for r in (1, 3):
        ax.add_patch(Rectangle((4.2, 3.9 - (r + 1) * c), 4 * c, c, fill=False,
                               edgecolor=ORANGE, linewidth=2))
    for r in range(5):
        ax.text(4.0, 3.9 - (r + 0.5) * c, f"row {r}", fontsize=7.5, color=MUTED, ha="right", va="center")
    ax.text(10.4, 4.25, "output  (T, C) = (3, 4)", fontsize=9, color=MUTED)
    cell_grid(ax, 10.4, 3.9, table[[3, 1, 3]], LIGHT_ORANGE, c, fontsize=8, fmt="{:.1f}")
    arrow(ax, (1.7, 3.0), (3.1, 3.0))
    arrow(ax, (7.0, 3.0), (10.1, 3.0))
    ax.text(2.4, 3.2, "look up", fontsize=8, color=MUTED, ha="center")
    ax.text(8.55, 3.2, "copy rows\n3, 1, 3", fontsize=8, color=MUTED, ha="center")
    ax.text(0.2, 0.2, "An embedding is a lookup: ID 3 always returns row 3. The numbers in the table are learned.",
            fontsize=8.5, color=MUTED)
    save(fig, "embedding_lookup")


def fig_pos_emb(checkpoint: Path) -> None:
    """Chapter 7: cosine similarity between the trained positional embeddings."""
    from generate import load_model

    model, _, _ = load_model(checkpoint, torch.device("cpu"))
    if model.pos_emb is None:
        return
    weight = F.normalize(model.pos_emb.weight.detach(), dim=-1)
    similarity = (weight @ weight.T).numpy()  # cosine similarity between every pair of positions
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    image = ax.imshow(similarity, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xlabel("position")
    ax.set_ylabel("position")
    ax.set_title("Similarity of learned positional embeddings", fontsize=10)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="cosine similarity")
    save(fig, "pos_emb_similarity")


# --------------------------------------------------------------------------- #
# Chapters 9 and 10
# --------------------------------------------------------------------------- #


def _heatmap(
    ax: Axes, values: torch.Tensor | np.ndarray, title: str, labels: list[str], fmt: str = "{:.2f}",
    vmax: float = 1.0, cmap: Colormap = SEQUENTIAL,
) -> None:
    """Draw a small matrix as coloured cells with the numbers written inside."""
    values = np.asarray(values, dtype=float)
    shown = np.where(np.isfinite(values), values, np.nan)
    ax.imshow(np.nan_to_num(shown, nan=0.0), cmap=cmap, vmin=0, vmax=vmax)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            text = "-inf" if np.isneginf(v) else fmt.format(v)
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5,
                    color="white" if np.isfinite(v) and v > 0.6 * vmax else TEXT)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.tick_params(length=0)
    ax.grid(False)


def fig_attention_example() -> None:
    """Chapters 9-10: scores, unmasked weights and masked weights of the worked example."""
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Q, K = X * 2.0, X
    scores = Q @ K.T / math.sqrt(2)
    mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    unmasked = F.softmax(scores, dim=-1)
    masked = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    labels = ["token 0", "token 1", "token 2"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9))
    _heatmap(axes[0], scores, "scores = Q·Kᵀ / √2", labels, vmax=3.0)
    _heatmap(axes[1], unmasked, "weights without mask", labels)
    _heatmap(axes[2], masked, "weights with causal mask", labels)
    axes[0].set_ylabel("query (who is looking)")
    axes[1].set_xlabel("key (who is being looked at)")
    fig.tight_layout()
    save(fig, "attention_example")


def fig_causal_mask() -> None:
    """Chapter 10: the causal mask, and one sequence as four training examples."""
    labels = list("hello")
    mask = np.tril(np.ones((5, 5)))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    ax = axes[0]
    ax.imshow(mask, cmap=SEQUENTIAL, vmin=0, vmax=2.2)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "can see" if mask[i, j] else "hidden", ha="center", va="center", fontsize=7,
                    color=TEXT if mask[i, j] else MUTED)
    ax.set_xticks(range(5), labels)
    ax.set_yticks(range(5), labels)
    ax.set_xlabel("token being looked at")
    ax.set_ylabel("token doing the looking")
    ax.set_title("The causal mask (lower triangle)", fontsize=10)
    ax.tick_params(length=0)
    ax.grid(False)

    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("One sequence = four training examples", fontsize=10)
    rows = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
    for r, (context, target) in enumerate(rows):
        y = 4.9 - r * 1.2
        for i, ch in enumerate("hell"):
            visible = i < len(context)
            ax.add_patch(Rectangle((0.6 + i * 0.95, y - 0.4), 0.9, 0.8, facecolor=LIGHT_BLUE if visible else "#f6f6f4",
                                   edgecolor="white", linewidth=2))
            ax.text(0.6 + i * 0.95 + 0.45, y, ch, ha="center", va="center", fontsize=10,
                    color=TEXT if visible else "#c8c7c1", family="DejaVu Sans Mono")
        arrow(ax, (4.7, y), (5.9, y))
        ax.add_patch(Rectangle((6.2, y - 0.4), 0.9, 0.8, facecolor=LIGHT_ORANGE, edgecolor="white", linewidth=2))
        ax.text(6.65, y, target, ha="center", va="center", fontsize=10, family="DejaVu Sans Mono")
        ax.text(7.4, y, f"position {r}", fontsize=8, color=MUTED, va="center")
    ax.text(0.6, 0.35, "blue = visible to that position,  orange = what it must predict", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "causal_mask")


# --------------------------------------------------------------------------- #
# Chapters 11 and 12
# --------------------------------------------------------------------------- #


def fig_transformer_block() -> None:
    """Chapter 11: one pre-norm transformer block with its residual connections."""
    fig, ax = blank_axes(6.4, 5.6, (0, 10), (0, 12.4))
    x = 4.2
    box(ax, x, 0.7, 3.2, 0.8, "x  (B, T, C)", LIGHT_GREY, MUTED)
    box(ax, x, 2.5, 3.2, 0.8, "LayerNorm", "#ffffff", MUTED)
    box(ax, x, 4.0, 3.2, 0.9, "Multi-head attention", LIGHT_BLUE, BLUE, weight="bold")
    ax.add_patch(plt.Circle((x, 5.5), 0.3, facecolor="white", edgecolor=TEXT, linewidth=1.2))
    ax.text(x, 5.5, "+", ha="center", va="center", fontsize=12)
    box(ax, x, 7.2, 3.2, 0.8, "LayerNorm", "#ffffff", MUTED)
    box(ax, x, 8.7, 3.2, 0.9, "Feed-forward", LIGHT_AQUA, AQUA, weight="bold")
    ax.add_patch(plt.Circle((x, 10.2), 0.3, facecolor="white", edgecolor=TEXT, linewidth=1.2))
    ax.text(x, 10.2, "+", ha="center", va="center", fontsize=12)
    box(ax, x, 11.6, 3.2, 0.8, "x  (B, T, C)", LIGHT_GREY, MUTED)
    for a, b in [(1.1, 2.1), (2.9, 3.55), (4.45, 5.2), (5.8, 6.8), (7.6, 8.25), (9.15, 9.9), (10.5, 11.2)]:
        arrow(ax, (x, a), (x, b), color=TEXT)
    # residual (skip) connections
    for y0, y1 in [(1.5, 5.5), (6.2, 10.2)]:
        ax.plot([x, 1.5, 1.5], [y0, y0, y1], color=ORANGE, linewidth=1.5)
        arrow(ax, (1.5, y1), (x - 0.3, y1), color=ORANGE, width=1.5)
    ax.text(1.3, 3.5, "residual\nconnection", fontsize=8, color=ORANGE, ha="right", va="center", weight="bold")
    ax.text(1.3, 8.2, "residual\nconnection", fontsize=8, color=ORANGE, ha="right", va="center", weight="bold")
    ax.text(6.1, 4.0, "tokens exchange\ninformation", fontsize=8, color=MUTED, va="center")
    ax.text(6.1, 8.7, "each token processes\nwhat it gathered", fontsize=8, color=MUTED, va="center")
    ax.text(6.1, 2.5, "pre-norm: normalise\nbefore each sub-layer", fontsize=8, color=MUTED, va="center")
    save(fig, "transformer_block")


def fig_gpt_architecture() -> None:
    """Chapter 12: the whole model from token IDs to logits, with weight tying."""
    fig, ax = blank_axes(7.0, 6.2, (0, 11), (0, 13.6))
    x = 4.6
    box(ax, x, 0.6, 4.4, 0.8, "token IDs  (B, T)", LIGHT_GREY, MUTED)
    box(ax, x - 1.25, 2.3, 2.3, 0.9, "token\nembedding", LIGHT_BLUE, BLUE, fontsize=8.5)
    box(ax, x + 1.25, 2.3, 2.3, 0.9, "positional\nembedding", LIGHT_BLUE, BLUE, fontsize=8.5)
    ax.add_patch(plt.Circle((x, 3.8), 0.3, facecolor="white", edgecolor=TEXT, linewidth=1.2))
    ax.text(x, 3.8, "+", ha="center", va="center", fontsize=12)
    arrow(ax, (x - 1.25, 1.0), (x - 1.25, 1.85), color=TEXT)
    arrow(ax, (x + 1.25, 1.0), (x + 1.25, 1.85), color=TEXT)
    arrow(ax, (x - 1.25, 2.75), (x - 0.25, 3.65), color=TEXT)
    arrow(ax, (x + 1.25, 2.75), (x + 0.25, 3.65), color=TEXT)
    for i, y in enumerate([5.3, 6.5, 8.6]):
        label = ["Block 1", "Block 2", "Block N"][i]
        box(ax, x, y, 4.4, 0.9, f"{label}:  attention + feed-forward", LIGHT_AQUA, AQUA, fontsize=8.5)
    ax.text(x, 7.55, "...", ha="center", va="center", fontsize=14, color=MUTED)
    box(ax, x, 10.1, 4.4, 0.8, "final LayerNorm", "#ffffff", MUTED)
    box(ax, x, 11.5, 4.4, 0.9, "LM head (Linear):  C → V", LIGHT_ORANGE, ORANGE, weight="bold", fontsize=8.5)
    box(ax, x, 13.0, 4.4, 0.8, "logits  (B, T, V)", LIGHT_GREY, MUTED)
    for a, b in [(4.1, 4.85), (5.75, 6.05), (9.05, 9.7), (10.5, 11.05), (11.95, 12.6)]:
        arrow(ax, (x, a), (x, b), color=TEXT)
    for y, text in [(3.8, "(B, T, C)"), (9.35, "(B, T, C)"), (10.78, "(B, T, C)")]:
        ax.text(x + 0.45, y, text, fontsize=7.5, color=MUTED, va="center")
    # weight tying
    ax.plot([x - 2.4, 0.9, 0.9, x - 2.2], [2.3, 2.3, 11.5, 11.5], color=ORANGE, linewidth=1.3, linestyle="--")
    ax.text(0.7, 6.9, "weight tying:\nthe same (V, C)\nmatrix", fontsize=8, color=ORANGE, ha="right", va="center",
            weight="bold")
    ax.text(7.2, 6.9, "n_layer blocks,\nall the same shape", fontsize=8, color=MUTED, va="center")
    ax.text(7.2, 13.0, "one score per vocabulary\nentry, at every position", fontsize=8, color=MUTED, va="center")
    save(fig, "gpt_architecture")


# --------------------------------------------------------------------------- #
# Chapters 13 and 14
# --------------------------------------------------------------------------- #


def fig_lr_schedule() -> None:
    """Chapter 13: the warmup + cosine learning-rate schedule of configs/small.yaml."""
    max_steps, warmup = 5000, 100
    steps = np.arange(max_steps + 1)
    lrs = [get_lr(int(s), 1e-3, 1e-4, warmup, max_steps) for s in steps]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(steps, lrs, color=BLUE)
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedule of configs/small.yaml")
    ax.annotate("linear warmup\n(first 100 steps)", (50, 0.0005), xytext=(330, 0.0004), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.annotate("cosine decay", (2600, lrs[2600]), xytext=(2900, 0.00068), fontsize=8.5, color=MUTED)
    ax.annotate("ends at min_lr = 0.0001", (5000, 1e-4), xytext=(5000, 0.00042), fontsize=8.5, color=MUTED,
                ha="right", arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_ylim(0, 0.00108)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    save(fig, "lr_schedule")


def fig_temperature() -> None:
    """Chapter 14: one set of logits at three temperatures."""
    tokens = ["e", "a", "o", "i", "u", "z"]
    logits = torch.tensor([3.0, 2.5, 2.0, 1.0, 0.0, -2.0])
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), sharey=True)
    for ax, (T, note) in zip(axes, [(0.5, "sharper: safe, repetitive"), (1.0, "the model's own distribution"),
                                    (2.0, "flatter: surprising, error-prone")]):
        probs = F.softmax(logits / T, dim=-1).tolist()
        ax.bar(tokens, probs, color=BLUE, width=0.6)
        for token, p in zip(tokens, probs):
            ax.text(token, p + 0.015, f"{p:.2f}", ha="center", fontsize=8, color=TEXT)
        ax.set_title(f"temperature {T}", fontsize=10)
        ax.set_xlabel(note, fontsize=8.5)
        ax.set_ylim(0, 0.78)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("probability")
    save(fig, "temperature")


def fig_topk_topp() -> None:
    """Chapter 14: the effect of top-k and top-p on a toy distribution."""
    from mini_gpt.model import top_k_filter, top_p_filter

    tokens = ["e", "a", "o", "i", "u", "z"]
    logits = torch.tensor([[3.0, 2.5, 2.0, 1.0, 0.0, -2.0]])
    original = F.softmax(logits, dim=-1)[0].tolist()
    panels = [("original", original, [True] * 6),
              ("top-k, k = 3", F.softmax(top_k_filter(logits, 3), dim=-1)[0].tolist(), None),
              ("top-p, p = 0.5", F.softmax(top_p_filter(logits, 0.5), dim=-1)[0].tolist(), None)]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), sharey=True)
    for ax, (title, probs, _) in zip(axes, panels):
        colors = [BLUE if p > 0 else GRID for p in probs]
        ax.bar(tokens, [max(p, 0.004) for p in probs], color=colors, width=0.6)
        for token, p in zip(tokens, probs):
            ax.text(token, p + 0.015, f"{p:.2f}" if p > 0 else "cut", ha="center", fontsize=8,
                    color=TEXT if p > 0 else MUTED)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.78)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("probability")
    axes[1].set_xlabel("keep the 3 best, renormalise", fontsize=8.5)
    axes[2].set_xlabel("keep tokens until they cover 50%", fontsize=8.5)
    save(fig, "topk_topp")


FIGURES = {
    "pipeline": fig_pipeline,
    "broadcasting": fig_broadcasting,
    "matmul": fig_matmul,
    "comp_graph": fig_comp_graph,
    "network": fig_network,
    "activations": fig_activations,
    "gradient_descent": fig_gradient_descent,
    "tiny_network": fig_tiny_network,
    "overfitting": fig_overfitting,
    "batch": fig_batch,
    "bpe_compression": fig_bpe_compression,
    "embedding_lookup": fig_embedding_lookup,
    "attention_example": fig_attention_example,
    "causal_mask": fig_causal_mask,
    "transformer_block": fig_transformer_block,
    "gpt_architecture": fig_gpt_architecture,
    "lr_schedule": fig_lr_schedule,
    "temperature": fig_temperature,
    "topk_topp": fig_topk_topp,
}
NEEDS_CHECKPOINT = {"next_token": fig_next_token, "pos_emb": fig_pos_emb}


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate the learning-guide figures.")
    parser.add_argument("--only", nargs="*", help="names of the figures to (re)generate")
    parser.add_argument("--checkpoint", default="checkpoints/small/best.pt")
    args = parser.parse_args()
    apply_style()

    wanted = args.only or [*FIGURES, *NEEDS_CHECKPOINT]
    checkpoint = Path(args.checkpoint)
    for name in wanted:
        if name in FIGURES:
            FIGURES[name]()
        elif name in NEEDS_CHECKPOINT:
            if checkpoint.exists():
                NEEDS_CHECKPOINT[name](checkpoint)
            else:
                print(f"  skipped {name}: no checkpoint at {checkpoint} (train the model first)")
        else:
            raise SystemExit(f"Unknown figure {name!r}. Choose from: {', '.join([*FIGURES, *NEEDS_CHECKPOINT])}")


if __name__ == "__main__":
    main()
