"""Shared matplotlib style so every figure in the repo looks like one family.

Colours are a fixed, colour-blind-safe categorical order (blue, orange, aqua, ...);
magnitudes (heatmaps) use a single-hue light-to-dark blue ramp.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # render straight to image files; no window needed

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical colours, always used in this order (series 1, 2, 3, ...).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

TEXT = "#0b0b0b"  # primary ink
MUTED = "#52514e"  # secondary ink (axis labels, annotations)
GRID = "#e4e3df"  # recessive grid lines
SURFACE = "#ffffff"

# One hue, light -> dark: the right encoding for "how much" (attention weights).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues_seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
# Two opposite hues around a neutral midpoint: the right encoding for +/- values.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red_div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a9a", "#b52b2b"]
)


def apply_style() -> None:
    """Set matplotlib defaults: thin marks, quiet axes, readable text."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=SERIES),
        }
    )
