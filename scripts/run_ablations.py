"""Ablation study: train several variants that each change ONE thing, and compare them.

An *ablation* removes or changes one component of a system to measure how much
that component matters. Every variant below starts from configs/ablation.yaml.

Usage (from the repository root):

    python scripts/run_ablations.py                    # run everything (skips finished runs)
    python scripts/run_ablations.py --only one_head    # run a single variant
    python scripts/run_ablations.py --seeds 1337 1338  # repeat with several seeds
    python scripts/run_ablations.py --max_steps 300    # quick smoke test

Outputs: results/ablations.md (table), results/ablations.json (raw numbers),
results/ablations.png (dot plot).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`, `import train`

import matplotlib.pyplot as plt  # noqa: E402
from plot_style import BLUE, MUTED, TEXT, apply_style  # noqa: E402

from mini_gpt.config import load_config  # noqa: E402
from mini_gpt.data import load_dataset  # noqa: E402
from mini_gpt.utils import fix_console_encoding  # noqa: E402
from train import train  # noqa: E402

BASE_CONFIG = "configs/ablation.yaml"

# name -> (description, config overrides). Each variant changes exactly one thing.
VARIANTS: dict[str, tuple[str, list[str]]] = {
    "baseline": ("Baseline: char tokenizer, 4 heads, positional embeddings, tied weights", []),
    "one_head": ("1 attention head instead of 4 (same total width)", ["model.n_head=1"]),
    "no_pos_emb": ("No positional embeddings", ["model.use_pos_emb=false"]),
    "untied": ("Output layer NOT tied to the token embedding", ["model.tie_weights=false"]),
    "bpe": ("BPE tokenizer (vocab 512) instead of characters", ["data.tokenizer=bpe"]),
}


def run_variant(name: str, seed: int, max_steps: int | None, force: bool) -> dict:
    """Train one variant with one seed (or load its cached result) and return its summary."""
    _, overrides = VARIANTS[name]
    out_dir = Path("checkpoints/ablations") / f"{name}_seed{seed}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"[skip] {name} (seed {seed}) already done: {summary_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    overrides = overrides + [f"train.seed={seed}", f"train.out_dir={out_dir.as_posix()}"]
    if max_steps is not None:
        overrides.append(f"train.max_steps={max_steps}")
    cfg = load_config(BASE_CONFIG, overrides)
    cfg.name = f"{name}_seed{seed}"

    print(f"\n{'=' * 70}\nAblation variant: {name} (seed {seed})\n{'=' * 70}")
    summary = train(cfg)

    # A loss "per token" cannot be compared between tokenizers: a BPE token
    # covers several characters, so each prediction is harder but there are
    # fewer of them. Converting to bits per CHARACTER puts everything on one scale:
    #   bits/char = (nats per token) / (characters per token) / ln(2)
    _, val_ids, tokenizer, _ = load_dataset(cfg.data)
    chars_per_token = len(tokenizer.decode(val_ids.tolist())) / len(val_ids)
    summary["val_chars_per_token"] = chars_per_token
    summary["val_bits_per_char"] = summary["best_val_loss"] / chars_per_token / math.log(2)
    summary["variant"] = name
    summary["seed"] = seed
    summary["max_steps"] = cfg.train.max_steps
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def aggregate(runs: list[dict]) -> list[dict]:
    """Average the runs of each variant over seeds."""
    rows = []
    for name, (description, _) in VARIANTS.items():
        group = [r for r in runs if r["variant"] == name]
        if not group:
            continue
        bpcs = [r["val_bits_per_char"] for r in group]
        rows.append(
            {
                "variant": name,
                "description": description,
                "n_seeds": len(group),
                "n_params": group[0]["n_params"],
                "vocab_size": group[0]["vocab_size"],
                "val_loss_per_token": mean(r["best_val_loss"] for r in group),
                "train_loss_per_token": mean(r["final_train_loss"] for r in group),
                "bits_per_char": mean(bpcs),
                "bits_per_char_min": min(bpcs),
                "bits_per_char_max": max(bpcs),
                "minutes": mean(r["seconds"] for r in group) / 60,
            }
        )
    return rows


def write_markdown(rows: list[dict], path: Path, max_steps: int, seeds: list[int]) -> None:
    """Write the results table as markdown."""
    base = next((r for r in rows if r["variant"] == "baseline"), None)
    lines = [
        "# Ablation study",
        "",
        f"Every variant starts from `{BASE_CONFIG}` and changes exactly one thing. "
        f"Each run: {max_steps} steps, seed(s) {', '.join(map(str, seeds))}. "
        "Generated by `python scripts/run_ablations.py`.",
        "",
        "**How to read it.** *Val loss / token* is the usual cross-entropy in nats. It is "
        "**not comparable across tokenizers**, because a BPE token covers several characters. "
        "*Bits / char* divides the loss by the average characters per token (and converts nats "
        "to bits), so it is comparable for every row. Lower is better.",
        "",
        "| Variant | What changed | Params | Val loss / token | Bits / char | vs baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        spread = ""
        if r["n_seeds"] > 1:
            spread = f" ({r['bits_per_char_min']:.3f}-{r['bits_per_char_max']:.3f})"
        delta = "-"
        if base is not None and r is not base:
            delta = f"{(r['bits_per_char'] / base['bits_per_char'] - 1) * 100:+.1f}%"
        lines.append(
            f"| `{r['variant']}` | {r['description']} | {r['n_params'] / 1e6:.2f}M | "
            f"{r['val_loss_per_token']:.3f} | {r['bits_per_char']:.3f}{spread} | {delta} |"
        )
    lines += [
        "",
        "![Ablation results](ablations.png)",
        "",
        "Notes:",
        "",
        "- *Val loss* is the best validation loss seen during the run (the checkpoint that would be kept).",
        "- The range in parentheses is the lowest and highest value over the seeds. Two variants "
        "whose ranges overlap cannot be told apart by this experiment."
        if len(seeds) > 1 else
        "- Only one seed was run per variant, so small differences may be run-to-run noise.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(rows: list[dict], path: Path) -> None:
    """Dot plot of bits per character: one dot per variant, a line for the range over seeds.

    The differences are a few percent, so the axis is zoomed in. That is honest
    for dots (which mark a position) but would be misleading for bars (whose
    length is read as the value), which is why this is not a bar chart.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(rows) + 1.4))
    rows = rows[::-1]  # first variant at the top
    for y, r in enumerate(rows):
        ax.plot([r["bits_per_char_min"], r["bits_per_char_max"]], [y, y], color=BLUE, linewidth=2.5,
                solid_capstyle="round", alpha=0.45)
        ax.plot(r["bits_per_char"], y, "o", color=BLUE, markersize=9, markeredgecolor="white",
                markeredgewidth=2, zorder=3)
        ax.text(r["bits_per_char_max"] + 0.008, y, f"{r['bits_per_char']:.3f}", va="center",
                color=TEXT, fontsize=9)
    baseline = next((r for r in rows if r["variant"] == "baseline"), None)
    if baseline is not None:
        ax.axvline(baseline["bits_per_char"], color=MUTED, linewidth=1, linestyle=":")
    low = min(r["bits_per_char_min"] for r in rows)
    high = max(r["bits_per_char_max"] for r in rows)
    ax.set_xlim(low - 0.03, high + 0.05)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)), [r["variant"] for r in rows])
    ax.set_xlabel("validation bits per character (lower is better; note the zoomed axis)")
    ax.set_title("Ablation study: dot = mean, line = range over seeds")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=MUTED, length=0)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ablation study.")
    parser.add_argument("--only", nargs="*", choices=list(VARIANTS), help="run only these variants")
    parser.add_argument("--seeds", nargs="*", type=int, default=[1337])
    parser.add_argument("--max_steps", type=int, default=None, help="override the step count")
    parser.add_argument("--force", action="store_true", help="re-run even if a result is cached")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    fix_console_encoding()

    runs = [
        run_variant(name, seed, args.max_steps, args.force)
        for name in (args.only or list(VARIANTS))
        for seed in args.seeds
    ]
    rows = aggregate(runs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(rows, out_dir / "ablations.md", runs[0]["max_steps"], args.seeds)
    plot_results(rows, out_dir / "ablations.png")

    print(f"\n{'variant':<12} {'params':>8} {'val/token':>10} {'bits/char':>10}")
    for r in rows:
        print(f"{r['variant']:<12} {r['n_params'] / 1e6:>7.2f}M {r['val_loss_per_token']:>10.3f} "
              f"{r['bits_per_char']:>10.3f}")
    print(f"\nSaved {out_dir / 'ablations.md'}, ablations.json and ablations.png")


if __name__ == "__main__":
    main()
