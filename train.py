"""Train a GPT on a text file.

Usage (from the repository root):

    python train.py --config configs/small.yaml
    python train.py --config configs/tiny_cpu.yaml
    python train.py --config configs/small.yaml --set train.max_steps=500 model.n_head=4

Everything the run produces goes into ``train.out_dir`` (e.g. checkpoints/small/):

    best.pt        the checkpoint with the lowest validation loss
    log.csv        step, train loss, val loss, learning rate, elapsed seconds
    samples.json   text generated at regular intervals during training
    samples.md     a readable early / middle / final comparison of those samples
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from mini_gpt.config import Config, load_config
from mini_gpt.data import get_batch, load_dataset
from mini_gpt.model import GPT
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.utils import (
    CSVLogger,
    configure_optimizer,
    fix_console_encoding,
    get_device,
    get_lr,
    save_checkpoint,
    set_seed,
)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    splits: dict[str, torch.Tensor],
    cfg: Config,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Average the loss over several random batches of each split.

    A single batch gives a noisy number; averaging ``eval_batches`` of them
    gives a much steadier estimate. ``model.eval()`` switches off dropout so we
    measure the model as it will actually be used.
    """
    model.eval()
    results: dict[str, float] = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.train.eval_batches)
        for i in range(cfg.train.eval_batches):
            x, y = get_batch(data, cfg.model.block_size, cfg.train.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            losses[i] = loss.item()
        results[name] = losses.mean().item()
    model.train()
    return results


@torch.no_grad()
def sample_text(
    model: GPT, tokenizer: Tokenizer, prompt: str, n_tokens: int, device: torch.device
) -> str:
    """Generate ``n_tokens`` tokens after ``prompt`` and return the decoded text."""
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, n_tokens, temperature=0.8, top_k=50)
    model.train()
    return tokenizer.decode(out[0].tolist())


def write_samples_markdown(samples: list[dict[str, Any]], path: Path, cfg: Config) -> None:
    """Write a markdown file showing samples from four stages of training."""
    if not samples:
        return
    last = samples[-1]["step"]
    # untrained, an early one, the one closest to the halfway point, and the final one
    wanted = {
        "Untrained (random weights)": samples[0],
        "Early": samples[min(1, len(samples) - 1)],
        "Middle": min(samples, key=lambda s: abs(s["step"] - last / 2)),
        "Final": samples[-1],
    }
    lines = [
        "# Samples during training",
        "",
        f"Config `{cfg.name}`: the same model sampled at different points of one training run "
        f"(temperature 0.8, top-k 50, prompt = a newline).",
        "",
    ]
    for stage, sample in wanted.items():
        val = f", val loss {sample['val_loss']:.3f}" if sample.get("val_loss") is not None else ""
        lines += [f"## {stage} - step {sample['step']}{val}", "", "```text", sample["text"].strip("\n"), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def train(cfg: Config, results_dir: str | None = None) -> dict[str, Any]:
    """Run one full training run described by ``cfg``.

    Returns:
        A summary dict (best validation loss, parameter count, time taken, ...).
    """
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    use_amp = cfg.train.amp and device.type == "cuda"  # mixed precision only makes sense on GPU
    out_dir = Path(cfg.train.out_dir)

    # ---- data ------------------------------------------------------------- #
    train_ids, val_ids, tokenizer, meta = load_dataset(cfg.data)
    cfg.model.vocab_size = tokenizer.vocab_size
    splits = {"train": train_ids, "val": val_ids}

    # ---- model ------------------------------------------------------------ #
    model = GPT(cfg.model).to(device)
    n_params = model.num_params()
    print(f"Run '{cfg.name}' on {device} (mixed precision: {use_amp})")
    print(f"Model: {cfg.model.n_layer} layers, {cfg.model.n_head} heads, n_embd={cfg.model.n_embd}, "
          f"block_size={cfg.model.block_size}, vocab_size={cfg.model.vocab_size}")
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # ---- optimizer -------------------------------------------------------- #
    optimizer = configure_optimizer(
        model, cfg.train.learning_rate, cfg.train.weight_decay, (cfg.train.beta1, cfg.train.beta2)
    )
    # The GradScaler multiplies the loss by a large factor before backward() so
    # that tiny float16 gradients do not round to zero, then divides it back out.
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    logger = CSVLogger(out_dir / "log.csv", ["step", "train_loss", "val_loss", "lr", "seconds"])
    samples: list[dict[str, Any]] = []
    best_val = float("inf")
    last_losses: dict[str, float] = {}
    start = time.time()

    # ---- the training loop ------------------------------------------------ #
    for step in range(cfg.train.max_steps + 1):
        # Set the learning rate for this step (warmup, then cosine decay).
        lr = get_lr(step, cfg.train.learning_rate, cfg.train.min_lr,
                    cfg.train.warmup_steps, cfg.train.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Every so often: measure train/val loss, log it, keep the best model.
        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps:
            last_losses = estimate_loss(model, splits, cfg, device, use_amp)
            elapsed = time.time() - start
            print(f"step {step:5d}/{cfg.train.max_steps} | train loss {last_losses['train']:.4f} | "
                  f"val loss {last_losses['val']:.4f} | lr {lr:.2e} | {elapsed:6.0f}s")
            logger.log({"step": step, "train_loss": f"{last_losses['train']:.5f}",
                        "val_loss": f"{last_losses['val']:.5f}", "lr": f"{lr:.6e}",
                        "seconds": f"{elapsed:.1f}"})
            if last_losses["val"] < best_val:
                best_val = last_losses["val"]
                save_checkpoint(
                    out_dir / "best.pt",
                    model_state=model.state_dict(),
                    config=cfg.to_dict(),
                    tokenizer=tokenizer.to_dict(),
                    step=step,
                    val_loss=best_val,
                )

        # Every so often: let the model write some text so we can watch it learn.
        if step % cfg.train.sample_interval == 0 or step == cfg.train.max_steps:
            text = sample_text(model, tokenizer, cfg.train.sample_prompt, cfg.train.sample_tokens, device)
            samples.append({"step": step, "val_loss": last_losses.get("val"), "text": text})
            print(f"----- sample at step {step} -----\n{text}\n" + "-" * 32)

        if step == cfg.train.max_steps:
            break  # the final evaluation is done; no more updates

        # 1) Get a batch of inputs x and targets y (y is x shifted by one token).
        x, y = get_batch(train_ids, cfg.model.block_size, cfg.train.batch_size, device)

        # 2) Forward pass: compute the loss. autocast runs most operations in
        #    float16 on the GPU (about half the memory, noticeably faster).
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)

        # 3) Backward pass: compute the gradient of the loss w.r.t. every parameter.
        optimizer.zero_grad(set_to_none=True)  # clear the gradients of the previous step
        scaler.scale(loss).backward()

        # 4) Gradient clipping: if the gradient is very long, shrink it to
        #    length grad_clip. This prevents one bad batch from wrecking the weights.
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)  # undo the AMP scaling before measuring the norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

        # 5) Update: AdamW moves every parameter a little, against its gradient.
        scaler.step(optimizer)
        scaler.update()

    seconds = time.time() - start
    print(f"Done in {seconds / 60:.1f} min. Best val loss {best_val:.4f}. "
          f"Checkpoint: {out_dir / 'best.pt'}")

    # ---- save the samples ------------------------------------------------- #
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    write_samples_markdown(samples, out_dir / "samples.md", cfg)
    if results_dir is not None:
        write_samples_markdown(samples, Path(results_dir) / "samples.md", cfg)

    return {
        "name": cfg.name,
        "n_params": n_params,
        "best_val_loss": best_val,
        "final_train_loss": last_losses["train"],
        "final_val_loss": last_losses["val"],
        "vocab_size": tokenizer.vocab_size,
        "chars_per_token": meta["chars_per_token"],
        "seconds": seconds,
        "device": str(device),
    }


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train a mini GPT.")
    parser.add_argument("--config", default="configs/small.yaml", help="path to a YAML config")
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                        help="override config values, e.g. --set train.max_steps=500")
    parser.add_argument("--results_dir", default=None,
                        help="also write samples.md into this folder (e.g. results)")
    args = parser.parse_args()
    fix_console_encoding()

    cfg = load_config(args.config, args.set)
    train(cfg, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
