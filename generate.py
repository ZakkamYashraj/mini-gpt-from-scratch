"""Generate text from a trained checkpoint.

Usage (from the repository root):

    python generate.py --prompt "ROMEO:"
    python generate.py --checkpoint checkpoints/small/best.pt --prompt "To be" \
        --max_new_tokens 500 --temperature 0.8 --top_k 50 --top_p 0.95 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_gpt.config import Config
from mini_gpt.model import GPT
from mini_gpt.tokenizer import Tokenizer, tokenizer_from_dict
from mini_gpt.utils import fix_console_encoding, get_device, load_checkpoint, set_seed


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[GPT, Tokenizer, Config]:
    """Rebuild the model and tokenizer stored in a checkpoint.

    The checkpoint contains the config (so we know the model's shape), the
    tokenizer (so we can turn text into IDs exactly as during training) and
    the learned weights.
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = Config.from_dict(checkpoint["config"])
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    model = GPT(cfg.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # switch off dropout
    return model, tokenizer, cfg


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate text with a trained mini GPT.")
    parser.add_argument("--checkpoint", default="checkpoints/small/best.pt")
    parser.add_argument("--prompt", default="\n", help="text the model continues from")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="<1 = safer, >1 = more random, 0 = always the most likely token")
    parser.add_argument("--top_k", type=int, default=None, help="sample only from the k best tokens")
    parser.add_argument("--top_p", type=float, default=None,
                        help="sample only from the top tokens covering this much probability")
    parser.add_argument("--seed", type=int, default=None, help="set for repeatable output")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    fix_console_encoding()

    if args.seed is not None:
        set_seed(args.seed)
    device = get_device(args.device)
    model, tokenizer, cfg = load_model(args.checkpoint, device)
    print(f"Loaded '{cfg.name}' ({model.num_params():,} parameters) on {device}\n")

    # Text -> token IDs, with a batch dimension of 1 in front: shape (1, T).
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
