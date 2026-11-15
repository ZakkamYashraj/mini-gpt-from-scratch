"""Download and tokenize the dataset.

Examples (run from the repository root):

    python data/prepare.py                                  # Tiny Shakespeare, char-level
    python data/prepare.py --tokenizer bpe --bpe_vocab_size 512
    python data/prepare.py --input data/my_notes.txt        # any UTF-8 text file
    python data/prepare.py --config configs/small.yaml      # use the settings of a config

``train.py`` calls the same code automatically, so running this script first is
optional; it is useful for checking that a new text file tokenizes correctly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `import mini_gpt` when this file is run as `python data/prepare.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_gpt.config import DataConfig, load_config  # noqa: E402
from mini_gpt.data import load_dataset, prepare_dataset  # noqa: E402
from mini_gpt.utils import fix_console_encoding  # noqa: E402


def main() -> None:
    """Parse arguments, prepare the dataset and print a short summary."""
    parser = argparse.ArgumentParser(description="Download and tokenize a text dataset.")
    parser.add_argument("--config", help="YAML config whose data section should be used")
    parser.add_argument("--input", help="path to a UTF-8 .txt file (overrides the config)")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], help="overrides the config")
    parser.add_argument("--bpe_vocab_size", type=int, help="overrides the config")
    parser.add_argument("--force", action="store_true", help="re-tokenize even if cached")
    args = parser.parse_args()
    fix_console_encoding()

    cfg = load_config(args.config).data if args.config else DataConfig()
    if args.input:
        cfg.input_file = args.input
    if args.tokenizer:
        cfg.tokenizer = args.tokenizer
    if args.bpe_vocab_size:
        cfg.bpe_vocab_size = args.bpe_vocab_size

    folder = prepare_dataset(cfg, force=args.force)
    train_ids, _, tokenizer, meta = load_dataset(cfg)

    print(f"Saved to {folder}")
    print(f"  vocab size        : {meta['vocab_size']}")
    print(f"  characters/token  : {meta['chars_per_token']:.2f}")
    preview = train_ids[:16].tolist()
    print(f"  first 16 token IDs: {preview}")
    print(f"  ... which decode to: {[tokenizer.token_to_str(i) for i in preview]}")


if __name__ == "__main__":
    main()
