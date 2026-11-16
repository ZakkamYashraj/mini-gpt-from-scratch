"""Learning Guide, Chapter 12: follow one batch through the whole GPT.

Run it from the repository root:  python examples/ch12_gpt_shapes.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`

import torch  # noqa: E402

from mini_gpt.config import ModelConfig, load_config  # noqa: E402
from mini_gpt.model import GPT  # noqa: E402

torch.manual_seed(0)

# --- 1. Shapes, stage by stage ---
print("1. Shapes through the model")
cfg = ModelConfig(vocab_size=65, block_size=32, n_layer=3, n_head=4, n_embd=64, dropout=0.0)
model = GPT(cfg)
B, T = 4, 16
idx = torch.randint(0, cfg.vocab_size, (B, T))  # a batch of random token IDs
targets = torch.randint(0, cfg.vocab_size, (B, T))

print(f"{'token IDs':<28}{tuple(idx.shape)}")
x = model.embed(idx)
print(f"{'after embeddings':<28}{tuple(x.shape)}")
for i, block in enumerate(model.blocks):
    x, _ = block(x)
    print(f"{f'after block {i + 1}':<28}{tuple(x.shape)}")
x = model.ln_f(x)
print(f"{'after final LayerNorm':<28}{tuple(x.shape)}")
logits = model.lm_head(x)
print(f"{'logits (after LM head)':<28}{tuple(logits.shape)}")

# --- 2. The loss of an untrained model ---
print("\n2. Loss before training")
_, loss = model(idx, targets)
print(f"loss = {loss.item():.3f}   uniform guessing = ln(65) = {math.log(65):.3f}")

# --- 3. Weight tying ---
print("\n3. Weight tying")
print("lm_head.weight shape:", tuple(model.lm_head.weight.shape),
      " token_emb.weight shape:", tuple(model.token_emb.weight.shape))
print("they are the same tensor:", model.lm_head.weight is model.token_emb.weight)
untied = GPT(ModelConfig(vocab_size=65, block_size=32, n_layer=3, n_head=4, n_embd=64, tie_weights=False))
print(f"parameters tied: {model.num_params():,}   untied: {untied.num_params():,}   "
      f"difference: {untied.num_params() - model.num_params():,} = 65 x 64")

# --- 4. Where the parameters of the main model are ---
print("\n4. Parameters of configs/small.yaml")
big = GPT(load_config("configs/small.yaml").model)
parts = {
    "token embedding (shared with LM head)": big.token_emb,
    "positional embedding": big.pos_emb,
    f"{len(big.blocks)} transformer blocks": big.blocks,
    "final LayerNorm": big.ln_f,
}
for name, module in parts.items():
    print(f"{name:<40}{sum(p.numel() for p in module.parameters()):>12,}")
print(f"{'total':<40}{big.num_params():>12,}")
