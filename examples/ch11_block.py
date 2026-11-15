"""Learning Guide, Chapter 11: the pieces of a transformer block.

Run it from the repository root:  python examples/ch11_block.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mini_gpt.config import ModelConfig  # noqa: E402
from mini_gpt.model import Block, FeedForward, LayerNorm, MultiHeadAttention  # noqa: E402

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)

# --- 1. Multi-head attention: shapes ---
print("1. Multi-head attention")
B, T, C = 2, 5, 8
x = torch.randn(B, T, C)
attn = MultiHeadAttention(n_embd=C, n_head=4, block_size=T)
out, weights = attn(x)
print("input", tuple(x.shape), "-> output", tuple(out.shape), " attention weights", tuple(weights.shape))
print("each of the 4 heads has head_size", attn.heads[0].head_size)
print("last row of head 0:", weights[0, 0, -1].data)
print("last row of head 1:", weights[0, 1, -1].data)

# --- 2. Layer normalisation by hand ---
print("\n2. LayerNorm")
v = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
mean = v.mean(dim=-1, keepdim=True)
var = v.var(dim=-1, keepdim=True, unbiased=False)
print("mean =", mean.item(), " variance =", var.item(), " std =", round(var.sqrt().item(), 3))
print("normalised by hand:", (v - mean) / var.sqrt())
print("our LayerNorm     :", LayerNorm(4)(v).data)
big = v * 1000  # a vector a thousand times larger...
print("LayerNorm(1000*v) :", LayerNorm(4)(big).data, " <- same result: the scale is removed")

# --- 3. Why residual connections matter ---
# Stack 30 small layers and measure how much gradient reaches the FIRST layer.
print("\n3. Residual connections and gradients")


def first_layer_gradient(use_residual: bool) -> float:
    torch.manual_seed(0)
    layers = [nn.Linear(16, 16) for _ in range(30)]
    h = torch.randn(4, 16)
    for layer in layers:
        update = torch.tanh(layer(h))
        h = h + update if use_residual else update
    h.sum().backward()
    return layers[0].weight.grad.abs().mean().item()


print(f"gradient at layer 1 of 30, without residual connections: {first_layer_gradient(False):.2e}")
print(f"gradient at layer 1 of 30, with residual connections   : {first_layer_gradient(True):.2e}")

# --- 4. The feed-forward block and the whole Block ---
print("\n4. Feed-forward and Block")
ffwd = FeedForward(n_embd=C)
print("feed-forward: ", tuple(x.shape), "->", tuple(ffwd(x).shape), " hidden width", ffwd.fc.out_features)
cfg = ModelConfig(vocab_size=65, block_size=T, n_layer=1, n_head=4, n_embd=C, dropout=0.0)
block = Block(cfg)
y, _ = block(x)
print("block:        ", tuple(x.shape), "->", tuple(y.shape), " (same shape in and out)")

# --- 5. Where the parameters of one block of the main model are ---
print("\n5. Parameters of one block of configs/small.yaml (n_embd 384, 6 heads)")
big_block = Block(ModelConfig(n_embd=384, n_head=6, block_size=256))
for name, module in [("attention", big_block.attn), ("feed-forward", big_block.ffwd),
                     ("2 x LayerNorm", nn.ModuleList([big_block.ln1, big_block.ln2]))]:
    print(f"{name:<14} {sum(p.numel() for p in module.parameters()):>9,}")
print(f"{'total':<14} {sum(p.numel() for p in big_block.parameters()):>9,}")
