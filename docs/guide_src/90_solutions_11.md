### Chapter 11

**11.1** `n_embd` must be divisible by `n_head`. Allowed: 4 (head size 96), 6 (64),
8 (48), 12 (32). Not allowed: 5, because $384 / 5$ is not a whole number. The
constructor of `MultiHeadAttention` raises a `ValueError` with a message saying so.

**11.2** Mean: $(1 + 1 + 3 + 3)/4 = 2$. Deviations: $(-1, -1, 1, 1)$. Variance: the
average of their squares, $1$. Standard deviation: $1$. Result: $(-1, -1, 1, 1)$.

**11.3** With `n_embd = 384`, 6 heads of size 64, and no biases:

- Attention: each head has three matrices of shape `(384, 64)`, so
  $6 \times 3 \times 384 \times 64 = 442{,}368$. The projection is `(384, 384)`:
  $147{,}456$. Total $589{,}824$.
- Feed-forward: `(384, 1536)` and `(1536, 384)`: $2 \times 384 \times 1536 = 1{,}179{,}648$.
- LayerNorm: one gain vector of 384 each, two of them: $768$.

Sum: $1{,}770{,}240$ per block, and six blocks hold $10{,}621{,}440$ of the model's
$10{,}745{,}088$ parameters.

**11.4** Measured gradient at the first layer:

| layers | without residual | with residual |
|---|---|---|
| 5 | 3.6e-02 | 1.2 |
| 30 | 7.6e-08 | 4.4 |
| 100 | 3.4e-26 | 45 |

Without residual connections the gradient shrinks by a roughly constant factor per
layer, so it falls off exponentially with depth; at 100 layers it is zero for all
practical purposes. With them it stays at a usable size at every depth.

**11.5**

```python
torch.manual_seed(0)
ffwd = FeedForward(8)
x = torch.randn(1, 5, 8)
a = ffwd(x)
x2 = x.clone(); x2[0, 2] += 1
b = ffwd(x2)
print([(a[0, i] != b[0, i]).any().item() for i in range(5)])
# [False, False, True, False, False]
```

Only position 2 changed. The feed-forward network never mixes positions; only attention
does.

**11.6** Replace the body of `Block.forward` by

```python
attn_out, weights = self.attn(x)
x = self.ln1(x + attn_out)
x = self.ln2(x + self.ffwd(x))
return x, weights
```

In one run of the tiny configuration for 1000 steps, the validation loss was 1.96 with
pre-norm and 2.05 with post-norm. That is a single seed on a 3-layer model, so treat it
as an indication only; the advantage of pre-norm grows with depth, where post-norm
models often need a careful warmup to train at all.
