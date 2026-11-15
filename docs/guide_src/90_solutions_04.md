### Chapter 4

**4.1** For $(3, 1, 4)$: $1 \cdot 3 + (-2) \cdot 1 + 0.5 \cdot 4 + 1 = 4$, and
$\text{ReLU}(4) = 4$. For $(0, 2, 0)$: $0 - 4 + 0 + 1 = -3$, and $\text{ReLU}(-3) = 0$.
The second input does not match the pattern this neuron detects, so the neuron stays
silent.

**4.2** Two more steps with $\eta = 0.1$:

| step | $w$ | $L'(w)$ | new $w$ |
|---|---|---|---|
| 5 | 1.771 | -2.458 | 2.017 |
| 6 | 2.017 | -1.966 | 2.214 |

With $\eta = 1.1$, starting from $w = 0$: $L'(0) = -6$, so $w$ becomes $6.6$. Then
$L'(6.6) = 7.2$, so $w$ becomes $6.6 - 7.92 = -1.32$. Then $L'(-1.32) = -8.64$, so $w$
becomes $8.184$. The distance from the minimum goes $3 \to 3.6 \to 4.32 \to 5.18$: it
*grows* by a factor 1.2 each step. Each step points the right way but is so long that it
overshoots the minimum and lands further away on the other side. In general
$w - 3$ is multiplied by $(1 - 2\eta)$ each step, so the method converges only when
$|1 - 2\eta| < 1$, that is, when $\eta < 1$.

**4.3** Logits `[1, 1, 1]`: the softmax is $(1/3, 1/3, 1/3)$, so the loss is
$-\ln(1/3) = \ln 3 \approx 1.10$. Logits `[10, 0, 0]` with correct token 1: the
probability of token 1 is $e^0 / (e^{10} + e^0 + e^0) = 1 / 22028.5$, so the loss is
$\ln(22028.5) \approx 10.0$. Being unsure costs about 1.1; being confidently wrong costs
ten times as much. Cross-entropy punishes misplaced confidence very heavily, which
pushes a model to say "I am not sure" (a flat distribution) when it is not sure.

**4.4** Typical final losses after 5000 steps:

| `H` | learning rate | final loss |
|---|---|---|
| 2 | 0.05 | about 0.014 |
| 16 | 0.05 | about 0.0026 |
| 64 | 0.05 | `nan` |
| 64 | 0.02 | about 0.0027 |

With two hidden neurons the network can only add up two S-shaped curves. That is enough
to produce the rough shape of one sine period (up, then down on each side) but not to
match it closely, so the loss stays about five times higher. More neurons give more
pieces to build the curve from.

With `H = 64` and the original learning rate the loss explodes into `nan`. Nothing is
wrong with the code: with 64 neurons the output is a sum of four times as many terms,
so the same change to each weight moves the output much more, the gradient is larger,
and a step of 0.05 now overshoots, as in exercise 4.2. Lowering the learning rate
to 0.02 fixes it. **The right learning rate depends on the model.** After that, 64
neurons do no better than 16 here: the task is simple and 16 are already enough.

**4.5** Change `h = np.tanh(z1)` to `h = z1` in `forward`, and in the backward pass
replace `d_z1 = d_h * (1 - h**2)` by `d_z1 = d_h` (the derivative of "do nothing" is 1).
The loss now stops at about 0.20 and never improves, however long you train, and the
prediction is a **straight line** through the origin with slope about 0.3. As Section
4.3 showed, without an activation the whole network collapses into one linear map,
and the best straight-line fit to a sine wave is poor. (Use a smaller learning rate,
such as 0.01, or this version diverges.)

**4.6**

```python
import torch

X_t, Y_t = torch.tensor(X), torch.tensor(Y)
W1_t = torch.tensor(W1, requires_grad=True)
b1_t = torch.tensor(b1, requires_grad=True)
W2_t = torch.tensor(W2, requires_grad=True)
b2_t = torch.tensor(b2, requires_grad=True)

pred_t = torch.tanh(X_t @ W1_t + b1_t) @ W2_t + b2_t
loss_t = ((pred_t - Y_t) ** 2).mean()
loss_t.backward()

z1, h, pred = forward(X)                      # the NumPy version, same weights
d_pred = 2 * (pred - Y) / len(X)
d_W1 = X.T @ ((d_pred @ W2.T) * (1 - h**2))
print(np.allclose(W1_t.grad.numpy(), d_W1))   # True
```

The gradients agree to many decimal places. Autograd performed exactly the calculation
of Section 4.7; it simply did the bookkeeping for you.
