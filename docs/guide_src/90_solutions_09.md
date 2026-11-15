### Chapter 9

**9.1** With identity matrices, $Q = K = V = X$.

Scores: $Q K^T / \sqrt{2}$. Each token's dot product with itself is 1 and with the other
is 0, so the scores are $\begin{pmatrix} 0.707 & 0 \\ 0 & 0.707 \end{pmatrix}$.

Weights: softmax of $(0.707, 0)$ is
$\left(\frac{e^{0.707}}{e^{0.707} + 1}, \frac{1}{e^{0.707} + 1}\right) = \left(\frac{2.028}{3.028}, \frac{1}{3.028}\right) = (0.670, 0.330)$.
So the weights are $\begin{pmatrix} 0.670 & 0.330 \\ 0.330 & 0.670 \end{pmatrix}$.

Outputs: $\text{weights} \cdot V$ with $V$ the identity, so the outputs equal the weights:
token 0 becomes $(0.670, 0.330)$ and token 1 becomes $(0.330, 0.670)$. Each token has
kept two thirds of itself and absorbed one third of the other.

**9.2** Three matrices of shape `(384, 64)`: $3 \times 384 \times 64 = 73{,}728$
parameters.

**9.3** Equal scores give equal weights, $1/T$ each, so attention reduces to the plain
average of Section 9.2 (an average of the *values*). Setting $W_q$ to all zeros does it:
every query is then the zero vector, so every dot product is 0. (So does $W_k = 0$.)
This is a useful way to see that plain averaging is a special case that attention can
always fall back on.

**9.4** Token 2 is $(1, 1)$, and the keys are $(1, 0)$, $(0, 1)$ and $(1, 1)$. For the
query $q = (a, b)$ to prefer key 0 over key 2 we need $a > a + b$, so $b$ must be
negative: the query must point *towards* the first direction and *away from* the second.
One choice:

```python
W_q = torch.tensor([[2.0, 0.0], [2.0, -4.0]])
```

Now token 2's query is $(1, 1) W_q = (4, -4)$. Its scaled scores are
$(2.83, -2.83, 0)$ and its weights are $(0.941, 0.003, 0.056)$: 94% on token 0. A
negative component in a query means "I specifically do *not* want tokens that have
this".

**9.5**

```python
torch.manual_seed(0)
W_q, W_k, W_v = torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)

def attention(X):
    scores = (X @ W_q) @ (X @ W_k).T / 2.0          # sqrt(4) = 2
    return F.softmax(scores, dim=-1) @ (X @ W_v)

X = torch.randn(5, 4)
perm = torch.tensor([3, 0, 4, 1, 2])
print(torch.allclose(attention(X)[perm], attention(X[perm]), atol=1e-6))   # True
```

Shuffling the input shuffles the output in exactly the same way, and nothing else
changes: each token's new vector is the same whichever position it sits in. Attention
by itself treats the sequence as a *set*. Order has to be supplied from outside, by the
positional embeddings.

**9.6**

```python
import numpy as np

def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def attention(X, W_q, W_k, W_v):
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    return softmax_rows(scores) @ V

X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
W_q = np.array([[2.0, 0.0], [0.0, 2.0]])
W_k = np.eye(2)
W_v = np.array([[1.0, 2.0], [3.0, 4.0]])
print(attention(X, W_q, W_k, W_v))
# [[2.5542 4.    ]
#  [3.229  4.6748]
#  [3.3457 5.0185]]
```

The same numbers as the "output WITHOUT mask" of Section 9.4.
