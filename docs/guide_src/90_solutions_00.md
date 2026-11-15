# Appendix A: Solutions to the exercises

Try each exercise properly before reading its solution. Being stuck for ten minutes is
not wasted time; it is the feeling of learning.

### Chapter 1

**1.1** `cat sat` has 7 characters, so there are 6 pairs (every character except the
first is a target once):

| context | target |
|---|---|
| `c` | `a` |
| `ca` | `t` |
| `cat` | (space) |
| `cat ` | `s` |
| `cat s` | `a` |
| `cat sa` | `t` |

In general a text of $n$ tokens gives $n - 1$ training examples.

**1.2** The loss is $-\ln$ of the probability given to the *true* token. True token `a`:
$-\ln(0.3) \approx 1.20$. True token `e`: $-\ln(0.6) \approx 0.51$. The model is
punished less when the truth is something it thought likely.

**1.3** Before training the model is guessing among 512 options, so the expected loss is
$\ln(512) \approx 6.24$. For the second part, solve $-\ln p = 2.08$:
$p = e^{-2.08} \approx 0.125$, which is $1/8$. A loss of 2.08 means the model is, in
effect, "hesitating between 8 equally likely options" instead of 512. (This "effective
number of options", $e^{\text{loss}}$, is called the *perplexity*.)

**1.4** The number of possible contexts is $65^{256}$. Taking logarithms,
$256 \times \log_{10}(65) \approx 256 \times 1.813 \approx 464$, so there are about
$10^{464}$ contexts. Our training file has about $10^{6}$ characters, so it contains at
most $10^6$ of them. (For comparison, the observable universe has roughly $10^{80}$
atoms.) Almost every context the model will ever be asked about has **never been seen**.
A lookup table has nothing to say about an unseen context. A model must therefore
*generalise*: treat contexts that are similar in the right ways as similar, which is
exactly what the neural network learns to do.

### Chapter 2

**2.1** (a) `(4, 3)`: the `(3,)` is added to every row. (b) **Error**: aligned on the
right, the sizes 3 and 4 meet and neither is 1. (c) `(4, 3)`: the `(4, 1)` column is
repeated across the 3 columns. (d) `(4, 5)`: inner sizes 3 and 3 match. (e) **Error**:
inner sizes are 3 and 4.

**2.2**

```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A.sum(axis=1)    # array([ 3,  7, 11])   sum of each row
A.mean(axis=0)   # array([3., 4.])       mean of each column
A.max()          # 6
```

**2.3**

```python
M / M.sum(axis=1, keepdims=True)
```

`M.sum(axis=1)` has shape `(2,)`. Aligned on the right against `(2, 3)`, the sizes 2 and
3 clash, so the division would fail. `keepdims=True` keeps the summed dimension as a
size of 1, giving shape `(2, 1)`, which broadcasts across the 3 columns. Remember this
pattern: the model's LayerNorm uses it (`keepdim=True` in PyTorch).

**2.4**

```python
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)      # optional, see below
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

z = np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
print(softmax(z).sum(axis=1))                 # [1. 1.]
```

Subtracting the row maximum does not change the answer (exercise 2.5) but it stops
`np.exp` from overflowing when the scores are large, for example `np.exp(1000)` is `inf`.

**2.5** Numerically, `np.allclose(softmax(z), softmax(z + 100))` is `True`. Why:

$$\frac{e^{z_i + 100}}{\sum_j e^{z_j + 100}} = \frac{e^{100} \, e^{z_i}}{e^{100} \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The common factor $e^{100}$ cancels. Softmax only sees differences between scores.

**2.6**

```python
import time

def matmul(A, B):
    n, k = A.shape
    k2, m = B.shape
    assert k == k2, "inner sizes must match"
    C = np.zeros((n, m))
    for i in range(n):              # every row of A
        for j in range(m):          # every column of B
            for p in range(k):      # dot product of that row and that column
                C[i, j] += A[i, p] * B[p, j]
    return C

A, B = np.random.rand(100, 100), np.random.rand(100, 100)
t = time.time(); C1 = matmul(A, B); print("loops:", time.time() - t)
t = time.time(); C2 = A @ B;        print("numpy:", time.time() - t)
print(np.allclose(C1, C2))          # True
```

The loop version does a million multiplications in Python and takes around half a
second; `@` takes well under a millisecond. That gap, a factor of a thousand or more, is
why all numerical code is written with arrays.
