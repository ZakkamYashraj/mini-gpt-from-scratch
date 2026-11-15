## Chapter 2: NumPy basics

> **Key ideas**
>
> - An **array** is a grid of numbers. Its **shape** says how many numbers there are
>   along each direction (each **dimension** or **axis**).
> - Arithmetic on arrays is **element-wise** and needs no loops, which is both shorter to
>   write and far faster to run.
> - **Broadcasting** lets arrays of different shapes be combined by (virtually) repeating
>   the smaller one.
> - **Matrix multiplication** (`@`) combines rows of one array with columns of another.
>   The shape rule `(n, k) @ (k, m) -> (n, m)` will be used on every page of this guide.
> - **Softmax** turns any list of numbers into probabilities.

### 2.1 Why arrays?

A neural network is, underneath, a very large amount of arithmetic on grids of numbers.
Python lists are too slow for this: adding two lists of a million numbers with a `for`
loop takes around a tenth of a second, while the same addition on arrays takes about a
millisecond, because the loop runs in optimised machine code instead of in Python.

**NumPy** is the standard Python library for arrays. We use it directly in a few places
(storing token IDs, Chapter 4's from-scratch network), but the main reason to learn it is
that PyTorch, the library the model is written in, copies NumPy's ideas almost exactly.
Everything in this chapter transfers unchanged to Chapter 3.

All the code of this chapter is in `examples/ch02_numpy_basics.py`. Run it with
`python examples/ch02_numpy_basics.py`, and better still, retype it line by line in a
Python prompt.

### 2.2 Arrays and shapes

{{code examples/ch02_numpy_basics.py from="import numpy" to="M[:, 1]"}}

{{output examples/ch02_numpy_basics.py from="1. Arrays and shapes" to="the whole second column"}}

Three words to learn here:

- The **shape** of an array is a tuple with its size along each dimension. `v` has shape
  `(3,)`: one dimension, of length 3. (The comma is just how Python writes a tuple with
  one element.) `M` has shape `(2, 3)`: 2 rows, 3 columns.
- `ndim` is the number of dimensions: 1 for a vector, 2 for a matrix. Arrays can have
  more. In this project the most common array has **three** dimensions, with shape
  `(B, T, C)`. Do not try to picture it as a cube; picture it as "a list of `B` matrices,
  each with `T` rows and `C` columns".
- **Indexing** counts from 0, as with lists. `M[1, 2]` is row 1, column 2. A colon means
  "everything along this dimension", so `M[:, 1]` is column 1 of every row.

> **Note**
>
> The habit that prevents most bugs in this field: **know the shape of every array you
> touch**. Whenever you are unsure, `print(x.shape)`. Experienced people do this
> constantly, and the code in this repository has a shape comment on nearly every line.

### 2.3 Element-wise arithmetic

{{code examples/ch02_numpy_basics.py section="2."}}

{{output examples/ch02_numpy_basics.py from="2. Element-wise arithmetic" to="add ACROSS"}}

`v + 10` adds 10 to every element; `v * v` multiplies matching elements (it is **not**
matrix multiplication). Functions such as `np.exp` are applied to each element.

The `axis` argument causes more confusion than anything else in NumPy, so go slowly.
`M.sum(axis=0)` means "add along dimension 0", that is, *down the rows*. Dimension 0
disappears from the shape: `(2, 3)` becomes `(3,)`, one total per column.
`M.sum(axis=1)` adds along dimension 1, *across the columns*, and `(2, 3)` becomes
`(2,)`, one total per row.

The reliable way to think about it: **the axis you name is the axis that vanishes.**

### 2.4 Reshaping

{{code examples/ch02_numpy_basics.py section="3."}}

{{output examples/ch02_numpy_basics.py from="3. Reshaping" to="[3. 6.]]"}}

`reshape` rearranges the same numbers into a different grid. The total count must stay
the same (6 numbers can become `(2, 3)`, `(3, 2)`, `(6,)` or `(1, 6)`, but not `(4, 2)`).
Writing `-1` for one size tells NumPy to work that size out. The **transpose** `M.T`
swaps rows and columns.

We will need reshaping once in the model: to hand a `(B, T, V)` array to a function
that wants a 2-dimensional `(B*T, V)` array.

### 2.5 Broadcasting

What should `M + row` mean when `M` has shape `(2, 3)` and `row` has shape `(3,)`?
NumPy's answer: add `row` to *every row* of `M`. Stretching the smaller array to fit the
larger one is called **broadcasting**.

{{code examples/ch02_numpy_basics.py section="4."}}

{{output examples/ch02_numpy_basics.py from="4. Broadcasting" to="[210. 220. 230.]]"}}

![Broadcasting. The smaller array is repeated (grey copies) until the shapes match, then the addition is element-wise. No copies are really made; NumPy just behaves as if they were.](figures/broadcasting.png)

The rule NumPy follows:

1. Write the two shapes one above the other, **aligned on the right**.
2. Go through the columns. In each column the two sizes must be **equal**, or **one of
   them must be 1**, or one of them is missing (a missing size counts as 1).
3. Wherever a size is 1 (or missing), that array is repeated along that dimension.

Check the three examples against the rule:

```text
M    (2, 3)        M    (2, 3)        row     (3,)
row     (3,)       col  (2, 1)        col  (2, 1)
----------         ----------         ----------
     (2, 3)             (2, 3)             (2, 3)
```

The third one surprises people: a `(3,)` and a `(2, 1)` produce a `(2, 3)`. Both arrays
were stretched. This is useful, and it is also the source of a classic silent bug:
adding a `(3,)` to a `(3, 1)` by mistake gives a `(3, 3)` instead of an error.

Where the model uses broadcasting: adding the positional embeddings, of shape `(T, C)`,
to a batch of token embeddings, of shape `(B, T, C)`. Aligned on the right, `(T, C)`
is missing its first size, so it is repeated for every sequence in the batch.

### 2.6 Matrix multiplication

The single most important operation in deep learning. For two matrices `A` and `B`,
the entry in row $i$, column $j$ of `A @ B` is the **dot product** of row $i$ of `A`
with column $j$ of `B`: multiply matching elements and add them up.

$$(AB)_{ij} = \sum_{k} A_{ik} \, B_{kj}$$

{{code examples/ch02_numpy_basics.py section="5."}}

{{output examples/ch02_numpy_basics.py from="5. Matrix multiplication" to="A @ x ="}}

Check one entry by hand. Row 2 of `A` is `[5, 6]`; column 3 of `B` is `[0, 2]`. Their dot
product is $5 \cdot 0 + 6 \cdot 2 = 12$, which is `C[2, 3]`.

![The shape rule of matrix multiplication: the inner sizes must match and disappear; the outer sizes give the shape of the result.](figures/matmul.png)

**The shape rule:** `(n, k) @ (k, m)` gives `(n, m)`. The two inner numbers must be
equal; the result takes the two outer numbers. Say it out loud until it is automatic,
because you will use it to read every line of the model.

Two ways to think about what `A @ x` *means*, both useful later:

- **As a machine that transforms vectors.** A matrix of shape `(n, k)` turns a vector of
  length `k` into a vector of length `n`. A neural network layer is exactly this, with
  the matrix's entries being the learned parameters.
- **As many dot products at once.** A dot product measures how much two vectors point in
  the same direction (large and positive when aligned, zero when perpendicular, negative
  when opposed). Attention (Chapter 9) uses a matrix multiplication to compute the dot
  product of every token with every other token in one step.

> **Detail you can ignore for now**
>
> When arrays have more than two dimensions, `@` multiplies the **last two** dimensions
> and treats the others as a batch: `(B, T, k) @ (B, k, m)` gives `(B, T, m)`, as if you
> had looped over `B` and multiplied each pair of matrices. You will see this in the
> attention code.

### 2.7 Softmax

A model's raw outputs are arbitrary numbers, positive or negative, called **logits**.
To interpret them as probabilities we need numbers that are positive and add up to 1.
**Softmax** does the conversion in two steps:

1. apply $e^x$ to every number, which makes everything positive and keeps the order;
2. divide by the total, which makes them sum to 1.

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

{{code examples/ch02_numpy_basics.py section="6."}}

{{output examples/ch02_numpy_basics.py from="6. Softmax" to="probabilities ="}}

The largest score gets the largest probability, but the others do not drop to zero: this
"soft" version of picking the maximum is where the name comes from. Two properties to
remember:

- Only **differences** between scores matter. Adding 100 to every score changes nothing
  (try it in exercise 2.5).
- Because of the exponential, a score that is larger by 1 gets a probability about 2.7
  times larger ($e \approx 2.718$). A score of $-\infty$ gets probability exactly 0,
  since $e^{-\infty} = 0$. Chapter 10 uses this to hide the future from the model.

### Exercises

**2.1 (easy)** Without running anything, give the shape of each result, or say that it is
an error: (a) `np.ones((4, 3)) + np.ones((3,))`; (b) `np.ones((4, 3)) + np.ones((4,))`;
(c) `np.ones((4, 3)) + np.ones((4, 1))`; (d) `np.ones((4, 3)) @ np.ones((3, 5))`;
(e) `np.ones((4, 3)) @ np.ones((4, 3))`. Then check with NumPy.

**2.2 (easy)** Create the array `[[1, 2], [3, 4], [5, 6]]`. Using one line each, compute
the sum of each row, the mean of each column, and the largest element.

**2.3 (medium)** Each row of `M = np.array([[1., 2., 3.], [4., 5., 6.]])` should be
divided by its own sum, so that every row sums to 1. Write this in one line with
broadcasting. (Hint: look up the `keepdims=True` argument of `sum`, and think about why
you need it.)

**2.4 (medium)** Write a function `softmax(z)` that works on a 2-dimensional array and
applies softmax to **each row** separately. Test that every row of the result sums to 1.

**2.5 (medium)** Verify numerically that `softmax(z)` and `softmax(z + 100)` are equal.
Then explain *why*, starting from the formula.

**2.6 (hard)** Write your own `matmul(A, B)` using three nested `for` loops and no `@`.
Check it against `A @ B` on random matrices, and time both on two `100 x 100` matrices.

### Check your understanding

1. What is the shape of `np.zeros((5, 2, 7)).sum(axis=1)`?
2. `a` has shape `(3, 1)` and `b` has shape `(1, 4)`. What is the shape of `a + b`?
3. Why is `(4, 3) @ (4, 3)` an error while `(4, 3) * (4, 3)` is fine?
4. What does a large positive dot product between two vectors tell you?
5. A softmax receives the scores `[5.0, 5.0, 5.0, 5.0]`. What does it output?
