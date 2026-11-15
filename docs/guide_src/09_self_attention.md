## Chapter 9: Self-attention

> **Key ideas**
>
> - **Self-attention** lets every token gather information from the other tokens of the
>   sequence. It is the one place in the model where tokens interact.
> - Each token produces three vectors: a **query** (what am I looking for?), a **key**
>   (what do I contain?) and a **value** (what do I hand over?).
> - The match between token $i$'s query and token $j$'s key, measured by a dot product,
>   decides how much of token $j$'s value flows to token $i$.
> - In matrix form: $\text{weights} = \text{softmax}(QK^T / \sqrt{d})$ and
>   $\text{output} = \text{weights} \cdot V$.
> - The weights are not parameters. They are recomputed for every input. The parameters
>   are the three matrices that produce the queries, keys and values.

This is the most important chapter of the guide. Go slowly, and do the hand calculation
in Section 9.4 with a pencil. The notebook `notebooks/walkthrough.ipynb` covers the same
ground interactively and is worth running alongside.

### 9.1 What we need

After Chapter 7, each token is a vector that knows what it is and where it is. After
Chapter 8, we know the problem: those vectors do not talk to each other, so the
prediction at each position sees only one token.

We need an operation that takes the `T` vectors of a sequence and gives back `T` new
vectors, where each new vector contains information from the *other* tokens: ideally,
from the tokens that are relevant to it.

Consider predicting the next character after `The king is dea`. The token `a` at the end
would like to know: what are the letters of the word I am in? (`d`, `e`: so probably
`d` next.) Was there a subject earlier, and what was it? Which tokens are relevant
depends on the content of the text, so the operation must be flexible about where it
looks.

### 9.2 A first attempt: just average

The crudest way to let token $i$ know about the others is to replace its vector by the
average of all the vectors. Written with a matrix of weights, for three tokens:

$$\text{output} = W X \qquad\text{with}\qquad W = \begin{pmatrix} 1/3 & 1/3 & 1/3 \\ 1/3 & 1/3 & 1/3 \\ 1/3 & 1/3 & 1/3 \end{pmatrix}$$

Here $X$ has one row per token, and row $i$ of $W$ says how much of each token goes into
the new token $i$. Each row of $W$ sums to 1, so each output is a **weighted average**
of the inputs.

This does mix information, but blindly: every token gets the same summary, and nearby,
relevant tokens count no more than distant, irrelevant ones.

The key insight of attention is to keep this exact structure,
$\text{output} = W \cdot (\text{something})$ with rows of $W$ summing to 1, but to let
the model **compute the weights $W$ from the tokens themselves**. Then each token can
have its own row of weights, concentrated on whatever is relevant *to it*.

### 9.3 Queries, keys and values

How should token $i$ decide how much weight to put on token $j$? Attention uses an idea
borrowed from searching.

Imagine a library. You arrive with a **question**. Every book has a **label** on its
spine describing what it is about. You compare your question with each label, and you
take the most from the books whose labels match best. What you take away is not the
label but the book's **contents**.

In attention every token plays all three roles at once, and for each role it produces a
separate vector:

| Vector | Role in the analogy | The question it answers |
|---|---|---|
| **query** $q_i$ | your question | "what am I, token $i$, looking for?" |
| **key** $k_j$ | the label on the spine | "what do I, token $j$, contain?" |
| **value** $v_j$ | the contents of the book | "what will I hand over if I am chosen?" |

Each is obtained by multiplying the token's vector by a learned matrix. With the token
vectors as the rows of $X$:

$$Q = X W_q \qquad K = X W_k \qquad V = X W_v$$

The matrices $W_q$, $W_k$ and $W_v$ are the **parameters** of attention, and they are
the same for every token and every position. The width of the query, key and value
vectors is called the **head size**; call it $d$.

**How well does a query match a key?** By their **dot product**. Section 2.6 said that a
dot product is large and positive when two vectors point the same way. So

$$\text{score}_{ij} = q_i \cdot k_j$$

is large when what token $i$ is looking for matches what token $j$ advertises. Computing
this for every pair $(i, j)$ at once is one matrix multiplication, $Q K^T$: row $i$ of
$Q$ dotted with row $j$ of $K$ (the transpose turns the rows of $K$ into columns).

**From scores to weights.** Scores can be any numbers. We need positive weights that sum
to 1 in each row. That is exactly what softmax does (Section 2.7), applied to each row.

**From weights to output.** The new vector for token $i$ is the weighted average of the
*values*: $\text{output}_i = \sum_j \text{weight}_{ij} \, v_j$. For all tokens at once,
this is the matrix product $\text{weights} \cdot V$.

Putting it together, the complete formula of attention:

$$\text{Attention}(X) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

The division by $\sqrt{d}$ is a technical detail explained in Section 9.6. Everything
else you have now seen.

### 9.4 A worked example, by hand

Three tokens, each a vector of two numbers, and three small matrices chosen so that the
arithmetic is easy. The code is `examples/ch09_attention_by_hand.py`.

{{code examples/ch09_attention_by_hand.py from="# --- 1. Three tokens" to="V = X @ W_v"}}

**Step 1: queries, keys, values.** Multiply $X$ by each matrix. $W_q$ doubles
everything, $W_k$ changes nothing, and for the values, token 2's row is
$(1, 1) W_v = (1 \cdot 1 + 1 \cdot 3,\; 1 \cdot 2 + 1 \cdot 4) = (4, 6)$.

{{output examples/ch09_attention_by_hand.py from="Q =" to="[4., 6.]])"}}

**Step 2: scores.** Take the dot product of every query with every key, then divide by
$\sqrt{d} = \sqrt{2} \approx 1.414$. Do the bottom row yourself. Token 2's query is
$(2, 2)$:

- with key 0, $(1, 0)$: $\;2 \cdot 1 + 2 \cdot 0 = 2$, and $2 / 1.414 = 1.414$
- with key 1, $(0, 1)$: $\;2 \cdot 0 + 2 \cdot 1 = 2$, and $2 / 1.414 = 1.414$
- with key 2, $(1, 1)$: $\;2 \cdot 1 + 2 \cdot 1 = 4$, and $4 / 1.414 = 2.828$

{{output examples/ch09_attention_by_hand.py from="scores = Q @ K.T" to="2.8284]])"}}

Token 2 matches itself best (2.83) and the other two equally (1.41). Token 0's query
$(2, 0)$ gets a score of 0 with key 1, $(0, 1)$: the two are perpendicular, so the dot
product is zero. Token 0 is "not looking for" what token 1 offers.

**Step 3: softmax, row by row.** For the bottom row, $(1.414, 1.414, 2.828)$:

$$e^{1.414} = 4.113 \qquad e^{1.414} = 4.113 \qquad e^{2.828} = 16.918 \qquad \text{sum} = 25.144$$

$$\text{weights} = \left(\tfrac{4.113}{25.144},\; \tfrac{4.113}{25.144},\; \tfrac{16.918}{25.144}\right) = (0.164,\; 0.164,\; 0.673)$$

{{output examples/ch09_attention_by_hand.py from="weights WITHOUT mask" to="0.6728]])"}}

Each row sums to 1. Read row 2 as: "token 2 takes 16% of its new vector from token 0,
16% from token 1 and 67% from itself."

**Step 4: the output.** Each new vector is the weighted average of the three value
vectors $(1, 2)$, $(3, 4)$, $(4, 6)$. For token 2:

$$0.164 \cdot (1, 2) + 0.164 \cdot (3, 4) + 0.673 \cdot (4, 6) = (3.35,\; 5.02)$$

{{output examples/ch09_attention_by_hand.py from="output WITHOUT mask" to="5.0185]])"}}

That is all of attention: four small matrix operations. The left and middle panels of
the figure show the scores and the weights you just computed. (The right panel belongs
to the next chapter.)

![The worked example. Left: the scaled scores. Middle: the attention weights (softmax of each row). Right: the same weights once the causal mask of Chapter 10 is applied.](figures/attention_example.png)

### 9.5 What is learned and what is computed

A point that confuses almost everyone at first:

- The **weights** (the $3 \times 3$ table above) are **not parameters**. They are
  computed from the input, and a different input gives different weights. For a new
  sentence, each token decides afresh where to look.
- The **parameters** are $W_q$, $W_k$ and $W_v$. Training adjusts these three matrices
  so that the queries and keys they produce lead to useful weights. In our example we
  chose them by hand. In the real model they start random and are learned.

This is the source of attention's power. A fixed weight matrix, as in Section 9.2, can
express one fixed pattern of who-listens-to-whom. Three learned matrices express a
*rule* for working out who should listen to whom, which then applies to any text.

Why "self"-attention? Because the queries, keys and values all come from the same
sequence: the sequence attends to itself.

### 9.6 Why divide by the square root of the head size?

A dot product of two vectors of width $d$ is a sum of $d$ terms. With more terms the sum
tends to be larger in size: its typical magnitude grows like $\sqrt{d}$. In our model a
head has width 64, so raw scores would be around 8 times larger than with width 1.

Softmax reacts strongly to the *size* of its inputs. Feed it large numbers and it puts
nearly all the weight on the single largest one:

```python
>>> torch.softmax(torch.tensor([0.5, -0.3, 0.1, 0.8]), dim=0)
tensor([0.2882, 0.1295, 0.1932, 0.3891])
>>> torch.softmax(torch.tensor([0.5, -0.3, 0.1, 0.8]) * 8, dim=0)
tensor([0.0829, 0.0001, 0.0034, 0.9136])
```

An attention head that starts out putting 91% of its weight on one token has almost made
up its mind before learning anything. Worse, in that saturated state the gradient
flowing through the softmax is nearly zero, so the head learns slowly. Dividing the
scores by $\sqrt{d}$ cancels the growth and keeps the scores at a similar, moderate size
whatever the head width. This is why the operation is called **scaled dot-product
attention**.

### 9.7 The code

Now read the real class. It has one step you have not met yet, the mask in step 3. Skip
it for now; the next chapter is about it.

{{code mini_gpt/model.py from="class CausalSelfAttentionHead" to="return out, weights"}}

Match it against the formula:

- **`__init__`** creates $W_q$, $W_k$, $W_v$ as three `nn.Linear` layers without bias. A
  `nn.Linear(n_embd, head_size)` multiplies a vector of width `n_embd` by a learned
  matrix and returns a vector of width `head_size`.
- **Step 1** computes $Q$, $K$, $V$. The input `x` has shape `(B, T, C)`; a Linear layer
  acts on the last dimension, so each result has shape `(B, T, head_size)`.
- **Step 2** is $QK^T / \sqrt{d}$. The only new thing is the batch dimension.
  `k.transpose(-2, -1)` swaps the last two dimensions, turning `(B, T, head_size)` into
  `(B, head_size, T)`. Then `@` multiplies the last two dimensions for each sequence in
  the batch separately (Section 2.6): `(B, T, head_size) @ (B, head_size, T)` gives
  `(B, T, T)`. One $T \times T$ table of scores per sequence.
- **Step 4** is the softmax over the last dimension, that is, across each row.
- **Step 5** is $\text{weights} \cdot V$: `(B, T, T) @ (B, T, head_size)` gives
  `(B, T, head_size)`. Dropout (Section 5.2) is applied to the weights during training.

The method returns the weights as well as the output, so that we can plot them in
Chapter 16.

The last lines of the example script load our hand-chosen matrices into this class and
confirm that it reproduces the hand calculation (with the mask, as computed in the next
chapter):

{{output examples/ch09_attention_by_hand.py from="CausalSelfAttentionHead output" to="matches the hand computation"}}

> **Detail you can ignore for now**
>
> The example stores `W_q.T` in `head.query.weight`. That is only because `nn.Linear`
> keeps its matrix transposed: it computes `x @ weight.T`. It makes no difference to
> anything you need to understand.

### 9.8 Three properties worth remembering

**All positions are computed at once.** Nothing in the formula goes through the sequence
step by step. Every row of the output is produced by the same few matrix
multiplications, which a GPU performs in parallel. This is the practical reason
transformers replaced the earlier step-by-step designs (recurrent networks).

**The cost grows with the square of the length.** The score table has $T \times T$
entries. Doubling the context length quadruples the work and the memory of this step.
That is the main reason context lengths are limited.

**Attention ignores order.** The weight between tokens $i$ and $j$ depends only on their
vectors, not on how far apart they are. Without the positional embeddings of Chapter 7,
shuffling the earlier tokens would not change what a token gathers from them. Exercise
9.5 lets you see this.

### Exercises

**9.1 (easy)** Two tokens: $x_0 = (1, 0)$ and $x_1 = (0, 1)$. Let $W_q$, $W_k$ and
$W_v$ all be the identity matrix (so $Q = K = V = X$), with head size 2. Compute the
scores, the weights and the outputs by hand.

**9.2 (easy)** How many parameters does one attention head have when `n_embd` is 384,
`head_size` is 64, and there are no biases?

**9.3 (medium)** Suppose all the scores in a row are equal. What are the weights? What
does attention reduce to in that case? What single change to the parameters would make
every score equal?

**9.4 (medium)** In the worked example, change $W_q$ so that token 2 puts most of its
weight on token 0 instead of on itself. (Hint: what should token 2's query look like to
match key 0 and not key 1?) Verify with the script.

**9.5 (hard)** Show that attention without a mask ignores order. Write a function
`attention(X)` using fixed random `W_q`, `W_k`, `W_v`. Apply it to a matrix `X` of 5
random tokens, and to the same tokens in a shuffled order. How are the two outputs
related?

**9.6 (hard)** Implement `attention(X, W_q, W_k, W_v)` in NumPy only, including a
row-wise softmax, and check it against the numbers of Section 9.4.

### Check your understanding

1. In one sentence each, what are a query, a key and a value?
2. Which operation measures how well a query matches a key?
3. Are the attention weights parameters of the model? Explain.
4. What are the shapes of the scores and of the output, for input of shape `(B, T, C)`?
5. What goes wrong if the scores are not divided by $\sqrt{d}$?
