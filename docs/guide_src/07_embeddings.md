## Chapter 7: Embeddings and positional embeddings

> **Key ideas**
>
> - A token ID is just a label. Before the model can compute with a token, the ID is
>   swapped for a vector of numbers, its **embedding**.
> - An **embedding table** has one row per token. Looking up an ID returns its row. The
>   rows are parameters: they are learned.
> - Token embeddings say *what* a token is but not *where* it is. A second table, the
>   **positional embedding**, has one row per position and is **added** to the token
>   embedding.
> - After this step, a batch of token IDs of shape `(B, T)` has become a tensor of shape
>   `(B, T, C)`: the format every later layer works with.

### 7.1 Why not feed the IDs straight in?

The tokenizer gives us integers such as `[18, 47, 56, 57]`. Could the network do
arithmetic on these directly?

No, because the numbers are arbitrary labels. With our character tokenizer `a` is 39 and
`z` is 64, but `z` is not "more" than `a` in any useful sense, and the fact that 40
(`b`) is close to 39 (`a`) means nothing. A neural network multiplies its inputs by
weights, so it would treat these meaningless sizes and distances as meaningful.

What we want is to give each token a list of numbers that the model itself can shape
during training: a point in a space with many dimensions, where the model is free to
place related tokens near each other. That list is the token's **embedding**.

### 7.2 An embedding is a lookup table

The **embedding table** is a matrix with `V` rows (one per token in the vocabulary) and
`C` columns. `C` is the **embedding dimension**, called `n_embd` in the code; it is 384
in our main configuration. To embed a token, take its row.

![An embedding lookup. Each ID selects one row of the table. The same ID always returns the same row.](figures/embedding_lookup.png)

In PyTorch this table is `nn.Embedding`. The example file is
`examples/ch07_embeddings.py`:

{{code examples/ch07_embeddings.py section="1."}}

{{output examples/ch07_embeddings.py from="1. Lookup" to="row 0 of the output"}}

Three IDs went in, three rows of four numbers came out: shape `(3,)` became `(3, 4)`.
The ID 3 appears twice and returns the same row both times.

The table starts as random numbers. It is an ordinary parameter, so `backward()`
computes a gradient for it and the optimizer adjusts it like any other weight. Only the
rows that were actually looked up in a batch receive a gradient from that batch.

> **Note**
>
> A lookup is the same thing as a matrix multiplication in disguise. Represent token 3
> as a **one-hot vector**: all zeros except for a 1 in position 3. Multiplying that
> vector by the table picks out row 3 exactly, because every other row is multiplied
> by 0. The example file checks this:
>
> ```text
> one_hot @ table gives the same result: True
> ```
>
> So an embedding layer *is* a linear layer whose input happens to be one-hot. The
> lookup is simply a much faster way of computing it. Chapter 12 uses this fact to
> explain weight tying.

In the model, the token embedding is one line of `GPT.__init__`:

{{code mini_gpt/model.py from="# Token embedding: a table" to="self.token_emb ="}}

### 7.3 What do the learned vectors mean?

Nobody decides what the 384 numbers of a token mean. They end up being whatever helps
predict the next token. A useful consequence is that tokens which behave similarly in
text end up with similar vectors, because the model benefits from treating them
alike.

We can check this on the trained model. For a few characters, the script finds the
characters whose embedding vectors point in the most similar direction (measured by the
**cosine similarity**: the dot product of the two vectors after scaling both to
length 1).

{{output examples/ch07_embeddings.py from="6. Nearest neighbours" to="closest to '.'"}}

<!-- RESULTS-EMBEDDINGS -->

### 7.4 The problem: no sense of order

Token embeddings have a blind spot. The embedding of `g` is the same vector whether the
`g` is the first character of the text or the hundredth.

Why does that matter? The attention mechanism of Chapter 9 gathers information from
other tokens by taking a *weighted sum* of their vectors. And a sum does not care about
order:

{{code examples/ch07_embeddings.py section="3."}}

{{output examples/ch07_embeddings.py from="3. Order" to="order 2-1-0"}}

Identical. If the tokens were the words "dog bites man" and "man bites dog", the sum
could not tell the two sentences apart. A model built only on token embeddings would
treat a text as an unordered bag of tokens. For language, where order carries most of
the meaning, that is fatal.

### 7.5 The fix: positional embeddings

The fix is disarmingly simple. Create a **second** embedding table with one row for
each *position*: row 0 for the first token of the sequence, row 1 for the second, up to
row `block_size - 1`. Then **add** the position's row to the token's row.

$$x_t = \text{token\_emb}[\text{id}_t] + \text{pos\_emb}[t]$$

{{code examples/ch07_embeddings.py section="4."}}

{{output examples/ch07_embeddings.py from="4. Positions" to="different vector"}}

The same token now has a different vector at each position, so "where am I?" is written
into every vector the rest of the network sees. Like the token table, the position table
starts random and is learned: the model discovers for itself what it is useful to know
about positions.

Here is the method of the model that does both lookups and the addition:

{{code mini_gpt/model.py symbol="GPT.embed"}}

Follow the shapes for a batch of `B` sequences of length `T`:

- `idx` has shape `(B, T)`: integers.
- `self.token_emb(idx)` has shape `(B, T, C)`: each integer was replaced by a row of `C`
  numbers.
- `torch.arange(T)` is the tensor `[0, 1, ..., T-1]`, so `self.pos_emb(positions)` has
  shape `(T, C)`.
- The addition is `(B, T, C) + (T, C)`. By the broadcasting rule of Section 2.5 (align
  on the right; a missing size counts as 1), the positional rows are repeated for every
  sequence in the batch. This is what we want: position 5 means the same thing in every
  sequence.

{{output examples/ch07_embeddings.py from="5. Shapes with a batch" to="token embeddings:"}}

The `self.drop(x)` at the end applies dropout (Section 5.2). The `if` exists so that the
positional embedding can be switched off for the ablation experiment of Chapter 15,
which measures how much it matters.

### 7.6 Questions you might be asking

**Why add the two vectors instead of joining them end to end?** Joining
(concatenating) would also work, but it would make the vectors wider. Adding keeps the
width at `C`. It sounds as if adding would scramble the two kinds of information
together, but 384 dimensions is a lot of room: the model learns to keep "what" and
"where" in different directions of the space, and later layers can read either one.

**What happens beyond `block_size`?** The position table has exactly `block_size` rows.
There is no row for position 256 when `block_size` is 256, which is one of the two
reasons the model cannot process longer sequences (the other is the attention mask of
Chapter 10). The `if T > self.cfg.block_size` check turns this into a clear error
message instead of a confusing crash.

> **Detail you can ignore for now**
>
> The original transformer paper did not learn its positional vectors. It computed them
> with fixed sine and cosine formulas. GPT-2 switched to learned tables like ours, which
> is simpler. Recent models use other schemes again (for example "rotary" embeddings).
> The idea that matters is the same in all of them: position information must be put in
> somehow, because attention alone cannot see it.

What did our model learn about positions? The figure compares every pair of rows of the
trained position table.

![Cosine similarity between the learned positional embeddings of every pair of positions, in the trained model. Blue means the two vectors point the same way; red means opposite ways.](figures/pos_emb_similarity.png)

<!-- RESULTS-POSEMB -->

### Exercises

**7.1 (easy)** `idx` has shape `(64, 256)` and `n_embd` is 384. What are the shapes of
`token_emb(idx)`, of `pos_emb(torch.arange(256))`, and of their sum?

**7.2 (easy)** How many parameters are in the token embedding table and in the positional
embedding table of the main configuration (vocabulary 65, `n_embd` 384, `block_size`
256)? Which is bigger, and does that surprise you?

**7.3 (medium)** Without looking at the example file, write code that builds a one-hot
matrix for the IDs `[2, 0, 2, 1]` with a vocabulary of 3 and shows that multiplying it by
an embedding table equals the lookup.

**7.4 (medium)** Create `nn.Embedding(10, 4)`, look up the IDs `[1, 1, 7]`, sum the
result to a single number, and call `backward()`. Inspect `emb.weight.grad`. Which rows
are non-zero, and why is row 1 different from row 7?

**7.5 (hard)** Implement the fixed sinusoidal positional encoding of the original
transformer paper: for position $p$ and dimension index $i$,
$PE(p, 2i) = \sin(p / 10000^{2i/C})$ and $PE(p, 2i+1) = \cos(p / 10000^{2i/C})$. Produce
the table for `block_size = 256`, `C = 384`, and plot the cosine similarity between all
pairs of positions as in the figure above. How does it compare with the learned one?

### Check your understanding

1. Why can the model not work directly with token IDs as numbers?
2. What are the shape and the meaning of the token embedding table?
3. What goes wrong without positional embeddings?
4. In `x + self.pos_emb(positions)`, the shapes are `(B, T, C)` and `(T, C)`. Why is
   that allowed?
5. Are the embedding tables fixed or learned?
