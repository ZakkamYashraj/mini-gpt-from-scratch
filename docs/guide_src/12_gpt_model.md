## Chapter 12: The full GPT model and weight tying

> **Key ideas**
>
> - The GPT is the pieces of Chapters 7 to 11 in a row: embeddings, a stack of blocks, a
>   final LayerNorm, and a Linear layer called the **LM head** that turns each vector
>   into one score per vocabulary token.
> - The shape goes `(B, T)` → `(B, T, C)` → ... → `(B, T, C)` → `(B, T, V)`.
> - **Weight tying** uses one and the same matrix for the token embedding and for the LM
>   head, since both have shape `(V, C)` and do mirror-image jobs.
> - Weights are **initialised** with small random numbers so that an untrained model
>   starts near the loss of uniform guessing.
> - Almost all of the 10.7 million parameters are inside the blocks.

### 12.1 The whole model in one picture

![The complete GPT. Token IDs enter at the bottom; scores for the next token leave at the top. The dashed line marks the two places that share one weight matrix.](figures/gpt_architecture.png)

Reading from the bottom:

1. **Token IDs**, shape `(B, T)`.
2. **Embeddings** (Chapter 7): token embedding plus positional embedding, giving
   `(B, T, C)`.
3. **`n_layer` blocks** (Chapter 11), each `(B, T, C)` → `(B, T, C)`. Six of them in the
   main model.
4. **A final LayerNorm**. Because the blocks use pre-norm, the residual stream itself is
   never normalised on the way up, so it is normalised once at the end.
5. **The LM head** ("language-model head"): a Linear layer from width `C` to width `V`.
   For every position it produces `V` numbers: the logits for the next token.

The constructor builds exactly these parts, in this order:

{{code mini_gpt/model.py symbol="GPT.__init__"}}

The second half of the constructor (weight tying and initialisation) is explained in
Sections 12.4 and 12.5.

### 12.2 The forward pass

{{code mini_gpt/model.py symbol="GPT.forward"}}

Four lines compute the logits, one for each stage of the picture. The second half
computes the loss, and it is exactly the code of the bigram model (Section 8.3):
flatten `(B, T, V)` to `(B*T, V)` and call `F.cross_entropy`. As promised then, the only
difference between the bigram model and a GPT is *how the logits are computed*.

`targets` is optional. During training we pass it and get a loss. During generation
there is no right answer to compare with, so we pass only `idx` and use the logits.

Follow a batch through with real shapes (`examples/ch12_gpt_shapes.py`):

{{code examples/ch12_gpt_shapes.py section="1."}}

{{output examples/ch12_gpt_shapes.py from="1. Shapes through the model" to="logits (after LM head)"}}

The width stays at `C = 64` all the way up the stack and changes only at the very last
step, where each vector of 64 numbers becomes 65 scores, one per character.

### 12.3 What the LM head computes

The LM head is a Linear layer without bias, so it multiplies by a matrix `W` of shape
`(V, C)`. For one position with final vector $h$ (of width `C`), the logit of token $k$
is

$$\text{logit}_k = W_k \cdot h$$

the **dot product** of $h$ with row $k$ of the matrix. You know what a dot product
measures: how much two vectors point the same way. So the LM head has one vector per
token, and the model's prediction is "which tokens' vectors does my final vector $h$
point towards?" The blocks' job, seen this way, is to steer $h$ towards the vector of
the right next token.

### 12.4 Weight tying

Now compare two matrices in the model:

| | shape | job |
|---|---|---|
| token embedding table | `(V, C)` | token ID → vector (row $k$ is the vector *for* token $k$) |
| LM head weight | `(V, C)` | vector → token scores (row $k$ is the vector that *votes for* token $k$) |

Same shape, and mirror-image jobs. In both, row $k$ is "the direction in space that
means token $k$". **Weight tying** takes the obvious step and uses a single matrix for
both:

{{code mini_gpt/model.py from="if cfg.tie_weights:" to="self.lm_head.weight = self.token_emb.weight"}}

After this assignment the two modules hold the **same tensor object**. There is one set
of numbers, used in two places. During backpropagation it receives gradient from both
uses, and the two contributions add (as in Exercise 7.4).

{{output examples/ch12_gpt_shapes.py from="3. Weight tying" to="difference:"}}

Why do it?

- **Fewer parameters.** It saves `V × C` of them. For us that is only
  $65 \times 384 = 24{,}960$, a rounding error. But for GPT-2, with a vocabulary of
  50,257 and width 768, it saves 38.6 million of 124 million parameters.
- **Shared learning.** Every time a token is *predicted*, its row is updated, and that
  also improves its *input* representation, and vice versa. Rare tokens benefit most.

Does it help *our* model? This is a claim we can test instead of taking on trust, and
Chapter 15 does: with a 65-character vocabulary the tied and untied models turn out to
be indistinguishable. That is the expected result. The argument for tying is strongest
when the vocabulary is large, and ours is tiny. We keep it because it is what GPT-2
does, it costs nothing, and it matters as soon as you switch to a BPE vocabulary.

### 12.5 Initialisation

Before training, every parameter needs a starting value. The choice matters more than
you might expect.

{{code mini_gpt/model.py from="# Initialise weights with small random numbers" to="nn.init.normal_(param"}}

{{code mini_gpt/model.py symbol="GPT._init_weights"}}

Every weight matrix and embedding table is filled with random numbers drawn from a
**normal distribution** (the bell curve) with mean 0 and standard deviation 0.02. Biases
start at 0. (`self.apply(fn)` calls `fn` on every sub-module.)

**Why random?** If all the weights were equal, all the neurons of a layer would compute
the same thing and receive the same gradient, and they would stay identical forever
(Section 4.4).

**Why small?** Recall the bigram experiment of Section 8.4: large initial weights make
an untrained model *confidently wrong*, which is worse than guessing. With weights of
size 0.02 the logits start close to zero, the softmax is nearly flat, and the loss
starts near $\ln(65) = 4.17$:

{{output examples/ch12_gpt_shapes.py from="2. Loss before training" to="loss ="}}

For comparison, the same model initialised with standard deviation 1.0 starts with a
loss of about 18. Training would have to spend its first few hundred steps just undoing
that.

> **Detail you can ignore for now**
>
> The loop after `self.apply` gives a still smaller starting scale to the last Linear
> layer of every attention and feed-forward branch (the ones named `proj`), dividing by
> $\sqrt{2 \cdot \text{n\_layer}}$. Each block adds two contributions to the residual
> stream, so with many blocks the stream would otherwise grow as it goes up. This is a
> GPT-2 detail; the model trains without it, just a little less smoothly.

### 12.6 Counting the parameters

`train.py` prints the parameter count when it starts. Where are they?

{{output examples/ch12_gpt_shapes.py from="4. Parameters of configs/small.yaml" to="total"}}

Check the pieces: the token table is $65 \times 384$, the position table
$256 \times 384$, each block has 1,770,240 (Section 11.2), and the final LayerNorm has a
gain vector of 384. Nearly 99% of the model is in the six blocks, and two thirds of each
block is its feed-forward network.

The method that counts:

{{code mini_gpt/model.py symbol="GPT.num_params"}}

`p.numel()` is the number of entries in a tensor. Because of weight tying, the shared
matrix appears once in `self.parameters()`, so it is counted once.

### 12.7 The configuration that defines the shape

Every size in this chapter comes from one small dataclass:

{{code mini_gpt/config.py symbol="ModelConfig"}}

A **dataclass** is a class that mainly holds data; Python writes the `__init__` for you
from the list of fields. Creating a model is then `GPT(ModelConfig(n_layer=4, ...))`,
and the YAML files in `configs/` simply provide these values.

### Exercises

**12.1 (easy)** For the main configuration with a batch of 64 sequences of 256 tokens,
give the shape of the tensor after each of the five stages of Section 12.1.

**12.2 (medium)** Count the parameters of `configs/tiny_cpu.yaml` by hand (vocabulary
65, `n_embd` 96, `block_size` 64, 3 layers, 3 heads, no biases, tied weights). Check
your answer against what `train.py` prints.

**12.3 (medium)** GPT-2 "small" has a vocabulary of 50,257 tokens and `n_embd = 768`.
How many parameters does weight tying save? What fraction is that of its 124 million?

**12.4 (medium)** Build a small model and re-initialise all its Linear and Embedding
weights with standard deviation 1.0 instead of 0.02. What is the loss on a random batch?
Explain the number using what you know about cross-entropy.

**12.5 (hard)** Write a function `top_predictions(model, tokenizer, text, k=5)` that
returns the `k` most likely next tokens after `text`, with their probabilities. Try it
on the trained model with the contexts `"First Citi"` and `"ROMEO:\n"`. (Use
`load_model` from `generate.py`.)

### Check your understanding

1. List the stages of the forward pass, in order, with the shape after each.
2. What does the LM head do to a single final vector?
3. Which two matrices are tied, and why is that possible?
4. Why are the initial weights small random numbers and not zeros?
5. Where are most of the model's parameters?
