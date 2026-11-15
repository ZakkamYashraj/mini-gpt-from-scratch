## Chapter 11: Assembling a transformer block

> **Key ideas**
>
> - **Multi-head attention** runs several small attention heads side by side, so the
>   model can follow several kinds of relationship at once.
> - The **feed-forward** network processes each token's vector separately, after
>   attention has gathered information into it. Attention communicates; feed-forward
>   computes.
> - A **residual connection** adds a layer's output to its input, $x + f(x)$, instead of
>   replacing it. This is what allows many layers to be stacked and still trained.
> - **Layer normalisation** rescales each token vector to a standard size, which keeps
>   the numbers well behaved through a deep stack.
> - A **transformer block** is these four pieces together. Its output has the same shape
>   as its input, so blocks can be stacked like bricks.

A single attention head is the core idea, but not yet a working model. Four more
ingredients turn it into the unit that is repeated to make a GPT. None of them is as
subtle as attention. The examples are in `examples/ch11_block.py`.

### 11.1 Multi-head attention

One attention head produces one table of weights, so each token can focus in one way at
a time. But a token often needs several things at once. To continue `The king is dea`,
the final `a` might want the previous letters of the current word (to spell it) *and*
the subject of the sentence *and* whether we are inside a speech or a stage direction.

The solution is to run several heads **in parallel**, each with its own $W_q$, $W_k$,
$W_v$, and therefore each free to learn a different pattern. To keep the cost the same,
each head is made narrower: with embedding width `C` and `n_head` heads, each head has
`head_size = C / n_head`. Our main model has `C = 384` and 6 heads of width 64.

The outputs of the heads, each of shape `(B, T, head_size)`, are **concatenated** (joined
end to end along the last dimension) to give `(B, T, C)` again. A final Linear layer
then mixes the heads' results together.

{{code mini_gpt/model.py symbol="MultiHeadAttention"}}

It really is "the single head, several times": `__init__` creates a list of
`CausalSelfAttentionHead` objects, and `forward` loops over them. (`nn.ModuleList` is a
Python list that PyTorch knows about, so that the parameters inside are found.)

{{output examples/ch11_block.py from="1. Multi-head attention" to="last row of head 1"}}

Shape in, `(2, 5, 8)`; shape out, `(2, 5, 8)`. The weights have shape
`(B, n_head, T, T)`: one $T \times T$ table per head. Even with random, untrained
matrices, the two heads shown distribute their attention differently.

> **Detail you can ignore for now**
>
> Production code does not loop over heads. It computes all heads in a single 4-D
> tensor operation, which is faster. The result is mathematically identical. We keep
> the loop because it makes the structure obvious; replacing it is a good exercise once
> everything else is clear (see Chapter 21).

### 11.2 The feed-forward network

Attention *moves* information between tokens: each output is a weighted average of
values. It does not do much *thinking* about what was gathered. That is the job of the
**feed-forward** network (also called the MLP): a small two-layer network of the kind
you built in Chapter 4, applied to each token's vector separately.

{{code mini_gpt/model.py symbol="FeedForward"}}

It widens the vector to four times its size, applies the GELU activation (Section 4.3),
and narrows it back. The same weights are used at every position, and positions do not
interact here at all: token 3's vector is transformed without any reference to token 4.

A useful way to hold the two halves in your head:

- **attention = communication**: tokens exchange information;
- **feed-forward = computation**: each token digests what it received.

Most of the model's parameters are in these feed-forward layers, as the count for one
block of the main model shows:

{{output examples/ch11_block.py from="5. Parameters of one block" to="total"}}

### 11.3 Residual connections

Suppose we simply stacked layers, each feeding the next: $x \to f_1(x) \to f_2(f_1(x))
\to \dots$ With many layers this is very hard to train, for a reason you can understand
from the chain rule.

The gradient that reaches the first layer is a product of one factor per layer on the
way back. If those factors are typically a bit smaller than 1, the product of thirty of
them is practically zero. The early layers receive no usable signal and do not learn.
This is the **vanishing gradient** problem, and for years it kept neural networks
shallow.

A **residual connection** (or skip connection) changes each layer from "replace" to
"add":

$$x \leftarrow x + f(x)$$

The layer's output is added to its input. Consider what this does to the derivative:

$$\frac{\partial}{\partial x}\bigl(x + f(x)\bigr) = 1 + f'(x)$$

That "$1$" is the point. However small $f'(x)$ is, the factor contributed by this layer
is close to 1, not close to 0. The gradient has a direct path, straight through all the
`+` operations, from the loss back to the earliest layers.

The experiment below stacks 30 small layers and measures how much gradient arrives at
the first one, with and without the additions:

{{code examples/ch11_block.py from="def first_layer_gradient" to="return layers[0]"}}

{{output examples/ch11_block.py from="3. Residual connections and gradients" to="with residual connections"}}

Without residual connections the gradient at the first layer is tens of millions of
times smaller than with them. That layer would effectively never
learn.

There is also a helpful way to *think* about a network built this way. Picture the
token's vector as a stream flowing upward through the model (people call it the
**residual stream**). Each layer reads the stream, computes something, and **adds** its
contribution back. No layer overwrites what earlier layers wrote. A layer that has
nothing useful to add can output nearly zero and leave the stream untouched, which is
also why it is easy to start training from "every layer does almost nothing".

### 11.4 Layer normalisation

As vectors pass through many layers, their overall size can drift, growing in some
layers and shrinking in others. Layers then receive inputs at unpredictable scales,
which makes training unstable.

**Layer normalisation** fixes the scale. For each token vector separately it:

1. subtracts the mean of the vector's entries, so that they average 0;
2. divides by their standard deviation, so that their variance is 1;
3. multiplies by a learned **gain** and (optionally) adds a learned **bias**, so the
   model can still choose a different scale if that is useful.

$$\hat{x} = \frac{x - \text{mean}(x)}{\sqrt{\text{var}(x) + \epsilon}} \qquad \text{out} = \text{gain} \cdot \hat{x} + \text{bias}$$

**A worked example.** Take the vector $(2, 4, 6, 8)$. Its mean is $5$. The deviations
from the mean are $(-3, -1, 1, 3)$; their squares average to
$(9 + 1 + 1 + 9)/4 = 5$, which is the variance; the standard deviation is
$\sqrt{5} = 2.236$. Dividing: $(-1.342, -0.447, 0.447, 1.342)$.

{{output examples/ch11_block.py from="2. LayerNorm" to="LayerNorm(1000*v)"}}

The last line is the property that matters: a vector a thousand times larger gives the
same result. Whatever happened to the scale in earlier layers is erased.

The repository implements this by hand, in a few lines:

{{code mini_gpt/model.py symbol="LayerNorm"}}

`dim=-1` means "along the last dimension", the `C` numbers of one token, and
`keepdim=True` keeps that dimension with size 1 so that the subtraction and division
broadcast correctly (the pattern from Exercise 2.3). The gain starts at 1 and the bias
at 0, so at the start the layer is a pure normalisation. The tiny constant `eps`
prevents division by zero for a vector whose entries are all equal.

The mean and variance are taken over the entries of a single token's vector. Nothing is
mixed between different tokens or between different sequences of the batch, so layer
normalisation cannot leak information from the future.

> **Note**
>
> PyTorch has this built in as `nn.LayerNorm`. Ours exists so that you can see there is
> nothing hidden in it, and `tests/test_attention.py` checks that the two give the same
> numbers.

### 11.5 The block

Now assemble the pieces.

![One transformer block. The token vectors flow upward. Each sub-layer (attention, then feed-forward) reads a normalised copy of the stream and adds its result back.](figures/transformer_block.png)

{{code mini_gpt/model.py symbol="Block"}}

The `forward` method is three lines, and each piece of this chapter is visible in it:

- `self.ln1(x)`: normalise, then
- `self.attn(...)`: let the tokens communicate, then
- `x = x + attn_out`: add the result to the stream (residual connection);
- `self.ln2(x)`, `self.ffwd(...)`, `x = x + ...`: the same pattern for the feed-forward
  network.

Applying the normalisation *before* each sub-layer, inside the residual branch, is
called **pre-norm**. The original 2017 transformer normalised *after* the addition
("post-norm"). Pre-norm, used from GPT-2 onwards, leaves the residual stream itself
untouched, which makes deep models noticeably easier to train.

The most important property of a block is almost boring:

{{output examples/ch11_block.py from="4. Feed-forward and Block" to="same shape in and out"}}

**The output has the same shape as the input**: `(B, T, C)` in, `(B, T, C)` out. So a
block can be followed by another block, and another. The GPT of the next chapter is
exactly that: a stack.

### 11.6 What stacking buys

Why would more blocks help? In the first block, each token can gather information only
from the raw embeddings of earlier tokens. In the second block, it gathers from vectors
that have *already* been enriched by the first block. A token can therefore attend to
"the token that the first block marked as the start of this word", or "the name that
the first block attached to this speech". Each block builds on the descriptions written
by the blocks below it, so depth allows patterns of patterns.

Dropout (Section 5.2) appears in three places: on the attention weights, after the
attention projection, and after the feed-forward network. Its only purpose is to reduce
overfitting, and it is switched off whenever the model is evaluated or generates text.

### Exercises

**11.1 (easy)** `n_embd` is 384. Which of these values of `n_head` are allowed: 4, 5, 6,
8, 12? What is the head size in each allowed case? What does the code do for a
forbidden value?

**11.2 (easy)** Apply layer normalisation by hand (gain 1, bias 0, ignore `eps`) to the
vector $(1, 1, 3, 3)$.

**11.3 (medium)** Reproduce the parameter counts of one block (attention 589,824,
feed-forward 1,179,648, LayerNorm 768) by hand from the shapes of the weight matrices.
Remember that the configuration has `bias: false`.

**11.4 (medium)** In `examples/ch11_block.py`, change the residual experiment from 30
layers to 5 and to 100. How does the gradient at the first layer change in each case?

**11.5 (medium)** The feed-forward block treats every position independently. Verify
it: pass a tensor through a `FeedForward`, then change the vector at one position and
pass it again. Which output positions changed?

**11.6 (hard)** Write a `PostNormBlock` that computes `x = ln1(x + attn(x))` and
`x = ln2(x + ffwd(x))` instead. Train the tiny configuration with each kind of block
and compare the validation loss. (You can do this by temporarily editing `Block`.)

### Check your understanding

1. Why use six heads of width 64 instead of one head of width 384?
2. Which part of the block moves information between tokens, and which part does not?
3. What would be lost if `x = x + attn_out` were written `x = attn_out`?
4. What does layer normalisation do to a token vector, in one sentence?
5. Why is it essential that a block's output has the same shape as its input?
