## Chapter 16: Looking inside: attention visualisation

> **Key ideas**
>
> - Attention weights are one of the few parts of a neural network that can be read
>   directly: for every token, they say where it looked.
> - A **heatmap** shows them as a grid. Row $i$ is the token doing the looking; column
>   $j$ is the token being looked at; the colour is the weight.
> - In our trained model the first layer contains heads with a single clear job: two of
>   them always look at the **previous character**, one at the character **two back**.
> - Deeper layers look further away and spread their attention more widely.
> - One picture of one sentence is an anecdote. Measure over many sequences before
>   claiming anything, and remember that *where a head looks* is not the same as *why
>   the model made a prediction*.

### 16.1 Why look at all?

So far we have judged the model from the outside: by its loss and by the text it writes.
Most of what happens inside is hard to interpret. A vector of 384 numbers in the
residual stream does not mean anything obvious.

The attention weights are an exception. Each row is a probability distribution over
tokens with a plain meaning: "this token took this fraction of its new information from
that token". Since Chapter 9 the attention classes have been returning their weights as
well as their output. Now we use them.

### 16.2 Getting the weights out

The model has a method for this. It is the forward pass of Chapter 12 with one
difference: it keeps the weights that `forward` throws away.

{{code mini_gpt/model.py symbol="GPT.attention_maps"}}

The result is a list with one tensor per layer, each of shape `(B, n_head, T, T)`. For
the main model and one sentence of 48 characters that is 6 tensors of shape
`(1, 6, 48, 48)`: 36 heads, each with a $48 \times 48$ table.

The script `scripts/visualize_attention.py` loads the checkpoint, encodes a sentence,
and asks for these maps:

{{code scripts/visualize_attention.py from="text = args.text.replace" to="maps = [m.float().cpu()"}}

Each head's table is then drawn with matplotlib's `imshow`, which colours a grid of
numbers:

{{code scripts/visualize_attention.py symbol="plot_layer"}}

Every heatmap uses the same colour scale, from 0 (white) to 1 (dark blue), so that
different heads can be compared at a glance. Run it with:

```bash
python scripts/visualize_attention.py
python scripts/visualize_attention.py --text "ROMEO:\nO, speak again!" --layers 0 2 5
```

### 16.3 How to read a heatmap

![Attention weights of one head of the first layer, for the sentence "First Citizen: / Before we proceed, hear me speak." Each row sums to 1.](../results/attention_closeup.png)

Take any row, say the row labelled `s` in `First`. Reading along it tells you which
earlier characters that `s` drew from. Some patterns to recognise:

- **Nothing above the diagonal.** The upper-right triangle is empty in every heatmap you
  will ever draw from this model. That is the causal mask of Chapter 10, visible to the
  eye.
- **The first row is always a single dark cell.** The first token can only attend to
  itself, so its weight there is 1.
- **Dark cells on the diagonal** mean a token attending to itself.
- **A dark line just below the diagonal** means every token attends to the token
  immediately before it. A line two cells below means "two tokens back".
- **A vertical stripe** means many different tokens all attend to the same earlier
  token: that token holds something widely useful.
- **A row that is pale all over** means the weight is spread thinly across many tokens:
  the head is averaging, not selecting.

The head in the figure is easy to read: there is a dark line exactly **two cells below
the diagonal**. Whatever the characters are, each token takes nearly all of its
information from the token two positions earlier.

### 16.4 The first layer: heads with one job

Here are all six heads of the first layer on the same sentence.

![The six attention heads of layer 0.](../results/attention_layer0.png)

Three of them are strikingly sharp, and they are sharp in a way that does not depend on
*which* characters are present:

- **Heads 0 and 4** show a line one cell below the diagonal: each token looks at the
  **previous character**.
- **Head 2** shows a line two cells below the diagonal: each token looks at the
  character **two back**.
- **Heads 1, 3 and 5** are softer: a blur covering the previous two or three characters.

That is what one sentence shows. Is it true in general? A heatmap cannot answer that; a
measurement can. The script `scripts/attention_stats.py` runs the model on 32 validation
sequences of 128 characters and averages. For the first layer:

{{include results/attention_stats.md from="| Layer | Head | self" to="| 0 | 5 |"}}

The pattern holds. Averaged over about 3,500 token positions, head 0 puts 94% of its
weight on the previous character and head 4 puts 95% there. Head 2 puts 95% on the
character two back. The last column says the same thing differently: these three heads
spread their weight over an "effective" 1.0 to 1.1 tokens, which means they look at
exactly one. (The effective number is $e^{\text{entropy}}$, the same idea as the
perplexity of Exercise 8.2: 1 for a head that attends to a single token, $n$ for a head
that spreads its weight evenly over $n$.)

Why would the model build these? Think about what the first layer has to work with:
each position holds only its own character and its position. Before anything clever can
happen, a position needs to know its immediate neighbours. With "the previous
character" and "the character two back" copied into each position, the next layers can
work with the last three characters at once: a learned version of the trigram model of
Exercise 8.5. This is an interpretation, though a natural one: what was *measured* is
where these heads look, not what the rest of the network does with it.

Two more observations, both checkable in the figure:

- **Heads 0 and 4 do the same job.** The model spent two of its six first-layer heads
  on one function. Such redundancy is common in trained transformers, and it fits the
  ablation of Chapter 15, where reducing four heads to one cost only 1.3%.
- **The exceptions are at punctuation.** In the sample sentence, at the `:`, the `,` and
  the final `.`, heads 0 and 4 attend to the token itself instead of the previous one.
  The rule the heads learned is not purely "one step back".

For contrast, the same statistics on an **untrained** model give an effective number of
about 63 for every head, with under 2% of the weight on the previous character: with
weights near zero, every score is near zero, and the softmax spreads the attention
evenly over everything visible. All of the structure above was learned.

### 16.5 Deeper layers: looking further

![The six attention heads of the last layer (layer 5).](../results/attention_layer5.png)

The last layer looks completely different. No clean lines; instead scattered dark
cells, pale washes, and vertical stripes where many tokens attend to the same earlier
character. These patterns depend on the *content* of the sentence: change the sentence
and they move, whereas the lines of layer 0 stay where they are.

The averages per layer show the trend across the whole model:

{{include results/attention_stats.md from="| Layer | Mean distance" to="| 5 |"}}

- **Layer 0** is local: on average it looks 4.6 characters back and concentrates on
  about 3 tokens.
- **Layer 1** is the surprise: it is by far the most diffuse, spreading each token's
  attention over about 46 tokens at an average distance of 31. Having gathered the
  immediate neighbours in layer 0, the model appears to take a broad average over the
  recent context next.
- **Layers 2 and 3** are local again (about 8 characters).
- **Layers 4 and 5** reach progressively further back (17, then 26 characters) while
  staying fairly selective (12 to 17 tokens).

So the simple story "early layers are local, later layers are global" is roughly right
for this model but not exactly: layer 1 breaks the pattern. That is a typical outcome of
actually measuring. The model was not designed to organise itself this way; it found an
arrangement that lowers the loss, and our job is to describe what it found, not what we
expected.

### 16.6 What attention maps do not tell you

It is tempting to read a heatmap as an explanation: "the model predicted `z` *because*
it looked at the `C`". Be careful, for three reasons.

1. **After the first layer, positions no longer hold single characters.** By layer 5,
   the vector at the position of `C` contains information gathered from many other
   positions. "Token 20 attends to token 6" means it reads the *vector* at position 6,
   which may be mostly about something else.
2. **A weight is only half of the story.** What flows is the weight times the *value*
   vector. A large weight on a value that is nearly zero moves nothing; a small weight
   on a large value can matter a lot. And the residual connection carries information
   forward without any attention at all.
3. **One sentence proves little.** Patterns should be checked across many inputs, as we
   did in Section 16.4, before being stated as facts about the model.

A fair summary of what we established: we know *where* each head looks, reliably and
with numbers. We have plausible ideas about *why*. Establishing the why properly, by
intervening on the model and measuring what changes, is an active research field called
**mechanistic interpretability**, and a small model that you fully understand is an
excellent place to start in it.

### Exercises

**16.1 (easy)** In any heatmap of this model, what must the first row look like, and
why? What about the cell in the top-right corner?

**16.2 (easy)** A row of a heatmap for a token at position 10 shows weight 0.5 on
position 9 and 0.5 on position 5, and nothing elsewhere. What is the mean attention
distance for this row, and what is its effective number of tokens?

**16.3 (medium)** Run `scripts/visualize_attention.py` on a sentence of your own with a
repeated word, for example `"the king and the queen and the king"`. Do the heads of
layer 0 change their pattern? Do those of layer 5?

**16.4 (medium)** Before running anything: what should the attention statistics of an
*untrained* model look like, and why? Then check, by building a fresh
`GPT(cfg.model)` and passing its `attention_maps` to `head_statistics` from
`scripts/attention_stats.py`.

**16.5 (hard)** Heads 0 and 4 of layer 0 do the same job. Test whether the model needs
both: write a function that runs the model with one chosen head's output set to zero,
and measure the validation loss with head 0 removed, with head 4 removed, and with both
removed. What do you expect? (This is your first step into mechanistic
interpretability, and, as in Chapter 15, predict first.)

### Check your understanding

1. In an attention heatmap, what do the rows, the columns and the colours mean?
2. Why is the upper-right triangle always empty?
3. What does a dark line one cell below the diagonal tell you about a head?
4. Why is a single heatmap not enough to make a claim about a head?
5. Give one reason why "token A attends to token B" does not explain a prediction.
