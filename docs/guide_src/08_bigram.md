## Chapter 8: The bigram model, a first working baseline

> **Key ideas**
>
> - A **bigram model** predicts the next token from the current token alone. It is the
>   simplest language model that works at all.
> - It can be built by **counting**, with no learning, or as a one-layer neural network.
>   Trained properly, the network arrives at the same answer as the counting.
> - It gives us a complete, working pipeline (data, model, loss, training loop,
>   generation) in forty lines, and a **baseline** loss of about 2.48 to beat.
> - Its weakness is the whole motivation for the rest of Part 2: it cannot use any
>   context beyond one token.

### 8.1 Why start with something this simple?

Before building the transformer we build the simplest thing that could possibly work,
for two reasons.

First, it gets the *whole pipeline* running end to end. If the data loading, the loss,
the training loop and the sampling all work for a trivial model, then when we swap in a
complicated model and something breaks, we know where to look.

Second, it gives a **baseline**: a number that any serious model must beat. A loss of
1.5 means little by itself. "1.5, when one-character lookup gets 2.48 and guessing gets
4.17" means a great deal.

The code for this chapter is `examples/ch08_bigram.py`.

### 8.2 A bigram model by counting

A **bigram** is a pair of neighbouring tokens. The bigram model asks a single question:
*given the current character, how often did each character follow it in the training
text?*

Build a table `counts` with 65 rows and 65 columns, where `counts[a, b]` is the number
of times character `b` came immediately after character `a`:

{{code examples/ch08_bigram.py section="1."}}

`current` is the text without its last character and `following` is the text without
its first, so the pairs `(current[i], following[i])` are exactly the neighbouring pairs.
`index_put_` with `accumulate=True` adds 1 to `counts[a, b]` for each of the million
pairs. (A `for` loop over the pairs would do the same thing, slowly.)

Dividing each row by its sum turns counts into probabilities. Row `t` of `probs` is then
a probability distribution: the model's prediction for what follows a `t`.

{{output examples/ch08_bigram.py from="1. Counting" to="guessing uniformly"}}

After `t`, the most likely next character is `h` (as in "the", "that", "thou"), then a
space. That is real knowledge about English, obtained by counting alone.

The validation loss is 2.48. Remember from Chapter 1 what that means: it is the average
of $-\ln(p)$, where $p$ is the probability the model gave to the character that really
came next. Since $e^{-2.48} \approx 1/12$, the bigram model is about as uncertain as
someone choosing among 12 equally likely characters, down from 65 for pure guessing.

### 8.3 The same model as a neural network

Now the same model again, but learned by gradient descent instead of counted. Why
bother, if counting already works? Because counting does not extend to longer contexts
(Exercise 8.5 shows where it breaks down), while the neural version does. This small
network is the seed that the GPT grows from.

{{code examples/ch08_bigram.py from="class BigramLM" to="return logits, loss"}}

The model's only parameter is a `(V, V)` table, created as an `nn.Embedding`. Row `a`
holds 65 numbers: the **logits** (unnormalised scores) for what follows token `a`.
Compare with the counting model: there, row `a` held probabilities; here it holds the
scores that a softmax will turn into probabilities.

Follow the shapes in `forward`, for a batch of `B` sequences of `T` tokens:

- `idx` is `(B, T)`. Looking up each ID gives `logits` of shape `(B, T, V)`: for every
  position of every sequence, 65 scores.
- `targets` is `(B, T)`: the correct next token at every position (Section 5.3).
- `F.cross_entropy` expects a 2-dimensional tensor of logits, one row per prediction,
  so `logits.view(B * T, V)` flattens the batch and time dimensions into one long list
  of `B*T` predictions, and `targets.view(B * T)` does the same for the answers.
  `view` is PyTorch's `reshape`.
- `F.cross_entropy` then applies softmax to each row, takes $-\ln$ of the probability of
  the correct token, and averages over all `B*T` predictions, exactly the recipe of
  Section 4.5.

Hold on to this method. The `forward` method of the real GPT ends with these same lines;
the only difference is how the logits are computed.

### 8.4 Training and sampling

The training loop is the five steps of Chapter 3:

{{code examples/ch08_bigram.py section="3."}}

{{output examples/ch08_bigram.py from="3. Training the neural bigram model" to="validation loss of the neural"}}

Two things to notice in the output.

**The loss ends at about 2.49, almost exactly the 2.48 of the counting model.** Gradient
descent, knowing nothing about counting, has found the same table. This is reassuring:
counting gives the provably best bigram table, and our training loop finds its way
there.

**The loss starts at about 4.6, which is *higher* than the 4.17 of uniform guessing.**
How can an untrained model be worse than guessing? `nn.Embedding` initialises its table
with random numbers of typical size 1. After the softmax those give a distribution that
is far from flat: the untrained model is *confidently wrong*, and Exercise 4.3 showed
how severely cross-entropy punishes that. This is why the GPT initialises its weights
with much smaller numbers (Chapter 12), so that it starts out close to 4.17.

The individual training losses also jump around (2.25, then 2.54) instead of falling
smoothly. That is the batch noise of Section 5.3: each number is measured on a
different random batch of only $32 \times 8 = 256$ characters. For a trustworthy number
you average over many batches, which is what the repository's `estimate_loss` function
does.

Generating text follows the loop of Chapter 1:

{{code examples/ch08_bigram.py from="def generate" to="return idx"}}

`logits[:, -1, :]` keeps only the scores at the last position, because that is the
prediction for the token we are about to add. `torch.multinomial` draws a token at
random according to the probabilities. `torch.cat` appends it. Here is what comes out:

{{output examples/ch08_bigram.py from="4. A sample from the bigram model"}}

It is nonsense, but it is *English-shaped* nonsense: mostly lowercase letters, spaces in
plausible places, the occasional capital after a line break, and letter pairs that
really occur in English. Every neighbouring pair is plausible. Nothing longer is.

### 8.5 What the bigram model cannot do

The bigram model decides what follows `h` without knowing whether that `h` came after
`t` (probably `e` next), after `w` (probably `a`, `e`, `i` or `o`), or at the start of a
word. Its entire view of the past is a single character.

Look again at `forward`: the logits at position `t` are computed from `idx` at position
`t` and nothing else. The tokens of a sequence never interact.

The fix is to let the vector at each position *gather information from the positions
before it* before the prediction is made. The mechanism that does this is called
**self-attention**, and it is the subject of the next chapter. Everything else in a GPT
is support for it.

For reference, the baselines so far on the validation set:

| Model | What it sees | Validation loss |
|---|---|---|
| Uniform guessing | nothing | 4.17 |
| Character frequencies only | nothing (but knows `e` is common) | 3.35 |
| Bigram | the previous character | 2.48 |
| Our GPT (Chapter 13) | the previous 256 characters | about 1.47 |

### Exercises

**8.1 (easy)** For the text `abababba`, build the bigram count table by hand (it is
2 by 2). What probability does the counting model give to `b` following `a`? Do not add
the smoothing `+1`.

**8.2 (easy)** A model has validation loss 2.48 and another has 1.47. Convert both to
"number of equally likely choices" using $e^{\text{loss}}$. This number is called the
**perplexity**.

**8.3 (medium)** Confirm the explanation of the high starting loss: create
`nn.Embedding(65, 65)`, compute the cross-entropy against some real targets, then
multiply the logits by 0.01 and compute it again. What do you get, and why?

**8.4 (medium)** In the training loop, change `block_size=8` to `block_size=64`. Does
the final loss of the bigram model improve? Explain why or why not.

**8.5 (hard)** Build a **trigram** counting model, which predicts from the previous two
characters, using a `(V, V, V)` table of counts. What validation loss does it reach?
Then explain why this approach cannot be continued to contexts of 10 or 256 characters.

### Check your understanding

1. What information does a bigram model use to predict the next token?
2. What does `logits.view(B * T, V)` do, and why is it needed?
3. Why is the loss of an untrained bigram network above $\ln(65)$?
4. In `generate`, why do we take `logits[:, -1, :]`?
5. What is a baseline for?
