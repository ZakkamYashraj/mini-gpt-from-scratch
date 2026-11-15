# Part 1: Foundations

## Chapter 1: What a language model is

> **Key ideas**
>
> - A **language model** is a function that takes some text and returns a probability
>   for every possible *next* piece of text.
> - Text is cut into small pieces called **tokens**. Our first tokens are single characters.
> - **Training** means adjusting millions of numbers (the **parameters**) so that the
>   model gives high probability to what really came next in real text.
> - **Generating** means repeatedly asking "what comes next?", picking an answer, adding
>   it to the text, and asking again.
> - A GPT is one particular design for that function. This guide builds it piece by piece.

### 1.1 The one job of a language model

Finish this sentence:

> The cat sat on the ...

You probably thought "mat", perhaps "floor" or "sofa". You certainly did not think
"democracy" or "seventeen". Without noticing, you assigned *probabilities* to possible
continuations: high for "mat", lower for "sofa", almost zero for "seventeen".

A **language model** does exactly this and nothing else. It is a function:

- **input**: a piece of text (called the **context**)
- **output**: a probability for every possible next piece

That is the whole job. ChatGPT, Claude, and the model in this repository all do this one
thing. Everything that looks like intelligence (answering questions, writing code,
translating) comes out of doing next-piece prediction extremely well.

### 1.2 Tokens: the pieces of text

What is a "piece"? Models do not work with whole words directly, because there are too
many words and new ones appear all the time. Instead, text is cut into **tokens** from a
fixed list called the **vocabulary**.

The simplest choice, and the one we start with, is: *one token = one character*. Our
training text is a 1 MB file of Shakespeare plays ("Tiny Shakespeare"). It contains
exactly 65 distinct characters: the 26 lowercase letters, the 26 uppercase letters, the
space, the newline, and some punctuation. So our vocabulary has 65 tokens.

Each token gets a number, its **token ID**, from 0 to 64. The model never sees letters,
only these numbers. Chapter 6 builds the code that converts text to IDs and back, and
then a smarter tokenizer whose tokens are common chunks of characters.

### 1.3 "Predicting the next token", concretely

Take the text `hello`. With character tokens it contains four prediction problems:

| the model sees (context) | the correct next token (target) |
|---|---|
| `h` | `e` |
| `he` | `l` |
| `hel` | `l` |
| `hell` | `o` |

For each context, the model outputs 65 numbers: one probability per vocabulary token,
all between 0 and 1, adding up to exactly 1. Such a list is called a **probability
distribution**.

The figure below shows a real example from the model trained in this repository. The
context is `First Citi`, and the bars show the probability the model gives to each of its
favourite next characters. It is almost certain that the next character is `z` (as in
"First Citizen", a speaker name that appears many times in the plays).

![The model's probabilities for the character after "First Citi". Only the eight most likely of the 65 characters are shown.](figures/next_token.png)

Notice what the model is **not** doing: it is not looking anything up, and it has no list
of rules about English. It has only learned, from examples, which characters tend to
follow which contexts.

### 1.4 How good is a prediction? A first look at the loss

To improve the model we need a single number that says how bad its predictions are. This
number is called the **loss**; smaller is better.

The idea: look at the probability $p$ that the model gave to the token that *actually*
came next.

- If $p$ is close to 1, the model was confident and right. The loss should be near 0.
- If $p$ is close to 0, the model was confident and wrong. The loss should be large.

The function $-\ln(p)$ (minus the natural logarithm) has exactly this shape:

| probability given to the correct token | loss $= -\ln(p)$ |
|---|---|
| 1.0 | 0.00 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.01 | 4.61 |

This loss is called the **cross-entropy** loss. We average it over many predictions.

One value is worth remembering. A model that knows nothing would spread its probability
equally over all 65 characters, giving $p = 1/65$ to the correct one. Its loss would be

$$-\ln\left(\frac{1}{65}\right) = \ln(65) \approx 4.17$$

So **4.17 is the loss of pure guessing** for our vocabulary. When you start training, the
loss should begin near 4.17 (if it does not, something is wrong: we will use this as a
test in Chapter 17). Our finished model reaches about 1.5.

> **Note**
>
> Why the logarithm, and not simply $1 - p$? One reason is that the probability of a
> whole text is the *product* of the probabilities of its tokens, and the logarithm turns
> a product into a sum, which is far easier to work with. Chapter 4 returns to this.

### 1.5 Training in one paragraph

The model is a function with millions of adjustable numbers inside it, called
**parameters** (or **weights**). At the start they are random, so the predictions are
garbage and the loss is about 4.17. Training repeats one step thousands of times:

1. take a batch of contexts from the real text, together with the true next tokens;
2. ask the model for its predictions and compute the loss;
3. work out, for every parameter, whether nudging it up or down would lower the loss;
4. nudge every parameter a tiny amount in the helpful direction.

Step 3 uses derivatives, and it is the part where your calculus becomes useful.
Chapter 4 explains it fully.

### 1.6 Generating text

Once trained, the model can write. The procedure is a loop:

1. start with some text, the **prompt**, for example `ROMEO:`;
2. ask the model for the distribution over the next token;
3. **sample** one token from that distribution (pick at random, but with likely tokens
   picked more often);
4. append the chosen token to the text;
5. go back to step 2.

Each new token is produced using all the text so far, including the tokens the model
itself just wrote. This is called **autoregressive** generation. Chapter 14 covers the
knobs that control it (temperature, top-k, top-p).

Here is text produced this way by our model at four moments during one training run. At
step 0 the parameters are random. Watch the model discover, in order: which characters
are common, where spaces and line breaks go, short words, the "NAME: speech" layout of a
play, and finally something that looks like Shakespeare from a distance.

{{include results/samples.md from="## Untrained" demote="2"}}

### 1.7 What we are going to build

The figure shows the whole pipeline. Each box is a chapter of Part 2 and a file in the
repository.

![The pipeline: text is tokenized into IDs, the model turns IDs into next-token probabilities, and a sampled token is appended to the text.](figures/pipeline.png)

| Piece | What it does | File | Chapter |
|---|---|---|---|
| Tokenizer | text ↔ token IDs | `mini_gpt/tokenizer.py` | 6 |
| Data pipeline | text file → arrays of IDs → training batches | `mini_gpt/data.py`, `data/prepare.py` | 5, 6 |
| Model | token IDs → next-token scores | `mini_gpt/model.py` | 7 to 12 |
| Training loop | adjusts the parameters | `train.py` | 13 |
| Generation | writes new text | `generate.py` | 14 |
| Experiments | measure and compare | `scripts/` | 15, 16 |
| Tests | prove the pieces are correct | `tests/` | 17 |

The model we train has about 10 million parameters. For scale: GPT-2 (2019) had 124
million in its smallest version, and current large models have hundreds of billions.
Ours is the same *kind* of machine, just very small, which is exactly what makes it
possible to understand completely.

### Exercises

**1.1 (easy)** Write out all the (context → target) pairs for the text `cat sat`, using
character tokens. How many are there? Remember that the space is a token too.

**1.2 (easy)** A model outputs these probabilities for the next character:
`e`: 0.6, `a`: 0.3, `o`: 0.1. The true next character is `a`. What is the loss for this
prediction? What would it be if the true next character were `e`?

**1.3 (medium)** A model's vocabulary has 512 tokens. What loss would you expect before
any training? If, after training, the average loss is 2.08, what probability is the model
giving to the correct token "on average" (precisely: what single probability $p$ has
$-\ln p = 2.08$)?

**1.4 (hard)** Why not skip the neural network and simply store, for every possible
context, a table of what came next? Estimate how many different contexts of 256
characters exist when there are 65 possible characters, and compare that with the number
of characters in our training file (about one million).

### Check your understanding

1. In one sentence, what are the input and the output of a language model?
2. What is the difference between a token and a token ID?
3. What does a loss of 0 mean? Can a loss be negative?
4. Why does generation feed the model its own output?
5. True or false: a trained language model contains a copy of its training text that it
   searches through.
