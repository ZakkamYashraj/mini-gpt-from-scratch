## Chapter 14: Generating text

> **Key ideas**
>
> - Generation is a loop: get the distribution for the next token, **sample** one token
>   from it, append it, repeat.
> - **Temperature** divides the logits before the softmax. Below 1 makes the model more
>   cautious and repetitive; above 1 makes it more adventurous and error-prone.
> - **Top-k** keeps only the `k` most likely tokens. **Top-p** keeps the smallest set of
>   tokens whose probabilities add up to `p`. Both remove the long tail of unlikely
>   tokens that causes most mistakes.
> - The model can only see `block_size` tokens, so the context is cropped to the most
>   recent ones.

### 14.1 The generation loop

A trained model maps a context to a probability distribution over the next token.
Chapter 1 described how to turn that into text, and the bigram model of Chapter 8 did
it in five lines. The real method has the same skeleton:

{{code mini_gpt/model.py symbol="GPT.generate"}}

Line by line:

- **`idx[:, -self.cfg.block_size :]`** keeps the last `block_size` tokens. The model has
  positional embeddings and a mask only for that many positions, so anything older has
  to be dropped. When the text is still shorter than `block_size`, this keeps all of it.
- **`self(idx_cond)`** runs the forward pass. The result has shape `(B, T, V)`:
  predictions at *every* position.
- **`logits[:, -1, :]`** keeps the prediction at the last position only. That is the one
  for the token that does not exist yet. The others predict tokens we already have.
- **`sample_next_token`** chooses a token from those logits (the rest of this chapter).
- **`torch.cat`** appends it, and the loop goes round again with a sequence one token
  longer.

`@torch.no_grad()` is there because nothing is being learned. Before calling `generate`
you must also call `model.eval()`, to switch off dropout.

> **Detail you can ignore for now**
>
> This loop is wasteful: to produce token 201 it recomputes everything for the first
> 200 tokens, which it already computed when producing token 200. Real systems store
> the keys and values of earlier tokens and reuse them (a **KV cache**). It makes
> generation much faster and changes nothing in the output. It is listed as future work
> in the README, and it is a good project once you understand the rest.

### 14.2 Why sample at all?

The obvious strategy is to always take the most likely token. This is called **greedy
decoding**, and it works badly. Text produced this way is dull and tends to fall into
loops ("the king and the king and the king..."), because once the model reaches a
phrase it has seen often, the most likely continuation leads back to the same phrase.

Sampling means picking at random according to the probabilities: a token with
probability 0.3 is chosen 30% of the time. This gives variety, and running the model
twice gives two different texts. But pure sampling has the opposite weakness. The
vocabulary contains many tokens that are individually very unlikely but together add up
to a noticeable amount of probability. Every so often sampling picks one of them, and
one strange character can derail everything after it.

The three settings of this chapter are ways of steering between "boring" and
"derailed". The examples are in `examples/ch14_sampling.py`, on a toy distribution over
six tokens with logits `[3.0, 2.5, 2.0, 1.0, 0.0, -2.0]`.

### 14.3 Temperature

**Temperature** is a positive number $T$. The logits are divided by it before the
softmax:

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

Recall from Section 2.7 that softmax depends on the *differences* between the logits.
Dividing by $T$ scales those differences:

- $T < 1$ makes the differences **larger**. The top token pulls further ahead. The
  distribution becomes sharper.
- $T > 1$ makes the differences **smaller**. The distribution becomes flatter, and
  unlikely tokens get more of a chance.
- $T = 1$ leaves the model's own distribution unchanged.

{{output examples/ch14_sampling.py from="1. Temperature" to="T = 2.0"}}

![The same six logits at three temperatures.](figures/temperature.png)

At $T = 0.5$ the favourite `e` goes from 46% to 66% and the rare `z` all but disappears.
At $T = 2$ the favourite falls to 33% and `z` rises from 0.3% to 2.7%.

The two extremes are worth knowing. As $T$ approaches 0, all the probability goes to
the single most likely token: greedy decoding. As $T$ grows very large, the distribution
approaches uniform: random characters. (The name comes from physics, where heating a
system makes its particles move more randomly.)

### 14.4 Top-k

**Top-k** sampling is a blunt instrument: keep the `k` tokens with the highest logits,
and give every other token a probability of zero.

{{code mini_gpt/model.py symbol="top_k_filter"}}

`torch.topk` returns the `k` largest values in order, so the last of them,
`values[..., -1, None]`, is the `k`-th largest logit. Everything below that threshold
is set to $-\infty$, which the softmax will turn into 0. This is the same trick as the
causal mask (Section 10.4).

{{output examples/ch14_sampling.py from="2. Top-k" to="top_k = 3"}}

With `k = 3`, the tokens `i`, `u` and `z` are removed, and the probabilities of the
three survivors are rescaled so that they still sum to 1 (46% becomes 51%, and so on).

The weakness of top-k is that `k` is fixed. When the model is sure (one token at 99%),
keeping 50 candidates still lets in junk. When the model is genuinely unsure among 200
reasonable words, keeping 50 cuts off good options.

### 14.5 Top-p (nucleus sampling)

**Top-p** adapts to the situation. Sort the tokens from most to least likely and keep
adding them to a set until their probabilities reach `p`. The set is called the
**nucleus**. A confident distribution gives a small nucleus; a flat one gives a large
nucleus.

{{code mini_gpt/model.py symbol="top_p_filter"}}

The steps: sort the logits in descending order; convert to probabilities; compute the
running total (`cumsum`); for each token, compute how much probability lies *ahead* of
it in the sorted order (`mass_before`); drop a token if the tokens ahead of it already
cover `p`. The last line (`scatter`) puts the surviving logits back in their original
vocabulary positions.

**By hand**, for our toy distribution with `p = 0.9`:

| token | probability | probability before it | keep? |
|---|---|---|---|
| `e` | 0.462 | 0.000 | yes |
| `a` | 0.280 | 0.462 | yes |
| `o` | 0.170 | 0.742 | yes (0.742 is still below 0.9) |
| `i` | 0.062 | 0.912 | no (0.9 is already covered) |
| `u` | 0.023 | 0.974 | no |
| `z` | 0.003 | 0.997 | no |

{{output examples/ch14_sampling.py from="3. Top-p" to="top_p = 0.5"}}

With `p = 0.9` three tokens survive; with `p = 0.5` only two do, because `e` and `a`
together already exceed one half.

![Top-k and top-p remove the unlikely tail and rescale what is left.](figures/topk_topp.png)

### 14.6 Putting the three together

{{code mini_gpt/model.py symbol="sample_next_token"}}

The order matters. Temperature is applied first, since it reshapes the distribution
that the filters then cut. Then top-k, then top-p. Finally a softmax turns the filtered
logits into probabilities and `torch.multinomial` draws one token. A temperature of
exactly 0 is treated as a special case, greedy decoding, because dividing by zero is
not possible.

To convince yourself that sampling follows the probabilities, draw a thousand tokens
and count:

{{output examples/ch14_sampling.py from="4. Draw 1000 samples"}}

The counts track the probabilities of Section 14.3 (about 46% `e` at $T = 1$), up to the
randomness of sampling.

### 14.7 `generate.py`

The script wraps all this for the command line. Its central function rebuilds the model
from a checkpoint:

{{code generate.py symbol="load_model"}}

This is where the contents of the checkpoint (Section 5.7) pay off: the saved config
gives the model its shape, the saved tokenizer converts the prompt exactly as during
training, and `load_state_dict` copies the saved weights into the freshly built model.

```bash
python generate.py --prompt "ROMEO:" --max_new_tokens 300 --temperature 0.8 --top_k 50
python generate.py --prompt "To be" --temperature 1.0 --top_p 0.9 --seed 42
```

Here is the trained model continuing the prompt `ROMEO:` at three temperatures (same
seed, no top-k or top-p). These are real outputs.

**Temperature 0.2:** safe, grammatical, and stuck in a loop.

```text
ROMEO:
I have no more than the state of the sea
Than the stone of the state of the story
That which they have been a power to the sea,
To the state of the state of the sea-side,
That we have been so best a straight of the sea
```

**Temperature 0.8:** varied, mostly real words, the layout of a play.

```text
ROMEO:
I lay thee, but some law I have made thee thou art.

BENVOLIO:
Besides, marry, O Romeo.

MISTRESS OVERDONE:
Now, be come to him, or I ne'er will be none.

MENENIUS:
What is my name?
```

**Temperature 1.5:** inventive, and falling apart.

```text
ROMEO:
Well, dull triumph: I am not well.

BISHOP OF;
I.

MENENIUS:
Well; the, early, O,
Draw.

MISTRESS OVERDONE:
Now, be companious troth,
Before it not not: else, terms lits again,
Of nope. Onlyme, say it were never by
```

At 0.2 the model keeps choosing its favourite continuations and circles around "the
state of the sea": the looping described in Section 14.2. At 1.5 it takes unlikely
characters often enough to produce non-words (`companious`, `Onlyme`) and a broken
speaker label (`BISHOP OF;`). At 0.8 it stays between the two. Notice also that the
model invents a dialogue between characters from *different* plays (Benvolio, Mistress
Overdone, Menenius): it has learned what a play looks like, not what any play says.

Sensible starting values are a temperature between 0.7 and 1.0 with either `top_k`
around 50 or `top_p` around 0.9. There is no correct setting; it is a matter of taste
and of what the text is for.

### Exercises

**14.1 (easy)** Three tokens have logits $(2, 1, 0)$. Compute their probabilities at
temperature 1 and at temperature 0.5.

**14.2 (easy)** Four tokens have probabilities $(0.4, 0.3, 0.2, 0.1)$. What are the
probabilities after top-k with `k = 2`? After top-p with `p = 0.6`? After top-p with
`p = 0.75`?

**14.3 (medium)** In `generate`, what would happen without the line that crops the
context to `block_size`? Try it: generate 300 tokens from a model with `block_size`
256 after commenting the crop out.

**14.4 (medium)** Generate text from the trained model at temperatures 0.2, 0.8 and 1.5
(no top-k or top-p). Describe the differences. Then add `--top_k 20` to the 1.5 run.
What changes?

**14.5 (hard)** Write `generate_stream(model, tokenizer, prompt, n)` that prints each
character as soon as it is produced, like a chat interface, instead of printing
everything at the end. Why is this easy with the character tokenizer and trickier with
the BPE tokenizer?

### Check your understanding

1. Why does `generate` use only `logits[:, -1, :]`?
2. What does a temperature of 0.5 do to the distribution? And a temperature of 2?
3. How do top-k and top-p differ in how many tokens they keep?
4. Why is temperature applied before the top-k and top-p filters?
5. Why must `model.eval()` be called before generating?
