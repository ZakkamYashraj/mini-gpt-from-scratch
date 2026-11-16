# Defend this project: a study guide

Everything you need to explain `mini-gpt-from-scratch` in an interview, a viva, or to a
curious friend. Read it aloud. If you can say each section in your own words without
looking, you are ready.

For the full explanations, with worked numbers and exercises, see
[LEARNING_GUIDE.md](LEARNING_GUIDE.md).

## 1. The 60-second version

> I implemented a GPT-style language model from scratch in PyTorch: the tokenizers, the
> attention mechanism, the transformer blocks, the training loop and the sampling code,
> without using PyTorch's built-in transformer or attention layers. It has about 10.7
> million parameters. Trained for about 22 minutes on a laptop GPU on one megabyte of
> Shakespeare, it reaches a validation loss of 1.46 per character and writes text that
> looks like a Shakespeare play. I then ran ablations to measure what each component
> contributes, for example that removing positional embeddings makes the model 10%
> worse, and I wrote tests that prove properties such as the model never looking at
> future tokens.

Three things make it more than a tutorial copy, and they are worth saying:

1. **It is tested.** 33 tests, including one that proves causality and one that proves
   the model can memorise a batch.
2. **It is measured.** Ablations with two seeds each, compared on a fair scale (bits per
   character), with the noise reported.
3. **It is reported honestly.** Including the overfitting after step 1,750, and an
   ablation (weight tying) that showed no effect.

## 2. Numbers to know by heart

| What | Value |
|---|---|
| Parameters | 10,745,088 (about 10.7M) |
| Shape | 6 layers, 6 heads, embedding width 384, context 256 characters |
| Dataset | Tiny Shakespeare: 1,115,394 characters, 65 distinct, 90/10 split |
| Loss of uniform guessing | ln(65) = 4.17 |
| Bigram baseline (validation) | 2.48 |
| Best validation loss | 1.46, at step 1,750 of 5,000 |
| Training time | about 22 minutes on an RTX 5050 (8 GB), mixed precision |
| GPU memory | about 4.6 GB |
| BPE | vocabulary 512 = 256 bytes + 256 merges, about 2.03 characters per token |
| Tests | 33 passing |

## 3. The core ideas in simple terms

### Attention

**One sentence.** Attention lets every token gather information from the other tokens,
taking more from the ones that are relevant to it.

**How.** Each token produces three vectors from its own vector, using three learned
matrices: a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a
**value** ("what do I pass on?"). The dot product of token *i*'s query with token *j*'s
key is a score for how relevant *j* is to *i*. A softmax turns each token's scores into
weights that sum to 1. The token's new vector is the weighted average of the values.

$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

**Analogy.** A library: you bring a question (query), each book has a label on its spine
(key), you take the contents (value) of the books whose labels match your question.

**Details you may be asked.**

- *Why divide by √d?* A dot product over more dimensions is larger in size. Large scores
  make the softmax nearly one-hot, where gradients vanish. The division keeps the scores
  at a steady scale.
- *Are the attention weights learned?* No. They are computed from the input each time.
  The learned parameters are the three matrices.
- *Why several heads?* One head gives one pattern of weights. Six heads of width 64 can
  follow six kinds of relationship for the cost of one head of width 384.
- *Cost?* The score table is T × T, so time and memory grow with the square of the
  context length.

**In the code.** `CausalSelfAttentionHead.forward` in `mini_gpt/model.py`: five numbered
steps.

### Causal masking

**One sentence.** The mask stops each token from attending to tokens that come after it,
so the model cannot cheat by reading the answer.

**Why it is needed.** In training, the model predicts the next token at every position
at once, and the target at position *t* is simply the input at position *t+1*. Without
a mask, position *t* could attend to position *t+1* and copy it.

**How.** A lower-triangular matrix says which pairs are allowed. Forbidden scores are
set to minus infinity *before* the softmax, so they get a weight of exactly zero and the
allowed weights still sum to 1.

**My evidence.** I removed the mask as an experiment. The training loss fell to 0.06
within 500 steps (an honest model cannot get below about 1.4), and the model generated
nothing but newline characters, because at generation time there is no future to copy.

**How I know mine works.** A test changes one token and asserts that every earlier
output is bit-for-bit unchanged. A second test removes the mask and asserts that the
outputs *do* change, to show that the first test can fail.

### BPE (byte-pair encoding)

**One sentence.** BPE builds a vocabulary of common chunks by starting from single bytes
and repeatedly merging the most frequent adjacent pair into a new token.

**How.** Count all adjacent pairs; merge the most frequent into a new token; repeat
until the vocabulary is the size you want. The ordered list of merges is the trained
tokenizer. To encode new text, replay the merges in the same order.

**Why bytes.** The base vocabulary is the 256 byte values, so any text in any language
can be encoded, even characters never seen in training. That is why the same tokenizer
will work for Telugu.

**What it learned on Shakespeare.** First merges: `" t"`, `"he"`, `" a"`, `"ou"`; by the
twelfth merge `" the"` is a single token.

**Trade-off.** Shorter sequences (2 characters per token, so twice the context) against
a larger vocabulary (512 instead of 65) in which each token is seen less often.

### Positional embeddings

**One sentence.** A learned vector for each position, added to the token's vector, so
the model knows where each token is.

**Why it is needed.** Attention computes a weighted *sum*, and a sum ignores order.
Without positions, "dog bites man" and "man bites dog" would look the same to the model.

**How.** A second embedding table with one row per position (256 rows). The row for the
position is added to the row for the token.

**Limitation.** There are only `block_size` rows, so the model cannot handle a longer
sequence. Modern models use other schemes (for example rotary embeddings).

### Weight tying

**One sentence.** The token embedding matrix and the output layer's weight matrix are
the same tensor.

**Why it is possible.** Both have shape (vocabulary size × embedding width), and in both,
row *k* is "the direction that means token *k*": one converts an ID into a vector, the
other scores a vector against every token.

**Why it is done.** It saves V × C parameters and ties what the model learns about a
token as an input to what it learns about it as an output. In GPT-2 this saves 38
million of 124 million parameters.

**What I found.** With my 65-character vocabulary it made no measurable difference (see
the ablation below). I say that plainly instead of claiming a benefit I did not observe.

### Three supporting ideas, in one line each

- **Residual connections**: `x = x + f(x)`. The gradient has a direct path through the
  additions, so deep stacks can be trained. In a 30-layer toy stack I measured the
  gradient at the first layer to be about 60 million times larger with them.
- **Layer normalisation**: rescale each token's vector to mean 0 and variance 1 (then a
  learned gain), so the scale of the numbers stays stable through the depth. I apply it
  before each sub-layer ("pre-norm", as in GPT-2).
- **Feed-forward block**: a two-layer network (width × 4, GELU) applied to each position
  separately. Attention moves information between tokens; this processes it. It holds
  two thirds of the parameters.

## 4. The results and what they mean

### The main run

All numbers below are from `results/training_log.csv`.

- Validation loss falls from 4.25 to its minimum of **1.465 at step 1,750**. (2.09 after
  only 250 steps, already better than the bigram model's 2.48.)
- After that the training loss keeps falling (to 0.63 by step 5,000) while the
  validation loss **rises** (to 1.69). That is **overfitting**: with only 1 MB of text,
  a 10.7M-parameter model starts memorising.
- The training script keeps the checkpoint with the best validation loss, so the saved
  model is the one from step 1,750. The last 3,250 steps made the model worse.
- The whole run took 22.5 minutes.

**If asked "what would you do about the overfitting?"**: more data is the real fix;
otherwise more dropout, a smaller model, or simply stopping early. One epoch is only
about 61 steps here, so by step 1,750 the model has seen the text about 29 times and by
step 5,000 about 82 times. I have evidence that dropout matters: in a side experiment on
the smaller ablation model, switching dropout off made the validation loss 3.8% worse
while the training loss went *down* (one seed).

### What the model learned, looked at directly

These are observations from the trained model, each reproducible with a script.

- **Predictions are confident where text is predictable.** After `First Citi` the model
  gives `z` a probability of 0.9999. After `ROMEO:` and a newline, the probability is
  spread over many capital letters (`I` 15%, `T` 14%, `W` 10%, ...).
- **The embedding table found categories by itself.** The nearest neighbours of `a` are
  `A`, `e`, `o`, `i` (its capital, then other vowels); of `.` are `?`, `;`, `!`, `:`.
  (`python examples/ch07_embeddings.py`)
- **First-layer heads have one job each.** Averaged over 32 validation sequences, two
  heads of layer 0 put 94% and 95% of their weight on the *previous* character, and a
  third puts 95% on the character *two back*. An untrained model spreads its attention
  evenly (under 2% on the previous character).
  (`python scripts/attention_stats.py`, `results/attention_stats.md`)
- **Those two previous-character heads are redundant.** Switching off either one changes
  the validation loss by 0.001 to 0.003; switching off both changes it by 0.012, about
  three times the sum. That fits the "one head" ablation costing only 1.3%.
- **Attention reaches further in deeper layers, roughly.** Mean attention distance per
  layer: 4.6, 31.1, 7.5, 8.6, 16.9, 26.4 characters. Layer 1 breaks the simple
  "early is local, late is global" story, and I report it that way.
- **Sampling temperature behaves as theory says.** At 0.2 the model loops ("the state of
  the sea ... the state of the sea-side"); at 1.5 it invents words ("companious").

A caution to state with these: attention maps show *where* a head looks, not *why* the
model predicts something. The head-removal experiment is stronger evidence than any
heatmap, because it changes the model and measures the effect.

### The ablations

Setup: a smaller baseline (4 layers, 4 heads, width 256, context 128, 3.2M parameters),
3,000 steps, two seeds per variant, the best validation loss of each run, converted to
bits per character. The two baseline runs differ by about 0.02, which is the noise
level.

| Variant | Bits/char | vs baseline | Verdict |
|---|---|---|---|
| Baseline | 2.116 | | |
| No positional embeddings | 2.327 | +10.0% | Clear effect |
| 1 head instead of 4 | 2.143 | +1.3% | Small, consistent |
| Untied weights | 2.102 | -0.6% | Within noise: no effect |
| BPE (512) instead of characters | 2.167 | +2.5% | Consistent; overfits more |

**No positional embeddings (+10%).** Ten times the noise: order information clearly
matters. The interesting part is that the model still works at all. A likely reason,
from the literature, is that the causal mask leaks position information: the first
token sees one token, the second sees two, and so on. I did not test that mechanism
myself, so I present it as a hypothesis.

**One head (+1.3%).** A small but consistent loss; the two seeds' ranges do not overlap
with the baseline's. For a small model on simple text, one wide head does nearly as well
as four narrow ones. I would expect the gap to grow with model and data size, but my
experiment does not show that.

**Untied weights (no effect).** The mean is slightly better, but the two runs overlap
with the baseline's two runs, so the experiment cannot tell them apart. The saving from
tying is only 16,640 parameters here. The honest conclusion is "no measurable difference
at this vocabulary size".

**BPE (+2.5%).** Per token the BPE loss is twice as large (3.02 against 1.47), but each
token covers two characters; per character the gap is only 2.5%. The BPE model overfits
much more (training loss 1.67 against validation 3.02): the same text gives half as many
training tokens spread over eight times as many token types. BPE's advantages need more
data than one megabyte.

**The methodological point to make.** A loss per token cannot be compared across
tokenizers. I converted everything to bits per character:
`loss / (characters per token × ln 2)`.

### What the experiments do not show

Two seeds only; one dataset; one model size; the checkpoint was chosen using the
validation set (a separate test set would be more rigorous). Say this before the
interviewer does.

## 5. Ten likely interview questions

**1. Walk me through what happens when I give your model the prompt "ROMEO:".**

The tokenizer turns the six characters into six integers. Each integer selects a row of
the token embedding table, and the row for its position is added, giving six vectors of
width 384. These pass through six transformer blocks. In each block, attention lets
every position gather information from itself and earlier positions, then a
feed-forward network processes each position; both are added to the running vector
through residual connections. After a final layer normalisation, a linear layer turns
the last position's vector into 65 scores, one per character. Softmax makes them
probabilities, I sample one character, append it, and repeat with the longer sequence.

**2. What is self-attention, and why are there queries, keys and values?**

It is a way for each token to build a new representation as a weighted average of
information from other tokens, with weights that depend on the content. The three roles
are separated because "what I am looking for" (query), "what I can be found by" (key)
and "what I contribute" (value) are different things. Scores are query-key dot products,
softmax makes them weights, and the output is the weighted sum of values.

**3. Why do you need a causal mask, and how do you know yours is correct?**

Because the training target at each position is the next input token, so without a mask
the model can copy the answer. I verified the consequence experimentally (loss 0.06,
output all newlines). My implementation is checked by a test that changes a token and
asserts that no earlier output changes, bit for bit, plus a control test showing the
check fails when the mask is removed.

**4. Why is the dot product scaled by the square root of the head size?**

The dot product's magnitude grows with the number of dimensions. Without scaling, the
softmax would be close to one-hot at initialisation, and a saturated softmax has very
small gradients. The scaling keeps the scores at unit scale regardless of head width.

**5. What did your positional-embedding ablation show?**

Removing them made the model 10% worse in bits per character, a difference ten times
the noise between seeds. The model did not fail completely, probably because the causal
mask gives some position information implicitly. I flag that explanation as a
hypothesis I did not test.

**6. What is weight tying, and did it help?**

Sharing one matrix between the input embedding and the output layer. In my ablation it
made no measurable difference, which is expected with a 65-token vocabulary where it
saves only about 17 thousand parameters. It matters for large vocabularies. I kept it
because it is standard and costs nothing.

**7. Characters or BPE: which is better, and how did you compare them?**

On this dataset, characters were about 2.5% better in bits per character. I could not
compare the raw losses, since a BPE token covers about two characters, so I normalised
by characters per token. The BPE model overfit more, because one megabyte of text gives
too few examples of each of 512 token types. With more data I would expect BPE to win,
as it does in every large model.

**8. Your validation loss rises after step 1,750. What is happening?**

Overfitting. The training loss keeps falling to 0.63 while the validation loss climbs
from 1.46 to 1.69: the model is memorising the training text. I keep the checkpoint
with the lowest validation loss. To reduce it I would add data first, then try more
dropout or a smaller model.

**9. How do you test machine-learning code, where bugs do not crash?**

By asserting things that must hold for any weights. Shapes. Properties: attention rows
sum to 1, nothing above the diagonal, causality, my LayerNorm matching PyTorch's. Round
trips for the tokenizers, including unseen and non-English text. Known values: the
untrained loss is close to ln(65). And an overfit test: the model must memorise one
batch and then regenerate it exactly, which exercises the targets, the loss, the
gradients and the optimizer together.

**10. What are residual connections and layer normalisation for?**

Residual connections add each sub-layer's output to its input. The derivative of
`x + f(x)` contains a constant 1, so gradients reach early layers undiminished; in a
30-layer toy network I measured the first-layer gradient to be tens of millions of times
larger with them. Layer normalisation standardises each token vector so that scales do
not drift through the depth. I use the pre-norm arrangement from GPT-2, which keeps the
residual path clean.

### Further questions worth preparing

- **How would you make generation faster?** A KV cache: store each layer's keys and
  values for past tokens, so each new token costs one position, not a full pass.
- **How would you make training faster?** Compute all heads in one batched tensor
  instead of a Python loop, and use PyTorch's built-in scaled-dot-product (flash)
  attention.
- **What does mixed precision do?** Runs the heavy matrix multiplications in float16; a
  gradient scaler prevents small gradients from underflowing to zero. I measured it: 196
  ms per step against 319 ms in pure float32, so 1.6 times faster. Peak memory was
  essentially unchanged (4.59 against 4.64 GB), because the parameters, gradients,
  optimizer state and residual stream stay in float32. So for this model the benefit is
  speed, not memory, and I would not repeat "it halves the memory" without measuring.
- **Why warmup and cosine decay?** Small steps while the weights are random and Adam's
  statistics are unreliable; large steps to make progress; small steps again to settle.
  Honest footnote: on the tiny 2,000-step CPU-sized model I compared it with a constant
  learning rate and the constant rate was marginally better (1.807 against 1.831, one
  seed, within noise). The schedule is standard for long runs of larger models; I did
  not demonstrate a benefit at the smallest scale.
- **Why AdamW and not SGD?** Per-parameter step sizes and momentum, which matter when
  gradients differ in scale across layers; plus weight decay applied correctly.
- **How is your model different from GPT-2?** The same design, much smaller (6 layers
  against 12, 10.7M parameters against 124M), character-level by default, and trained
  on 1 MB instead of 40 GB.
- **What was the hardest bug or surprise?** Have your own answer ready. Candidates from
  this project: the BPE model looking "twice as bad" until the loss was normalised per
  character; the mask-removal experiment; weight tying showing no effect; mixed
  precision not saving memory; the most diffuse attention being in layer 1, not the last
  layer. A real environment problem also happened: Windows Smart App Control blocked the
  newest CUDA build of PyTorch, and the fix was to use a slightly older official build,
  not to switch the protection off.

## 6. Being honest about how it was built

You will be asked. Have a straightforward answer:

- The design follows Andrej Karpathy's nanoGPT and his "Let's build GPT" lecture, and
  the README credits both.
- The repository was built with the help of an AI coding assistant; the commit history
  shows this.
- What you bring is understanding: you can explain every component, you can read the
  experiments critically, and (once you have done Part 4 of the learning guide) you have
  rebuilt it yourself from an empty folder.

Do not claim more than is true, and do not undersell what is: being able to explain,
test and critically evaluate a system is exactly the skill being assessed.

## 7. A last checklist

Can you, without notes:

- [ ] draw the architecture, from token IDs to logits, with shapes?
- [ ] write the attention formula and say what each symbol is?
- [ ] compute attention weights by hand for three tokens?
- [ ] explain why the mask is triangular and why minus infinity?
- [ ] run BPE by hand for three merges?
- [ ] explain each row of the ablation table, including the noise?
- [ ] say what the loss curve shows and what you would do about it?
- [ ] name three limitations of the project and how you would address them?
