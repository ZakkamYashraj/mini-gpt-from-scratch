# Part 3: Making it a real project

## Chapter 15: Experiments and ablations

> **Key ideas**
>
> - A claim such as "positional embeddings matter" is worth little until you have
>   **measured** it. An **ablation** removes or changes one component and measures the
>   difference.
> - A fair comparison changes **exactly one thing** and keeps everything else equal:
>   data, training steps, seed, model size.
> - Training is noisy. Repeat each run with different **seeds** to learn how big the
>   noise is, and do not believe differences smaller than that.
> - Losses per token cannot be compared between tokenizers. Convert to **bits per
>   character**.
> - Report what you found, including the results that did not match your expectations.

A working model is an engineering result. A measured comparison is a *scientific* one,
and it is what turns a project from "I followed a tutorial" into "I investigated
something". This chapter is about doing that properly on a small scale.

### 15.1 What makes a comparison fair

Suppose you train the model with 4 attention heads and get a validation loss of 1.47,
then change to 1 head, also lower the learning rate while you are at it, train for a
little longer, and get 1.49. What have you learned about heads? Nothing: three things
changed, and you cannot say which one moved the number.

The discipline of an experiment is simple to state:

1. **Choose a baseline**: one fixed configuration.
2. **Change exactly one thing** to make each variant.
3. **Keep everything else identical**: the data, the split, the number of steps, the
   batch size, the learning-rate schedule, the seed.
4. **Decide in advance what you will measure.** Here: the best validation loss reached
   during the run.

Removing or altering one component in this way is called an **ablation**.

### 15.2 The ablation script

The baseline is a model smaller than the main one, so that many runs fit in a
reasonable time (each takes about two minutes on the GPU):

{{code configs/ablation.yaml from="model:" to="tie_weights: true"}}

The variants are declared as data in `scripts/run_ablations.py`: a name, a description,
and the list of config overrides that defines the change.

{{code scripts/run_ablations.py from="# name -> (description, config overrides)" to="data.tokenizer=bpe" after="1"}}

Each variant is one line and one override. That is the "change exactly one thing" rule
made visible in the code. The function that runs one variant loads the baseline config,
applies the overrides, and calls the same `train()` function as `train.py`:

{{code scripts/run_ablations.py from="overrides = overrides + [" to="summary = train(cfg)"}}

Because it reuses `train()`, the ablation runs are trained by *exactly* the code that
trains the main model. A separate, slightly different training loop for experiments is a
common way to fool yourself.

### 15.3 Comparing different tokenizers: bits per character

One variant swaps the character tokenizer for BPE. Section 6.7 warned about the trap:
the two models report losses in different units. The character model's loss is "nats per
character"; the BPE model's is "nats per BPE token", and a BPE token covers about two
characters. (A **nat** is the unit you get from using the natural logarithm.)

To compare them, convert both to the same unit, **bits per character**:

$$\text{bits per character} = \frac{\text{loss per token}}{\text{characters per token} \times \ln 2}$$

Dividing by the characters per token spreads each token's loss over the characters it
covers. Dividing by $\ln 2 \approx 0.693$ converts nats to bits, the conventional unit
(one bit is the information in one fair coin flip).

**Worked for our two models.** Character baseline: loss 1.466 per token, 1 character per
token, so $1.466 / (1 \times 0.693) \approx 2.12$ bits per character. BPE: loss 3.017 per
token, 2.008 characters per token, so $3.017 / (2.008 \times 0.693) \approx 2.17$ bits
per character. A loss that looked twice as bad is, on a fair scale, 2.5% worse.

In the script:

{{code scripts/run_ablations.py from="cannot be compared between tokenizers" to="/ math.log(2)"}}

For a sense of scale: storing each of 65 characters with no model at all would take
$\log_2 65 = 6.0$ bits per character. Our model needs about 2.1. A language model *is*,
quite literally, a text compressor.

### 15.4 Noise: why one run is not enough

Train the same model twice with different random seeds and you get two different
numbers. The initial weights differ, the batches differ, and so the result differs
slightly. This spread is the **noise** of the experiment, and it sets the limit on what
you can conclude: *a difference between two variants that is no larger than the
difference between two runs of the same variant is not evidence of anything.*

So every variant here was run with two seeds (1337 and 1338). Two is the bare minimum:
enough to see roughly how large the noise is, not enough for precise statistics. The
table shows the mean, and in parentheses the lowest and highest value.

### 15.5 The results

{{include results/ablations.md from="| Variant" to="| `bpe`"}}

![Bits per character for each variant. The dot is the mean of two seeds, the line spans the two runs, and the dotted vertical line marks the baseline. Note that the axis does not start at zero.](../results/ablations.png)

First, the size of the noise. The two baseline runs gave 2.107 and 2.124: a spread of
about 0.02 bits per character between identical configurations. Keep that number in
mind for everything that follows.

### 15.6 Reading each result

**No positional embeddings: 10% worse. Clear.** The difference, 0.21 bits per
character, is ten times the noise, and both seeds agree. This confirms the argument of
Section 7.4: attention needs to be told about order.

But notice what did *not* happen: the model did not collapse. It still reaches 2.33
bits per character, far better than the bigram model. How can it work at all without
position information? The likely explanation, known from the research literature, is
that the causal mask leaks some of it in. The first token can attend only to itself, the
second to two tokens, and so on, so a token can in principle infer roughly how far along
it is from what it is allowed to see. We did not test this mechanism ourselves, so treat
it as a hypothesis. That is what honest reporting looks like: separate what you measured
from what you believe.

**One head instead of four: 1.3% worse. Small but real.** The ranges do not overlap
(2.137 to 2.150 against 2.107 to 2.124), and both seeds point the same way, so the
effect is probably real, but it is only slightly larger than the noise. With the same
total width, one wide head is nearly as good as four narrow ones for this small model
and simple data. The benefit of multiple heads (Section 11.1) is expected to grow with
the size of the model and the complexity of the text; our experiment can neither
confirm nor deny that.

**Untied weights: no detectable difference.** The untied model's mean is 0.6% *better*,
but its two runs (2.088 and 2.116) spread across the baseline's (2.107 and 2.124). The
ranges overlap, so this experiment cannot distinguish the two. As Section 12.4
anticipated, weight tying has little to offer when the vocabulary is 65 characters: it
saves 16,640 parameters out of 3.2 million. It would be wrong to report "untying
improves the model by 0.6%", and equally wrong to report "weight tying helps", on this
evidence. The supported statement is: *no measurable effect at this scale*.

**BPE instead of characters: 2.5% worse.** Very consistent across seeds (2.166 and
2.169), so the difference is real for this setup. The reason is visible in the raw
numbers: the BPE model's training loss ends far below its validation loss (1.67 against
3.02 nats per token), a much wider gap than the character model's (1.17 against 1.47).
It overfits harder. That makes sense: the same megabyte of text yields only half as many
training tokens, each of 512 token types is seen far less often than each of 65
characters, and the model has an embedding table eight times larger to fill. BPE's
advantages (twice the context in characters, no need to learn spelling) need more data
before they pay off. The conclusion is not "BPE is worse" but "on one megabyte of text,
with this model, characters are slightly better".

### 15.7 What this study does not show

Every experiment has limits, and stating them is part of the result.

- **Two seeds** per variant. Enough to spot noise, not enough for confident statistics.
- **One dataset, one model size, one training length.** The ranking could change with
  more data or a larger model; for BPE and for multiple heads, it probably would.
- **The validation set was used to choose the checkpoint.** We report the *best*
  validation loss of each run, which is very slightly optimistic. Careful research keeps
  a third split, a **test set**, that is touched only once, at the very end.
- **Each variant used the baseline's hyperparameters.** A variant might do better with
  its own tuned learning rate or dropout.

### 15.8 Designing your own experiment

A checklist to follow every time:

1. Write down the **question** in one sentence. ("Does dropout help this model?")
2. Write down your **prediction**, and why. Being wrong is where the learning is.
3. Define the **baseline** and the **single change**.
4. Fix the **budget** (steps, batch size) and the **metric** in advance.
5. Run at least **two seeds** of each.
6. Compare the difference with the spread between seeds.
7. Write the conclusion **no stronger than the evidence**, and list what the experiment
   does not show.

To add a variant to this project's study, add one line to `VARIANTS` and run
`python scripts/run_ablations.py`. Finished runs are cached, so only the new one is
trained.

### Exercises

**15.1 (easy)** A BPE model has a validation loss of 2.80 nats per token and its tokens
average 1.9 characters. A character model has a validation loss of 1.50. Which is
better?

**15.2 (easy)** Variant A scored 2.10 and 2.14 on two seeds; variant B scored 2.12 and
2.13. A friend concludes that B is better "because its average is lower and it is more
stable". What do you say?

**15.3 (medium)** Run two new ablations of your own choice against the baseline, for
example `model.dropout=0.0` and `model.n_layer=2`. Predict each result first. Add them
to `VARIANTS`, run the script, and read the results as in Section 15.6.

**15.4 (medium)** Why does the ablation use a smaller model and fewer steps than the
main configuration? What do you gain, and what do you risk?

**15.5 (hard)** Add a proper test set: split the data 80/10/10, select the checkpoint on
the validation split as now, and report the loss of that checkpoint on the test split.
Which functions need to change?

### Check your understanding

1. What is an ablation?
2. Why must only one thing change between the baseline and a variant?
3. Why run each variant with more than one seed?
4. Why can a loss per token not be compared between a character model and a BPE model?
5. The untied model scored slightly better than the tied one. Why is it wrong to
   conclude that untying helps?
