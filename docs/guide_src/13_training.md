## Chapter 13: Training the model

> **Key ideas**
>
> - The training loop is still the five steps of Chapter 3. Around them we add four
>   refinements that make training fast and stable.
> - A **learning-rate schedule**: warm up for 100 steps, then decay along a cosine curve.
> - **Mixed precision**: do most arithmetic in 16-bit numbers on the GPU, which makes
>   each training step about 1.6 times faster on our model.
> - **Gradient clipping**: if the gradient is unusually long, shorten it, so that one bad
>   batch cannot wreck the model.
> - **Evaluation**: at regular intervals, measure the loss on training and validation
>   data with dropout off, log it, and keep the best checkpoint.

### 13.1 The shape of `train.py`

The function `train` in `train.py` has three parts: set-up, the loop, and saving the
results. Here is the set-up:

{{code train.py from="set_seed(cfg.train.seed)" to="scaler = torch.amp.GradScaler"}}

Each line is something from an earlier chapter: the seed (Section 5.6), the device
(Section 3.3), the tokenized data (Section 6.8), the model (Chapter 12), and the AdamW
optimizer (Section 5.5). The vocabulary size is copied from the tokenizer into the model
configuration, because only after tokenizing do we know how many tokens there are.

`.to(device)` moves all of the model's parameters and buffers to the GPU. The last line
creates a `GradScaler`, which belongs to mixed precision (Section 13.3).

The loop then repeats four things: set the learning rate; every 250 steps evaluate;
every 250 steps print a sample; and do one training step. We take them one at a time.

### 13.2 The learning-rate schedule

Section 5.4 explained that the ideal learning rate changes during training. Ours follows
this curve:

![The learning rate over the 5000 steps of the main configuration.](figures/lr_schedule.png)

**Warmup (the first 100 steps).** The rate rises in a straight line from nearly zero to
its peak. At the very start the weights are random, the gradients are large and point
in erratic directions, and Adam's running averages (Section 5.5) have seen too few
gradients to be reliable. Large steps at that moment can throw the model somewhere bad.
A short period of small steps lets everything settle.

**Cosine decay (the rest).** The rate then slides down, following the shape of half a
cosine wave, from the peak to a floor of one tenth of the peak. Large steps early cover
ground quickly; small steps late make fine adjustments without overshooting. Nothing is
magical about the cosine; it is simply a smooth curve that starts flat, falls steadily,
and ends flat, and it works well in practice.

{{code mini_gpt/utils.py symbol="get_lr"}}

The cosine branch deserves a careful read:

- `progress` runs from 0 (end of warmup) to 1 (end of training).
- $\cos(\pi \cdot \text{progress})$ runs from $\cos(0) = 1$ down to $\cos(\pi) = -1$.
- So `coefficient` $= 0.5\,(1 + \cos(\dots))$ runs from 1 down to 0.
- The result is `min_lr` plus that fraction of the gap between `max_lr` and `min_lr`.

**Check three values by hand** for the main configuration (`max_lr` $= 10^{-3}$,
`min_lr` $= 10^{-4}$, 100 warmup steps, 5000 steps):

- Step 0: warmup, $10^{-3} \times 1/100 = 10^{-5}$.
- Step 2550: `progress` $= 2450/4900 = 0.5$, so the coefficient is
  $0.5\,(1 + \cos(\pi/2)) = 0.5$, and the rate is
  $10^{-4} + 0.5 \times 9 \times 10^{-4} = 5.5 \times 10^{-4}$.
- Step 5000: $10^{-4}$.

The training loop asks for the rate at every step and writes it into the optimizer:

{{code train.py from="# Set the learning rate for this step" to="] = lr"}}

### 13.3 Mixed precision

Computers store decimal numbers in a fixed number of bits. PyTorch's default,
**float32**, uses 32 bits and gives about 7 significant digits. **float16** uses 16
bits: about 3 significant digits, and nothing larger than 65,504.

A number with half the bits takes half the space, and a modern GPU can do arithmetic on
such numbers considerably faster. And neural networks are surprisingly tolerant of low
precision: the
computation is so noisy anyway (random batches, dropout) that the third decimal place
rarely matters. **Mixed precision** takes advantage of this: run most of the forward
pass in float16, and keep float32 for the few places that need it.

In PyTorch this takes two additions to the training step. First, the forward pass runs
inside an `autocast` block:

{{code train.py from="# 2) Forward pass" to="_, loss = model(x, y)"}}

Inside the block, PyTorch automatically chooses float16 for operations where that is
safe (matrix multiplications, mainly) and float32 for the rest. The parameters
themselves stay in float32 the whole time.

Second, the backward pass goes through the `GradScaler`:

{{code train.py from="# 3) Backward pass" to="scaler.scale(loss).backward()"}}

The problem it solves: gradients can be very small numbers, and float16 cannot
represent numbers much below about $10^{-7}$. Such a gradient would be rounded to
exactly zero (this is called **underflow**), and that parameter would stop learning. The
scaler multiplies the loss by a large factor before `backward()`, which multiplies every
gradient by the same factor and lifts them into the range float16 can hold. Before the
optimizer uses the gradients, it divides the factor back out:

{{code train.py from="# 5) Update" to="scaler.update()"}}

**Is it worth it? A measurement.** Timing 40 training steps of the main configuration on
the RTX 5050, with and without mixed precision:

| | time per step | peak GPU memory |
|---|---|---|
| float32 only (`amp: false`) | 319 ms | 4.64 GB |
| mixed precision (`amp: true`) | 196 ms | 4.59 GB |

Each step is **1.6 times faster**: the 5,000 training steps take about 16 minutes
instead of about 27. The memory, however, is practically unchanged, and that is worth noticing because
"mixed precision halves the memory" is often repeated. It did not here. The parameters,
their gradients and the optimizer's running averages all stay in float32, and so does
the residual stream (see the box below); only temporary results inside the blocks
become smaller. How much memory mixed precision saves depends on the model, and the
only way to know is to measure, which is the theme of Chapter 15.

On a CPU none of this applies; the line `use_amp = cfg.train.amp and device.type ==
"cuda"` switches it all off, and `autocast` and the scaler then do nothing.

> **Detail you can ignore for now**
>
> You may wonder how the residual stream stays accurate in float16. It does not have
> to: the embeddings are float32, and adding a float16 tensor to a float32 tensor gives
> float32. So the stream itself, and the hand-written LayerNorm that reads it, remain in
> full precision; only the heavy matrix multiplications inside the blocks use float16.

### 13.4 Gradient clipping

Now and then a batch produces an enormous gradient: an unusual passage, or an unlucky
combination of weights. One enormous step can undo hours of progress.

From multivariable calculus: the gradient is a vector, and it has a **length** (its
norm), the square root of the sum of the squares of all its components. Here that vector
has 10.7 million components, one per parameter, but the definition is the same.
**Gradient clipping** says: if the length exceeds a threshold (1.0 for us), scale the
whole vector down so that its length equals the threshold. The *direction* is kept;
only the size is limited.

**A small example.** A gradient of $(3, 4)$ has length $\sqrt{9 + 16} = 5$. Clipped to a
maximum length of 1 it becomes $(3/5, 4/5) = (0.6, 0.8)$: the same direction, length 1.
A gradient of $(0.3, 0.4)$, with length 0.5, is left alone.

{{code train.py from="# 4) Gradient clipping" to="clip_grad_norm_"}}

`scaler.unscale_` first removes the mixed-precision scaling factor, so that the length
is measured on the true gradients.

### 13.5 Measuring progress

The loss printed for a single training batch is too noisy to judge progress by, and it
is measured with dropout on. So at regular intervals the loop pauses and measures
properly:

{{code train.py symbol="estimate_loss"}}

Four things make this an honest measurement:

- `@torch.no_grad()` switches off gradient tracking: we are only looking, not learning.
- `model.eval()` switches off dropout, so the model is measured as it will be used. (And
  `model.train()` switches it back on afterwards. Forgetting either call is a classic
  bug.)
- It averages over many batches (100 in the main configuration) instead of trusting
  one.
- It measures **both** splits. The gap between them is how we detect overfitting
  (Section 5.2).

The loop calls it every `eval_interval` steps, prints and logs the result, and saves a
checkpoint whenever the validation loss is the best so far:

{{code train.py from="# Every so often: measure train/val loss" to="save_checkpoint("}}

At the same intervals the model also writes a short sample of text, so that you can
*see* the learning and not just watch a number fall.

### 13.6 Running it

```bash
python train.py --config configs/small.yaml --results_dir results
```

The run prints the device, the model's size and its parameter count, then a line per
evaluation. On the laptop this guide was written on (an RTX 5050 with 8 GB), the whole
run takes about 22 minutes. This is the log of that run, as saved in
`results/training_log.csv`:

```text
{{include results/training_log.csv}}
```

![Training and validation loss of the main run.](../results/loss_curve.png)

This run is a textbook example of everything in Section 5.2, so read the curve slowly.

- **Step 0: 4.25.** Just above $\ln(65) = 4.17$, as an untrained model should be.
- **Steps 0 to 1,000: fast progress.** The validation loss drops to 2.09 after 250 steps
  (already better than the bigram model's 2.48) and to 1.53 after 1,000. Both curves
  fall together: the model is learning patterns that hold for unseen text.
- **Step 1,750: the best validation loss, 1.465.** This is the checkpoint that is
  saved.
- **After step 1,750: overfitting.** The training loss keeps falling smoothly, all the
  way to 0.63, but the validation loss turns around and climbs to 1.69. The model has
  run out of general patterns it can extract from one megabyte of text and is now
  memorising that particular megabyte. The gap between the curves, 0.36 at step 1,750,
  grows to more than 1.0 by the end.

Two lessons. First, more training is not always better: the last 3,250 steps, almost
two thirds of the run, made the model *worse* at its real job. Second, this is why the
loop saves the best checkpoint and not the last one. `generate.py` loads the model from
step 1,750.

Why does it overfit so soon? One pass over the training data is only about 61 steps
(Exercise 5.3), so by step 1,750 the model has seen every part of the text about 29
times, and by step 5,000 about 82 times. A model with 10.7 million parameters and one
megabyte of text to learn from has more than enough capacity to start memorising. The
real cure is more data. Chapter 21 lists experiments you can try: more dropout, a
smaller model, stopping early.

The saved model is still a good one. A validation loss of 1.465 means that it gives the
correct next character a probability of $e^{-1.465} \approx 0.23$ on average, against
$1/65 \approx 0.015$ for guessing: it is as uncertain as someone choosing among about
4.3 equally likely characters.

### 13.7 If you have no GPU, or run out of memory

**No GPU.** Use the small configuration made for CPUs:

```bash
python train.py --config configs/tiny_cpu.yaml
```

It trains a model with 345 thousand parameters in about two minutes and reaches a
validation loss of about 1.84. The text is rougher, but everything in this guide can be
learned on it.

**"CUDA out of memory".** The main configuration needs about 4.6 GB of GPU memory. If
your card has less, lower `batch_size` first, then `block_size`. Measured on this
model: a batch size of 32 instead of 64 needs 2.4 GB, and a `block_size` of 128 instead
of 256 needs 2.1 GB.

```bash
python train.py --config configs/small.yaml --set train.batch_size=32
```

Chapter 20 has more on this error.

### Exercises

**13.1 (easy)** Using the formula in `get_lr`, compute the learning rate of the main
configuration at steps 50, 100 and 3775. (For the last one you need
$\cos(0.75\pi) \approx -0.7071$.)

**13.2 (easy)** A gradient has three components, $(2, -1, 2)$. What is its length? What
does it become after clipping with a maximum length of 1.0? And with a maximum of 5.0?

**13.3 (medium)** In `estimate_loss`, what would go wrong (a) without `model.eval()`,
(b) without the final `model.train()`, (c) without `@torch.no_grad()`? For each, say
whether the *numbers* would be wrong or only the *cost*.

**13.4 (medium)** Train the tiny configuration with a constant learning rate (set
`train.min_lr` equal to `train.learning_rate` and `train.warmup_steps=1`) and compare
the final validation loss with the scheduled run.

**13.5 (hard)** Add **early stopping** to `train.py`: stop the run when the validation
loss has not improved for `patience` evaluations in a row. Where does the counter go,
and what should happen to it when the loss improves?

### Check your understanding

1. Why does training begin with a warmup instead of the full learning rate?
2. What does `autocast` change, and what does it leave in float32?
3. What problem does the `GradScaler` solve?
4. Does gradient clipping change the direction of the update?
5. Why is the validation loss measured with `model.eval()`?
