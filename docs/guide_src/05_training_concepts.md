## Chapter 5: Training concepts

> **Key ideas**
>
> - The goal is not a low loss on the text the model trained on. It is a low loss on text
>   it has **never seen**. That is called **generalisation**.
> - We hold back 10% of the text as a **validation set** and never train on it, so that
>   we can measure generalisation honestly.
> - **Overfitting** means the model is memorising the training text: training loss keeps
>   falling while validation loss stops falling or rises.
> - Training uses small random **batches** of data, an **optimizer** (AdamW) that turns
>   gradients into updates, and a **learning rate** that sets the step size.
> - A **checkpoint** is the model saved to disk. We keep the one with the best validation
>   loss.

### 5.1 The real goal: generalisation

Imagine a student who prepares for an exam by memorising the answers to last year's
paper. On last year's paper they score 100%. On this year's, they fail. They have
*memorised*, not *learned*.

A neural network can do the same thing. With ten million parameters, our model is quite
capable of memorising long stretches of its one-megabyte training text. That would give
a wonderful training loss and a useless model, because the entire purpose of the model
is to handle text it has not seen.

So we need the equivalent of this year's exam paper: text that the model is never
trained on. Before training, the data is split in two:

- the **training set** (90%): the model learns from this;
- the **validation set** (10%): used only to measure the loss, never to compute a
  gradient.

The loss on the validation set, the **validation loss**, is our honest estimate of how
well the model **generalises** to new text. It is the number we actually care about.

Here is the split in the repository. The text has already been converted to one long
array of token IDs called `ids`:

{{code mini_gpt/data.py from="# The split is by position" to="train_ids, val_ids ="}}

For Tiny Shakespeare this gives 1,003,854 training tokens and 111,540 validation tokens.

> **Note**
>
> The split is by position: the first 90% of the file for training, the last 10% for
> validation. Why not pick random characters for validation? Because neighbouring
> pieces of text overlap heavily. If validation samples were sprinkled through the
> training text, the model would have seen almost all of each one, and the validation
> loss would look better than it deserves. Keeping one separate block is the honest way.

### 5.2 Overfitting and underfitting

During training we measure both losses at regular intervals and plot them. The shape of
the two curves tells you what is going on.

![The typical shape of the two loss curves. Both fall at first. If training goes on long enough, the validation loss turns upward while the training loss keeps falling.](figures/overfitting.png)

- **Both losses falling together.** The model is learning patterns that hold in general.
  Keep training.
- **Training loss falling, validation loss flat or rising.** This is **overfitting**: the
  model is now improving only by memorising details of the training text, and those
  details do not help (or actively hurt) on new text. The distance between the two
  curves is called the **generalisation gap**.
- **Both losses high and flat.** This is **underfitting**: the model is too small, or
  has not trained long enough, or the learning rate is wrong.

What to do about overfitting, roughly in order of effectiveness:

1. **Get more data.** The only complete cure. Tiny Shakespeare is small, which is why
   our model overfits if we let it.
2. **Stop at the right moment.** We save the model whenever the validation loss reaches
   a new minimum, so we always keep the best version (Section 5.7).
3. **Use a smaller model**, which has less capacity to memorise.
4. **Regularise**: make memorising harder. We use two regularisers, **dropout** and
   **weight decay**, described next.

**Dropout.** During training, each time data passes through a dropout layer, a random
fraction of the numbers (20% in our main configuration) is set to zero. The network can
therefore never rely on any single number being present, which forces it to spread
knowledge around and makes memorising exact sequences harder. Dropout must be switched
off when the model is *used*: `model.eval()` does that, and `model.train()` switches it
back on. You will see both calls in `train.py`.

**Weight decay** is covered with the optimizer in Section 5.5.

### 5.3 Batches

In Chapter 4 every step used all 200 data points. Our training set has a million tokens,
and one pass over all of it would take far too long to do thousands of times.

Instead, each step uses a small random sample of the data called a **batch** (or
mini-batch). The gradient computed from a batch is a noisy estimate of the gradient over
the whole dataset: not exactly right, but pointing in roughly the right direction, and
hundreds of times cheaper. Doing gradient descent with such estimates is called
**stochastic gradient descent** (stochastic means "involving randomness"). The noise
turns out to be mostly harmless, and even slightly helpful.

For our model, one training example is a sequence of `block_size` consecutive tokens,
and a batch is `batch_size` such sequences taken from random places in the text.

![One training sequence with block_size = 8. The targets are the inputs shifted by one position, so a single sequence provides eight prediction problems.](figures/batch_xy.png)

The function that builds a batch:

{{code mini_gpt/data.py symbol="get_batch"}}

Walk through it with `block_size = 8` and `batch_size = 4`:

- `torch.randint(len(data) - block_size, (batch_size,))` draws 4 random starting
  positions. The subtraction keeps a sequence from running off the end of the data.
- For each start `s`, the input is the 8 tokens `data[s : s + 8]` and the target is the
  8 tokens `data[s + 1 : s + 9]`, the same window moved one step to the right.
- `torch.stack` piles the 4 sequences on top of each other, giving two tensors of shape
  `(4, 8)`, which in the notation of this guide is `(B, T)`.

Two new **hyperparameters** have appeared. A hyperparameter is a number that *we*
choose, as opposed to a parameter, which the model learns.

- **`block_size`** (also called the **context length**) is the longest stretch of text
  the model can look at. Ours is 256 characters. Text further back is invisible to it.
- **`batch_size`** is how many sequences are processed per step. Larger batches give
  less noisy gradients but need more memory. Ours is 64. If you run out of GPU memory,
  this is the first number to lower.

In the main configuration, each step therefore processes $64 \times 256 = 16{,}384$
tokens.

> **Detail you can ignore for now**
>
> Many courses talk about **epochs**: one epoch is one complete pass through the training
> data. Because we draw random windows instead of walking through the data in order, we
> never complete an exact pass, so this project counts **steps** instead. Exercise 5.3
> converts between the two.

### 5.4 The learning rate

You met the learning rate in Chapter 4: it is the step size $\eta$ in
$\theta \leftarrow \theta - \eta \nabla L$, and the gradient-descent figure there showed
what goes wrong when it is too large. It is the single most important hyperparameter.

Useful facts:

- Sensible values for models like ours are between $10^{-4}$ and $10^{-3}$. Our main
  configuration uses $10^{-3}$ at its peak.
- **Too high**: the loss jumps around, rises, or becomes `nan` ("not a number", the sign
  that the numbers overflowed).
- **Too low**: the loss falls, but painfully slowly.
- The best learning rate is not constant during training. Large steps are good early,
  when the parameters are far from anywhere sensible; small steps are good late, for
  fine adjustments. So we change it as training proceeds, following a **schedule**.
  Chapter 13 covers ours (a warmup followed by a cosine-shaped decay).

### 5.5 Optimizers: from SGD to AdamW

An **optimizer** is the rule that turns gradients into parameter updates. Plain gradient
descent (called **SGD** in PyTorch) uses the simplest rule, $\theta \leftarrow \theta -
\eta g$, where $g$ is the gradient. It has two weaknesses.

**Weakness 1: every parameter gets the same step size.** In a real network, some
parameters have gradients thousands of times larger than others. A learning rate small
enough for the first kind is far too small for the second.

**Weakness 2: it has no memory.** Each step uses only the current, noisy gradient. In a
long narrow valley this produces the zig-zag path of the gradient-descent figure.

**Adam** fixes both. For every parameter separately, it keeps two running averages:

- $m$, the average of recent gradients. Using $m$ instead of the raw gradient is called
  **momentum**: like a heavy ball rolling downhill, the update keeps moving in the
  direction that has been consistently downhill, and the zig-zags average out.
- $v$, the average of recent *squared* gradients, which measures how large this
  parameter's gradients typically are.

The update is, in essence,

$$\theta \leftarrow \theta - \eta \, \frac{m}{\sqrt{v} + \epsilon}$$

Dividing by $\sqrt{v}$ is the clever part. A parameter whose gradients are typically
huge has its step divided by a huge number; one whose gradients are tiny has its step
divided by a tiny number. Every parameter ends up moving by roughly $\eta$ per step,
whatever the scale of its gradient. ($\epsilon$ is a very small number that prevents
division by zero.)

**AdamW** is Adam plus **weight decay** done correctly. Weight decay shrinks every
weight by a tiny fraction at each step, as if each weight were attached to zero by a
weak spring. A weight can only stay large if the data keeps pushing it to be large.
This discourages the extreme weights that memorisation tends to produce, so it is a
regulariser.

AdamW is the standard optimizer for transformers and the one we use:

{{code mini_gpt/utils.py symbol="configure_optimizer"}}

> **Detail you can ignore for now**
>
> The function splits the parameters into two groups so that weight decay applies only
> to weight *matrices* and not to biases or LayerNorm gains. This is a convention that
> works slightly better in practice. The `betas` are the two "how long is the memory"
> settings for $m$ and $v$.

### 5.6 Reproducibility: random seeds

Training involves randomness in several places: the initial weights, which batches are
drawn, which numbers dropout zeroes, and which tokens are sampled during generation. A
computer's "random" numbers actually come from a deterministic formula started from a
number called the **seed**. Fix the seed and you get the same sequence of random numbers,
and therefore the same training run, every time.

{{code mini_gpt/utils.py symbol="set_seed"}}

This matters for science. When you change one thing and the result changes, you want to
know that it changed because of your edit and not because of luck. (Chapter 15 comes
back to this: even with seeds, you must know how big "luck" is before trusting a small
difference.)

### 5.7 Checkpoints

A **checkpoint** is a file containing everything needed to use the model later: above
all the values of the parameters. In PyTorch, `model.state_dict()` returns a dictionary
mapping each parameter's name to its tensor, and `torch.save` writes a dictionary to
disk.

{{code mini_gpt/utils.py symbol="save_checkpoint"}}

In the training loop, a checkpoint is written every time the validation loss reaches a
new best value:

{{code train.py from="< best_val:" to="val_loss=best_val,"}}

Besides the weights, our checkpoint stores the configuration (so that the model can be
rebuilt with the right shape) and the tokenizer (so that text is converted to IDs in
exactly the same way as during training). A checkpoint without these is surprisingly
hard to use a month later.

### 5.8 One file to rule the hyperparameters

Every hyperparameter of a run lives in one YAML file. Here is the main one. You do not
need to understand every line yet; each is explained in the chapter where it matters.
Come back to this page as a reference.

{{code configs/small.yaml}}

The loader in `mini_gpt/config.py` turns this into Python objects, so the rest of the
code writes `cfg.model.n_layer` or `cfg.train.batch_size`. Any value can be overridden
from the command line without editing the file:

```bash
python train.py --config configs/small.yaml --set train.max_steps=500 model.n_head=4
```

### Exercises

**5.1 (easy)** A text has 250,000 tokens and `val_fraction` is 0.1. How many training
and validation tokens are there? Which part of the text does the validation set come
from?

**5.2 (easy)** For each situation, say whether it is overfitting, underfitting, or fine,
and suggest one thing to try. (a) train loss 0.4, validation loss 2.1. (b) train loss
2.4, validation loss 2.4, both flat for a long time. (c) train loss 1.25, validation
loss 1.48, both still slowly falling.

**5.3 (medium)** With `batch_size = 64`, `block_size = 256` and 1,003,854 training
tokens, about how many steps make one epoch (one pass over the data)? How many epochs
is a 5000-step run?

**5.4 (medium)** Train the tiny configuration for 300 steps three times, with learning
rates `1e-5`, `1e-3` and `1.0`. Predict the outcomes first.
`python train.py --config configs/tiny_cpu.yaml --set train.max_steps=300 train.learning_rate=1e-5 train.min_lr=1e-5`
(keep `min_lr` equal to `learning_rate` so the rate stays constant).

**5.5 (hard)** Implement gradient descent with momentum for $L(w) = (w - 3)^2$: keep a
velocity `v`, and at each step do `v = 0.9 * v + grad` followed by `w = w - lr * v`.
Starting from $w = 0$ with `lr = 0.1`, print $w$ for 30 steps. Compare with plain
gradient descent. What is different, and when would that be an advantage?

### Check your understanding

1. Why is the training loss not a trustworthy measure of how good the model is?
2. What do you see in the loss curves when a model overfits?
3. What does `block_size` control? What about `batch_size`?
4. In one sentence each: what do momentum and the division by $\sqrt{v}$ do in Adam?
5. Why does a checkpoint store the tokenizer as well as the weights?
