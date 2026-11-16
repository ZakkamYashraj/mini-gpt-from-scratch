## Chapter 20: Common errors and how to fix them

> **Key ideas**
>
> - Errors come in two kinds. **Loud** ones stop the program with a message; they are
>   the easy kind, because the message tells you where to look. **Silent** ones let the
>   program run and give worse results.
> - Most loud errors in this field are about **shapes**, **devices** or **dtypes**. The
>   cure is always the same: print the shapes.
> - For silent errors, work through a fixed checklist, starting with the overfit test.
> - Change one thing at a time, and make the problem small before trying to solve it.

The shape, index, dtype and device messages below were produced by actually making each mistake.
Use the chapter as a reference: when something breaks, find the message here.

### 20.1 Reading a traceback

When Python fails it prints a **traceback**: the chain of function calls that led to the
error. Beginners read it from the top and get lost in library code. Read it this way
instead:

1. Go to the **last line**. That is the actual error.
2. Then scan upward for the last line that mentions **a file of yours**
   (`model.py`, `train.py`), not a file inside `site-packages/torch/`. That is the line
   to fix.
3. Add `print(x.shape)` for every tensor on that line, immediately before it, and run
   again.

Step 3 solves most problems in this chapter in under a minute.

### 20.2 Shape errors

**Matrix multiplication with mismatched inner sizes.**

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x32 and 16x32)
```

The shape rule `(n, k) @ (k, m)` of Section 2.6 is violated: here the inner sizes are
32 and 16. Usual causes: a forgotten `.transpose(-2, -1)` on the keys in attention; a
`nn.Linear` created with the wrong input size; `head_size` confused with `n_embd`.
(PyTorch flattens the leading dimensions in this message, so `32x32` may be your
`(4, 8, 32)` tensor.)

**Element-wise operation on shapes that do not broadcast.**

```text
RuntimeError: The size of tensor a (32) must match the size of tensor b (8) at non-singleton dimension 2
```

You added or multiplied two tensors whose shapes, aligned on the right, clash
(Section 2.5). "Non-singleton" means that neither size is 1. Print both shapes; usually
one tensor is missing a dimension or has two dimensions swapped.

**Cross-entropy given three-dimensional logits.**

```text
RuntimeError: Expected target size [4, 65], got [4, 8]
```

`F.cross_entropy` was given logits of shape `(B, T, V)`. It wants `(B*T, V)` with
targets of shape `(B*T,)`. Fix: `logits.view(B * T, V)` and `targets.view(B * T)`, as in
Section 8.3.

**A reshape that does not fit.**

```text
RuntimeError: shape '[4, 65]' is invalid for input of size 2080
```

`view` and `reshape` can only rearrange the same number of elements. Here the tensor
has 2080 entries ($4 \times 8 \times 65$) and the requested shape has room for 260.
Check the arithmetic of your `view`.

**The dangerous one: a shape mistake that does not raise an error.** Broadcasting can
turn a mistake into a valid but wrong computation. Subtracting a `(B,)` tensor from a
`(B, 1)` tensor gives a `(B, B)` matrix, silently. If a loss or an intermediate tensor
has a shape you did not expect, stop and find out why. Shape comments on every line, as
in `mini_gpt/model.py`, are the defence.

### 20.3 Index, dtype and device errors

**A token ID outside the embedding table.**

```text
IndexError: index out of range in self
```

An ID is greater than or equal to the number of rows of an `nn.Embedding`. Two usual
causes: `vocab_size` in the model does not match the tokenizer (you changed tokenizer
but not the model, or loaded a checkpoint with the wrong tokenizer); or the sequence is
longer than `block_size`, so a position has no row in the positional table. On a GPU
the same mistake appears as a much more alarming message, "CUDA error: device-side
assert triggered"; if you see that, re-run on the CPU to get the readable version.

The repository turns the second cause into a clear message of its own:

```text
ValueError: Sequence length 9 exceeds block_size 8
```

**Floating-point numbers used as indices.**

```text
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)
```

Token IDs must be integers. Create them with `dtype=torch.long`, or convert with
`.long()`. This typically happens after loading IDs from NumPy as the wrong type.

**Tensors on different devices.**

```text
RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0
```

The model is on the GPU and something you fed it is not (or the reverse). Check three
places: the batch (`x.to(device)`, `y.to(device)`), the prompt in generation, and any
tensor *created inside* `forward`. That last one is why the model writes
`torch.arange(T, device=idx.device)`: a tensor created without `device=` is born on the
CPU.

### 20.4 Out of memory

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 8.00 GiB ...
```

The GPU's memory is full. In order of how often each helps:

1. **Lower `batch_size`.** Memory use is nearly proportional to it. Try half.
2. **Lower `block_size`.** Attention memory grows with its *square*.
3. **Make sure mixed precision is on** (`amp: true`).
4. **Close other programs using the GPU** (a second training run, a notebook that still
   holds a model). `nvidia-smi` in a terminal shows what is using the memory.
5. **Check evaluation and generation run under `torch.no_grad()`.** Without it PyTorch
   keeps the whole computation graph in memory.
6. Use a smaller model.

A subtle version: memory that grows slowly until it runs out. The classic cause is
keeping the loss *tensor* in a list (`losses.append(loss)`), which keeps its entire
computation graph alive. Store the number instead: `losses.append(loss.item())`.

### 20.5 The loss is `nan`

`nan` means "not a number". It appears when a calculation has no defined answer, and it
spreads: any arithmetic involving `nan` gives `nan`, so one bad value poisons every
parameter at the next update.

```python
>>> x = torch.tensor([1e20])
>>> x * x                     # too large for float32
tensor([inf])
>>> torch.log(torch.tensor(0.0))
tensor(-inf)
>>> torch.tensor(float("inf")) - torch.tensor(float("inf"))
tensor(nan)
```

The usual causes, most likely first:

1. **The learning rate is too high.** The weights grow, the logits overflow to `inf`,
   and `inf - inf` is `nan`. (Exercise 4.4 produced exactly this.) Divide it by 10.
2. **No gradient clipping**, and one bad batch produced a huge step.
3. **Taking `log` of a probability that is exactly 0**, in code where you compute the
   loss by hand. Use `F.cross_entropy` on the *logits*; it is written to avoid this.
4. **A whole row of attention scores masked to $-\infty$.** The softmax of a row
   containing only $-\infty$ is `nan`. With a correct causal mask this cannot happen,
   because every token may attend at least to itself. If you write your own mask, check
   that the diagonal is allowed.
5. **Bad data**: an empty batch, or `nan` already present in the input.

To find where it starts, add `assert torch.isfinite(loss), step` after the loss is
computed, then print the largest absolute value of each intermediate tensor on the step
before.

### 20.6 The loss does not go down

No error, but the loss stays near its starting value or improves far too slowly. Work
through this list in order. Each step takes minutes and rules out a whole family of
causes.

1. **Is the starting loss right?** It should be close to $\ln(V)$: 4.17 for 65
   characters. Much higher means bad initialisation; a different number may mean the
   vocabulary size is not what you think.
2. **Can the model overfit one batch?** Fix one small batch and train on it alone
   (Section 17.7). If the loss does not approach 0, the bug is in the model or the loop,
   not in the data or the hyperparameters.
3. **Are the targets right?** Decode `x[0]` and `y[0]` and read them. `y` must be `x`
   shifted by one character. A shift by zero makes the task trivial (loss near 0); a
   shift by two makes it nearly impossible.
4. **Is every step of the loop there?** A missing `optimizer.step()` means nothing ever
   changes. A missing `zero_grad()` means the gradients pile up. `loss.backward()`
   called on the wrong tensor means nothing useful is computed.
5. **Are the parameters in the optimizer?** A layer kept in a plain Python list, not an
   `nn.ModuleList`, is invisible to `model.parameters()` and never trains. Compare the
   parameter count with what you expect.
6. **Is the learning rate sensible?** Try `1e-3` and `3e-4`. Exercise 5.4 showed both
   failure modes.
7. **Is the model in training mode?** After an evaluation, `model.train()` must be
   called again.

### 20.7 Suspicious results

**The loss is far too good** (well below 1.0 on Shakespeare within a few hundred steps).
Information about the answer is leaking into the input. The causal mask is missing or
wrong (Section 10.2), or the targets are not shifted.

**Training loss falls, validation loss rises.** Not a bug: overfitting (Section 5.2).
Use the best checkpoint, add dropout, reduce the model, or get more data.

**Validation loss is lower than training loss.** Usually also not a bug. Dropout is
active when the training loss is computed inside the loop and off during evaluation, so
the model looks worse while training than it is. The repository avoids the confusion by
measuring *both* losses in evaluation mode.

**Generated text is garbage although the loss was good.** Check, in order: that
`model.eval()` was called; that the checkpoint was loaded into a model of the same
shape; and that the tokenizer is the one used in training. A model trained with one
vocabulary and decoded with another produces fluent nonsense.

**Results change from run to run.** Set the seed (Section 5.6). Some GPU operations are
still not perfectly deterministic, so tiny differences can remain; that is normal, and
it is one reason Chapter 15 repeats experiments with more than one seed.

### 20.8 Environment problems

**`ModuleNotFoundError: No module named 'mini_gpt'`.** You ran a script from the wrong
folder. Run everything from the repository root: `python train.py ...`, not
`cd scripts` followed by `python plot_loss.py`.

**`ModuleNotFoundError: No module named 'torch'`.** The virtual environment is not
active. Activate it (`.venv\Scripts\activate` on Windows) and try again.

**`torch.cuda.is_available()` is `False` on a machine with an NVIDIA card.** You have
the CPU-only build of PyTorch. Reinstall with the CUDA index URL from pytorch.org, and
check that the NVIDIA driver is up to date.

**`UnicodeEncodeError` when printing on Windows.** The console's default encoding cannot
show some characters. The scripts call `fix_console_encoding()` from `mini_gpt/utils.py`
to prevent this; do the same in your own.

**`OSError: [WinError 4551] An Application Control policy has blocked this file` when
importing `torch`.** Windows 11's *Smart App Control* refused to load one of PyTorch's
library files because it did not recognise that particular build. This happened while
this project was being made: the newest CUDA build was blocked, and a slightly older
official build of PyTorch loaded without complaint. So the first thing to try is another
official version from pytorch.org. Smart App Control is a security feature of your
computer; switching it off is a decision to take deliberately, not as a quick fix, and
Windows does not let you switch it back on afterwards without reinstalling.

### 20.9 A method for the bugs that are not in this chapter

1. **Make it small.** Tiny model, batch of 2, sequence of 8, CPU. Bugs that survive are
   much easier to see, and each attempt takes seconds.
2. **Reproduce it reliably.** Fix the seed. A bug that appears every time is half
   solved.
3. **Look at the actual values.** Print shapes, then minimum, maximum and mean. Decode
   token IDs back to text and read them.
4. **Change one thing at a time**, and predict the outcome before you run. A surprise
   is information.
5. **Go back to the last version that worked** (`git diff` shows exactly what changed
   since) and add the changes back one by one.
6. **Explain it aloud**, line by line, to anyone or to nobody. It is remarkable how
   often the bug appears in the middle of a sentence.

### Exercises

**20.1 (easy)** For each message, name the likely mistake and the fix:
(a) `mat1 and mat2 shapes cannot be multiplied (512x64 and 384x64)`;
(b) `Expected all tensors to be on the same device`;
(c) `index out of range in self`.

**20.2 (medium)** Introduce each of these bugs into a copy of the training loop, one at
a time, and record what you observe: remove `optimizer.zero_grad()`; remove
`optimizer.step()`; use `y = x` (no shift) as targets. Which are loud and which are
silent?

**20.3 (medium)** Build a `GPT` whose blocks are stored in a plain Python list instead
of an `nn.ModuleList`. Compare `num_params()` with the correct model. What would happen
if you trained it?

**20.4 (hard)** A friend's model reaches a training loss of 0.02 after 400 steps and
generates only newline characters. Without seeing the code, list the two most likely
causes and the test you would run to tell them apart.

### Check your understanding

1. Which line of a traceback should you read first, and which second?
2. What is the first thing to try for "CUDA out of memory"?
3. Name two common causes of a `nan` loss.
4. The loss will not go down. What is the first check, and the second?
5. Why is a loss that is "too good" a warning sign?
