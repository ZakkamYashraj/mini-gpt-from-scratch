## Chapter 3: PyTorch basics

> **Key ideas**
>
> - A PyTorch **tensor** is a NumPy-style array with two extra abilities: it can live on
>   a **GPU**, and it can remember how it was computed.
> - **Autograd** uses that memory to compute derivatives automatically. You write the
>   forward computation; PyTorch gives you every partial derivative.
> - An **`nn.Module`** is a class that bundles parameters with the code that uses them.
>   Every piece of our model is one.
> - A **training loop** is five lines: forward, loss, zero the gradients, backward, step.
>   The loop that trains our GPT is the same five lines.

### 3.1 What PyTorch gives us

PyTorch is a library for building and training neural networks. We use four things
from it:

1. **Tensors**: arrays, like NumPy's, that can also be stored and computed on a GPU.
2. **Autograd**: automatic computation of derivatives.
3. **`torch.nn`**: basic building blocks (a table of learnable vectors, a learnable
   matrix multiplication) and the `nn.Module` class to organise them.
4. **Optimizers**: the code that updates parameters using the derivatives.

We deliberately do **not** use PyTorch's ready-made transformer or attention layers.
Those we build ourselves; that is the point of the project.

The code for this chapter is `examples/ch03_pytorch_basics.py`.

### 3.2 Tensors

{{code examples/ch03_pytorch_basics.py section="1."}}

{{output examples/ch03_pytorch_basics.py from="1. Tensors" to="ids.dtype"}}

If you understood Chapter 2 you already understand tensors. The differences are small:

- PyTorch says **`dim`** where NumPy says `axis`. `t.sum(dim=1)` is NumPy's
  `t.sum(axis=1)`.
- Every tensor has a **dtype**, the kind of number it holds. Two matter to us.
  `float32` (a decimal number stored in 32 bits) is used for everything the model
  computes. `int64` (a whole number) is used for token IDs. Mixing them up is a common
  error: an embedding table, for instance, insists on being indexed with integers.
- A tensor with a single number in it can be turned into an ordinary Python number with
  `.item()`. You will see `loss.item()` wherever a loss is printed.

### 3.3 Devices: CPU and GPU

{{code examples/ch03_pytorch_basics.py section="2."}}

{{output examples/ch03_pytorch_basics.py from="2. Devices" to="running on"}}

A **CPU** has a few very capable cores. A **GPU** (graphics card) has thousands of simple
ones. Matrix multiplication is made of a huge number of independent multiply-and-add
operations, which is exactly the kind of work that can be shared out among thousands of
simple cores. For our model the GPU is roughly 20 to 50 times faster than the CPU.

In PyTorch every tensor lives on a **device**: `"cpu"` or `"cuda"` (CUDA is NVIDIA's
name for its GPU computing system). `tensor.to(device)` returns a copy on that device.
There is one rule: **all the tensors taking part in an operation must be on the same
device**. Break it and you get the error *"Expected all tensors to be on the same
device"*, which Chapter 20 shows how to fix.

The repository chooses the device once, with this function, and passes it around:

{{code mini_gpt/utils.py symbol="get_device"}}

Because of the `"auto"` option, the same code runs on your laptop's GPU and on a machine
with no GPU at all.

### 3.4 Autograd: derivatives for free

This is the feature that makes deep learning practical. Recall from calculus: for a
function of several variables, the **partial derivative** with respect to one variable
tells you how fast the function changes when you nudge that variable and hold the others
still.

Take $f(x, y) = x^2 y + 3y$. By hand:

$$\frac{\partial f}{\partial x} = 2xy \qquad\qquad \frac{\partial f}{\partial y} = x^2 + 3$$

At the point $x = 2$, $y = 5$ these are $20$ and $7$. Now let PyTorch do it:

{{code examples/ch03_pytorch_basics.py section="3."}}

{{output examples/ch03_pytorch_basics.py from="3. Autograd" to="df/dy"}}

What happened, step by step:

1. `requires_grad=True` tells PyTorch "I will want derivatives with respect to this
   tensor".
2. While `f` is computed, PyTorch quietly records every operation in a
   **computation graph** (the figure below): which inputs each result came from and
   which operation produced it.
3. `f.backward()` walks that graph from `f` back to the inputs, applying the chain rule
   at every step. This backward walk is called **backpropagation**; Chapter 4 does it by
   hand so that it holds no mystery.
4. The results are stored in the `.grad` attribute of each input.

![The computation graph PyTorch records for f = x²y + 3y. The forward pass (black) computes values left to right; the backward pass (orange) carries derivatives right to left.](figures/comp_graph.png)

The collection of all the partial derivatives of a function is called its **gradient**.
For our $f$ the gradient at $(2, 5)$ is the pair $(20, 7)$. In training, the function
is the loss, the variables are the model's ten million parameters, and the gradient is
ten million numbers: one partial derivative per parameter. Autograd computes all of them
with one call to `backward()`.

Two details about gradients that you need for the training loop:

- **Gradients accumulate.** Calling `backward()` *adds* to whatever is already in
  `.grad`. So before each new backward pass the old gradients must be cleared, which is
  what `optimizer.zero_grad()` does. Forgetting it is a classic bug (exercise 3.3).
- **Recording costs time and memory.** When you only want predictions, not derivatives
  (for example when generating text or measuring the validation loss), you switch the
  recording off with `torch.no_grad()`.

### 3.5 `nn.Module`: organising parameters

A model has many parameters, and we need to find them all (to update them), move them
all to the GPU, and save them all to disk. `nn.Module` is the base class that handles
this bookkeeping.

{{code examples/ch03_pytorch_basics.py section="4."}}

{{output examples/ch03_pytorch_basics.py from="4. nn.Module" to="parameters:"}}

The pattern is always the same, and every class in `mini_gpt/model.py` follows it:

- In `__init__`, call `super().__init__()` and create the parameters. Wrapping a tensor
  in `nn.Parameter` registers it as "to be learned" and switches on `requires_grad`.
- In `forward`, write the computation.
- To use the model, **call it like a function**: `model(xs)`. PyTorch then runs
  `forward` for you. (Do not call `model.forward(xs)` directly.)

Modules can contain other modules, and `model.parameters()` finds the parameters of all
of them, however deeply nested. That is how a GPT made of dozens of sub-modules can be
handed to an optimizer in one line.

Two ready-made modules appear throughout this guide. You could write both in three
lines; they are used only because they are convenient:

- `nn.Linear(n_in, n_out)` holds a weight matrix `W` of shape `(n_out, n_in)` and
  (optionally) a bias vector `b`, and computes `x @ W.T + b`. This is the "machine that
  transforms vectors" of Section 2.6, with learnable entries.
- `nn.Embedding(n, d)` holds a table of shape `(n, d)` and, given integer IDs, returns
  the corresponding rows. Chapter 7 is about this one.

### 3.6 The training loop

Here is a complete training run. The data are 50 points scattered around the line
$y = 2x + 1$. The model is `Line` from above, starting at $w = 0$, $b = 0$. The loop
should discover $w \approx 2$ and $b \approx 1$ by itself.

{{code examples/ch03_pytorch_basics.py section="5."}}

{{output examples/ch03_pytorch_basics.py from="5. Training loop" to="step 200"}}

The loss falls and `w`, `b` move to the right values. The five numbered lines inside the
loop are the most important five lines in this guide:

1. **Forward pass.** Run the model on the inputs to get predictions.
2. **Loss.** Measure how wrong the predictions are, as a single number. Here it is the
   mean squared error; for the language model it will be the cross-entropy of Chapter 1.
3. **Zero the gradients** left over from the previous step.
4. **Backward pass.** Compute the partial derivative of the loss with respect to every
   parameter.
5. **Step.** Change each parameter a little, in the direction that lowers the loss. For
   plain gradient descent this is `p = p - lr * p.grad`, where `lr`, the **learning
   rate**, sets the size of the step. Chapter 4 explains why this works.

Now look at the real training loop of this repository. Ignore the lines you do not yet
recognise and find the five steps:

{{code train.py from="# 1) Get a batch" to="scaler.update()"}}

The forward pass and loss are in step 2, zeroing and the backward pass in step 3, and the
update in step 5. The extra pieces (`autocast`, `scaler`, gradient clipping) are
refinements that Chapter 13 explains. The skeleton is the loop you just ran.

### Exercises

**3.1 (easy)** Create a tensor of shape `(2, 3, 4)` filled with random numbers
(`torch.randn`). Print its shape, its dtype, and the shape of its sum over `dim=1`.
Predict each answer before running.

**3.2 (easy)** Let $g(a, b) = ab + \sin(a)$. Compute $\partial g/\partial a$ and
$\partial g/\partial b$ by hand. Then check them with autograd at $a = 1$, $b = 2$.

**3.3 (medium)** In the autograd example, call the lines that compute `f` and
`f.backward()` a second time *without* clearing the gradients, then print `x.grad`.
Explain the number you see. Then fix it with `x.grad.zero_()`.

**3.4 (medium)** Change the training example so that the data follow
$y = 3x^2 - 2x + 1$ and the model is a `Quadratic` module with three parameters. Does it
recover 3, -2 and 1?

**3.5 (hard)** In the training loop, delete the optimizer and do the update yourself:
after `loss.backward()`, loop over `model.parameters()` and apply
`p -= lr * p.grad` inside a `with torch.no_grad():` block, then zero each gradient.
Confirm you get the same result. Why is the `no_grad` block needed?

### Check your understanding

1. Name the two abilities a PyTorch tensor has that a NumPy array lacks.
2. What does `loss.backward()` compute, and where does it put the results?
3. Why must gradients be zeroed in every iteration?
4. What is the difference between `nn.Parameter(torch.zeros(3))` and `torch.zeros(3)`
   inside a module?
5. You get the error "Expected all tensors to be on the same device". What happened?
