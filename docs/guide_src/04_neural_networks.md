## Chapter 4: Neural networks from scratch

> **Key ideas**
>
> - A **neuron** computes a weighted sum of its inputs, adds a **bias**, and passes the
>   result through a simple curved function called an **activation function**.
> - A **layer** is many neurons side by side, which is one matrix multiplication.
>   A **neural network** is layers applied one after another.
> - A **loss function** turns "how wrong is the network?" into one number.
> - **Gradient descent** lowers the loss by repeatedly moving every parameter a small
>   step *against* its partial derivative.
> - **Backpropagation** is the chain rule, applied layer by layer from the loss backwards,
>   to get all those partial derivatives efficiently.

In this chapter we build and train a neural network using only NumPy, computing every
derivative ourselves. You will never need to do this again, because PyTorch's autograd
does it for you. But having done it once, nothing about training will feel like magic.

### 4.1 One neuron

A **neuron** takes several numbers in and gives one number out:

$$\text{output} = \phi(w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b)$$

- $x_1, \dots, x_n$ are the inputs.
- $w_1, \dots, w_n$ are the **weights**: one per input, saying how much that input
  matters and in which direction.
- $b$ is the **bias**: a constant added on, which shifts the result up or down.
- $\phi$ is the **activation function**, a fixed curve applied at the end (Section 4.3).

The weights and the bias are the neuron's parameters: the numbers that training adjusts.

**A worked example.** Inputs $x = (2, -1)$, weights $w = (0.5, 1.5)$, bias $b = 0.5$,
activation $\tanh$:

$$w \cdot x + b = 0.5 \times 2 + 1.5 \times (-1) + 0.5 = 0 \qquad \tanh(0) = 0$$

Change the first input to 4 and the sum becomes 1, with output $\tanh(1) \approx 0.76$.

An analogy: think of a neuron as a tiny detector. The weights describe the pattern it is
looking for; the weighted sum is large when the input matches that pattern; the
activation turns the match score into a signal. One neuron can do very little. The power
comes from numbers.

### 4.2 A layer is a matrix multiplication

Put $H$ neurons next to each other, all reading the same inputs. Each has its own
weights, so the weights of the whole **layer** form a matrix. Computing all $H$ weighted
sums at once is one matrix multiplication:

$$h = \phi(x W + b)$$

If the input $x$ has $n$ numbers and the layer has $H$ neurons, then $W$ has shape
`(n, H)`, $b$ has shape `(H,)`, and by the shape rule of Section 2.6 the result has
shape `(H,)`. And if we feed in 200 inputs at once, as a matrix of shape `(200, n)`, we
get all 200 results as a `(200, H)` matrix with no extra code. Processing many inputs
together like this is called **batching**.

A **neural network** is several layers in a row: the output of one is the input of the
next. Layers between the input and the output are called **hidden layers**.

![The network of this chapter: one input, a hidden layer of 16 tanh neurons, one output. Every arrow is a weight. There are 16 + 16 + 16 + 1 = 49 parameters.](figures/network.png)

### 4.3 Why the activation function is essential

Suppose we left it out. Two layers would compute $(x W_1) W_2$. But matrix
multiplication is associative:

$$(x W_1) W_2 = x (W_1 W_2) = x W'$$

Two layers without activations are exactly equivalent to **one** layer with weight
matrix $W' = W_1 W_2$. The same holds for a hundred layers. A network without
activations can only ever compute straight-line relationships, however deep it is.

The activation function breaks this by bending the output of each layer. With bends in
between, stacked layers can build up any curved shape. Three activations are worth
knowing:

![Three activation functions. ReLU is the simplest; tanh is used in this chapter's network; GELU is the one inside our GPT.](figures/activations.png)

- **tanh** squashes any number into the range $(-1, 1)$. We use it in this chapter
  because its derivative is simple: $\tanh'(z) = 1 - \tanh^2(z)$.
- **ReLU** is $\max(0, z)$: pass positive numbers through, replace negative numbers by 0.
- **GELU** is a smooth version of ReLU. It is what GPT-2 used and what our model uses
  (`nn.GELU()` in `mini_gpt/model.py`). You do not need its formula; think "ReLU with the
  corner rounded off".

### 4.4 The task and the network

We will teach a network the function $y = \sin(x)$ for $x$ between $-\pi$ and $\pi$. Of
course nobody needs a neural network for that. The point is that the network starts
knowing nothing about sines and must discover the curve from 200 example points.

The whole program is `examples/ch04_tiny_network.py`. Here are the data and the
parameters:

{{code examples/ch04_tiny_network.py from="# --- 1. The data" to="b2 = np.zeros(1)"}}

The parameters start as small random numbers (drawn by `rng.normal`). They must not all
start at zero: if every hidden neuron began identical, every one would receive the same
gradient and they would stay identical forever, which wastes all but one of them.

The forward computation is two layers, written exactly as in Section 4.2:

{{code examples/ch04_tiny_network.py symbol="forward"}}

Follow the shapes: `X` is `(200, 1)` and `W1` is `(1, 16)`, so `z1` is `(200, 16)`.
Adding `b1`, of shape `(16,)`, uses broadcasting. `h @ W2` is `(200, 16) @ (16, 1)`,
giving `(200, 1)`: one prediction per input.

### 4.5 The loss function

To train, we need one number measuring how wrong the predictions are. For predicting
a number, the usual choice is the **mean squared error** (MSE): the average of
$(\text{prediction} - \text{target})^2$.

{{code examples/ch04_tiny_network.py symbol="loss_fn"}}

Squaring makes every error positive and punishes big errors much more than small ones.

For a language model the output is not a number but a choice among tokens, and the loss
is the **cross-entropy** from Chapter 1. Now that you know softmax, here is the complete
recipe:

1. the model outputs one **logit** per token, for example `[2.0, 1.0, 0.1]`;
2. softmax turns the logits into probabilities: `[0.659, 0.242, 0.099]`;
3. the loss is $-\ln$ of the probability of the correct token. If token 0 was correct,
   the loss is $-\ln(0.659) \approx 0.417$; if token 2 was correct, it is
   $-\ln(0.099) \approx 2.31$.

In PyTorch all three steps are one function, `F.cross_entropy(logits, targets)`. You will
see it in the model's `forward` method.

### 4.6 Gradient descent

The loss depends on the parameters. Change any weight and the predictions change, so the
loss changes. Write $L(\theta)$ for the loss as a function of all the parameters
$\theta$ (here 49 numbers; for our GPT, ten million).

Training is the search for parameters that make $L$ small. The picture to hold in your
head is a landscape: every possible setting of the parameters is a location, and the
loss is the height of the ground there. We are standing somewhere random and want to
walk downhill. We cannot see the landscape (it has 49 dimensions), but we *can* measure
the slope under our feet.

From multivariable calculus: the **gradient** $\nabla L$ is the vector of all partial
derivatives,

$$\nabla L = \left( \frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \dots \right)$$

and it points in the direction in which $L$ **increases** fastest. So to decrease $L$,
step the opposite way:

$$\theta \leftarrow \theta - \eta \, \nabla L$$

This update rule is **gradient descent**. The small positive number $\eta$ ("eta") is the
**learning rate**: it sets the length of each step.

**A worked example in one dimension.** Let $L(w) = (w - 3)^2$. The minimum is obviously
at $w = 3$. The derivative is $L'(w) = 2(w - 3)$. Start at $w = 0$ with $\eta = 0.1$:

| step | $w$ | $L'(w) = 2(w-3)$ | new $w = w - 0.1 \times L'(w)$ |
|---|---|---|---|
| 1 | 0.000 | -6.000 | 0.600 |
| 2 | 0.600 | -4.800 | 1.080 |
| 3 | 1.080 | -3.840 | 1.464 |
| 4 | 1.464 | -3.072 | 1.771 |

The derivative is negative, meaning "the loss goes down if $w$ goes up", so $w$
increases. Each step covers a fifth of the remaining distance to 3. The steps shrink
automatically as the slope flattens near the minimum.

![Gradient descent on a loss with two parameters. The rings are contour lines (points of equal loss). With a sensible learning rate the path walks steadily to the minimum; too large, and it bounces across the valley; much too large, and it flies away.](figures/gradient_descent.png)

The learning rate matters a great deal, as the figure shows. Too small and training
takes forever. Too large and each step overshoots the valley and lands higher up on the
other side, so the loss rises instead of falling. Choosing it is discussed in Chapter 5.

### 4.7 Backpropagation: the chain rule, organised

Gradient descent needs $\partial L / \partial \theta$ for every parameter. Our loss is a
chain of simple steps:

$$X \xrightarrow{\;W_1,\, b_1\;} z_1 \xrightarrow{\;\tanh\;} h \xrightarrow{\;W_2,\, b_2\;} \text{pred} \xrightarrow{\;\text{MSE}\;} L$$

The **chain rule** says how derivatives pass through a chain. If $L$ depends on $u$, and
$u$ depends on $v$, then

$$\frac{\partial L}{\partial v} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial v}$$

**Backpropagation** is nothing more than applying this rule repeatedly, starting at the
loss and moving backwards one step at a time. At each step we take the derivative
arriving from the right ("how does the loss change if my *output* changes?") and turn it
into the derivative for the left ("how does the loss change if my *input* changes?").

Here is the code. Every line is one application of the chain rule:

{{code examples/ch04_tiny_network.py from="# backward: apply the chain rule" to="d_b1 = d_z1.sum(axis=0)"}}

Line by line:

- **`d_pred`**. The loss is the mean of $(\text{pred} - Y)^2$ over $n$ points. The
  derivative of a square is twice the thing squared, and each point contributes $1/n$ of
  the mean: $\partial L/\partial \text{pred} = 2(\text{pred} - Y)/n$.
- **`d_W2`**. The step was `pred = h @ W2 + b2`. For a matrix multiplication there is a
  rule worth memorising: **if `out = A @ W`, then `dW = A.T @ d_out` and
  `dA = d_out @ W.T`.** You can check it with shapes alone: `h.T` is `(16, 200)`,
  `d_pred` is `(200, 1)`, and the product is `(16, 1)`, the shape of `W2`. A gradient
  always has the same shape as the thing it is the gradient of.
- **`d_b2`**. The bias was added to every one of the 200 rows, so it influenced the loss
  through all of them; its gradient is the sum over the rows.
- **`d_h`**. The other half of the matrix rule: the gradient for the *input* of the
  multiplication.
- **`d_z1`**. The step was `h = tanh(z1)`, applied element by element, so we multiply
  element by element by the derivative of tanh, which is $1 - \tanh^2(z_1) = 1 - h^2$.
- **`d_W1`, `d_b1`**. The same two rules again, one layer earlier.

Notice the pattern: the backward pass visits the same steps as the forward pass in
reverse order, and each step needs only its own local derivative. That is why it scales
to networks of any depth, and why a program (autograd) can do it for us.

> **Note**
>
> Where does the matrix rule come from? Take the smallest case: one input row
> $a = (a_1, a_2)$ and one output $o = a_1 w_1 + a_2 w_2$. Then
> $\partial o/\partial w_1 = a_1$, so by the chain rule
> $\partial L/\partial w_1 = a_1 \cdot \partial L/\partial o$. "Input times the gradient
> arriving at the output" is all that `A.T @ d_out` computes, for every weight at once.

With the gradients in hand, the update is gradient descent, exactly the formula of
Section 4.6:

{{code examples/ch04_tiny_network.py from="# update: step each parameter" to="b2 -= learning_rate * d_b2"}}

### 4.8 Checking a gradient with the definition

How do we know the backward pass is right? Go back to the *definition* of a partial
derivative: change one parameter by a tiny amount $\varepsilon$, keep everything else
fixed, and see how much the loss moves.

$$\frac{\partial L}{\partial w} \approx \frac{L(w + \varepsilon) - L(w - \varepsilon)}{2\varepsilon}$$

{{code examples/ch04_tiny_network.py section="4."}}

This is slow (two full forward passes for a single parameter) so it is useless for
training, but it is a perfect test. The chain-rule value and the nudged value should
agree to many decimal places.

### 4.9 Running it

{{output examples/ch04_tiny_network.py}}

The loss falls steadily; the two ways of computing the gradient agree to eight decimal
places; and the trained network's answers are close to the true sine values (within
about 0.06), including at points that are not exactly in the training set. It is not
perfect: plain gradient descent is slow, and more steps would bring it closer.

![Left: the network's curve at three moments of training, against the true sine curve. Right: the loss during training (note the logarithmic vertical axis).](figures/tiny_network_fit.png)

Think about what just happened. Nobody told the network anything about sine. Forty-nine
random numbers were nudged five thousand times, each time in the direction that reduced
the error a little, and the result is a function that traces a sine wave. The GPT is
trained by the very same procedure. Only the function being adjusted is bigger.

### 4.10 The same thing in PyTorch

Everything in Section 4.7 is what `loss.backward()` does for you. Here is this chapter's
network again, as scratch code to try:

```python
import math
import torch
import torch.nn as nn

X = torch.linspace(-math.pi, math.pi, 200).reshape(-1, 1)
Y = torch.sin(X)

model = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for step in range(5001):
    loss = ((model(X) - Y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()        # all of Section 4.7, automatically
    optimizer.step()       # the update of Section 4.6
```

(`nn.Sequential` simply applies the listed layers in order.) From here on we let
autograd compute gradients, and we concentrate on designing the function.

### Exercises

**4.1 (easy)** A neuron has weights $(1, -2, 0.5)$, bias $1$, and uses ReLU. What does it
output for the input $(3, 1, 4)$? And for $(0, 2, 0)$?

**4.2 (easy)** Continue the table of Section 4.6 for two more steps. Then redo the first
three steps with $\eta = 1.1$. What happens, and why?

**4.3 (medium)** Compute by hand the cross-entropy loss for the logits `[1.0, 1.0, 1.0]`
with correct token 0, and for the logits `[10.0, 0.0, 0.0]` with correct token 1. What do
the two answers tell you about being *unsure* versus being *confidently wrong*?

**4.4 (medium)** In `examples/ch04_tiny_network.py`, change the number of hidden neurons
`H` to 2, then to 64. How does the final loss change? Explain the result for `H = 2` by
thinking about how many "bends" two tanh neurons can provide. (If the loss turns into
`nan`, lower the learning rate and work out why that helps.)

**4.5 (medium)** Remove the activation: replace `h = np.tanh(z1)` by `h = z1` and fix the
backward pass to match. What is the best the network can now do, and what does its
prediction curve look like?

**4.6 (hard)** Verify the backward pass against autograd. Rewrite the forward pass with
PyTorch tensors (use `torch.tensor(W1, requires_grad=True)` and so on, starting from the
same NumPy values), call `loss.backward()`, and compare `W1_t.grad` with `d_W1`.

### Check your understanding

1. What are the parameters of a single neuron with three inputs? How many are there?
2. Why can a network without activation functions not learn the sine curve?
3. In the update $\theta \leftarrow \theta - \eta \nabla L$, why the minus sign?
4. What does backpropagation compute, and what rule from calculus is it built on?
5. The gradient of the loss with respect to a weight matrix of shape `(16, 1)` has what
   shape?
