"""Learning Guide, Chapter 4: a neural network from scratch, in NumPy only.

A network with one hidden layer learns the curve y = sin(x). Every gradient is
derived by hand with the chain rule: there is no PyTorch and no autograd here.

Run it with:  python examples/ch04_tiny_network.py
"""

import numpy as np

rng = np.random.default_rng(0)

# --- 1. The data: 200 points on a sine wave ---
X = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)  # inputs,  shape (200, 1)
Y = np.sin(X)  # targets, shape (200, 1)

# --- 2. The parameters: two weight matrices and two bias vectors ---
H = 16  # number of hidden neurons
W1 = rng.normal(0.0, 1.0, size=(1, H))  # input  -> hidden, shape (1, 16)
b1 = np.zeros(H)  # one bias per hidden neuron
W2 = rng.normal(0.0, 0.1, size=(H, 1))  # hidden -> output, shape (16, 1)
b2 = np.zeros(1)


def forward(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the network's prediction for every input row."""
    z1 = X @ W1 + b1  # (200, 16)  weighted sum going into each hidden neuron
    h = np.tanh(z1)  # (200, 16)  activation: squash each value into (-1, 1)
    pred = h @ W2 + b2  # (200, 1)   the output is a weighted sum of the hidden neurons
    return z1, h, pred


def loss_fn(pred: np.ndarray, Y: np.ndarray) -> float:
    """Mean squared error: the average of (prediction - target)^2."""
    return float(((pred - Y) ** 2).mean())


# --- 3. Training: forward pass, backward pass, update ---
learning_rate = 0.05
history = []
for step in range(5001):
    # forward
    z1, h, pred = forward(X)
    loss = loss_fn(pred, Y)
    history.append(loss)

    # backward: apply the chain rule one layer at a time, from the loss back to the input
    n = len(X)
    d_pred = 2 * (pred - Y) / n  # dL/dpred          (200, 1)
    d_W2 = h.T @ d_pred  # dL/dW2            (16, 1)
    d_b2 = d_pred.sum(axis=0)  # dL/db2            (1,)
    d_h = d_pred @ W2.T  # dL/dh             (200, 16)
    d_z1 = d_h * (1 - h**2)  # dL/dz1, because tanh'(z) = 1 - tanh(z)^2
    d_W1 = X.T @ d_z1  # dL/dW1            (1, 16)
    d_b1 = d_z1.sum(axis=0)  # dL/db1            (16,)

    # update: step each parameter a little way AGAINST its gradient
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2

    if step % 1000 == 0:
        print(f"step {step:4d}  loss {loss:.5f}")

# --- 4. Check a gradient numerically ---
# The definition of a partial derivative: nudge ONE number, see how the loss changes.
eps = 1e-5
original = W2[3, 0]
W2[3, 0] = original + eps
loss_plus = loss_fn(forward(X)[2], Y)
W2[3, 0] = original - eps
loss_minus = loss_fn(forward(X)[2], Y)
W2[3, 0] = original
numerical = (loss_plus - loss_minus) / (2 * eps)

z1, h, pred = forward(X)
analytic = (h.T @ (2 * (pred - Y) / len(X)))[3, 0]
print(f"\ndL/dW2[3,0]  by the chain rule: {analytic:.8f}   by nudging: {numerical:.8f}")

# --- 5. Try the trained network ---
for x_test in [0.0, np.pi / 2, -1.0]:
    prediction = forward(np.array([[x_test]]))[2][0, 0]
    print(f"x = {x_test:6.3f}   network says {prediction:7.4f}   sin(x) = {np.sin(x_test):7.4f}")
