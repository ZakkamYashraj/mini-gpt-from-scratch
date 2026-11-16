"""Learning Guide, Chapter 3: PyTorch basics.

Run it with:  python examples/ch03_pytorch_basics.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)  # make the "random" numbers the same on every run

# --- 1. Tensors ---
print("1. Tensors")
t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("t.shape:", t.shape, " t.dtype:", t.dtype)
print("t @ t.T =\n", t @ t.T)  # same operators as NumPy
print("t.sum(dim=1) =", t.sum(dim=1), " (PyTorch says dim where NumPy says axis)")
ids = torch.tensor([3, 1, 4])
print("ids.dtype:", ids.dtype, " (whole numbers become int64, used for token IDs)")

# --- 2. Devices ---
print("\n2. Devices")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_dev = t.to(device)  # copy the tensor to the GPU if there is one
print("running on:", device, " t_dev.device:", t_dev.device)

# --- 3. Autograd: automatic derivatives ---
print("\n3. Autograd")
# f(x, y) = x^2 * y + 3y.   By hand: df/dx = 2xy,  df/dy = x^2 + 3
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(5.0, requires_grad=True)
f = x**2 * y + 3 * y
f.backward()  # compute the partial derivatives of f with respect to x and y
print("f =", f.item())
print("df/dx =", x.grad.item(), " (by hand: 2*x*y =", 2 * 2.0 * 5.0, ")")
print("df/dy =", y.grad.item(), " (by hand: x^2 + 3 =", 2.0**2 + 3, ")")


# --- 4. nn.Module: a container for parameters ---
class Line(nn.Module):
    """The model y = w * x + b, with two learnable numbers w and b."""

    def __init__(self) -> None:
        """Create the two parameters, both starting at 0."""
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))  # nn.Parameter = "please learn this"
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute w * x + b for every element of x."""
        return self.w * x + self.b


print("\n4. nn.Module")
model = Line()
print("parameters:", [(name, p.item()) for name, p in model.named_parameters()])

# --- 5. A complete training loop ---
print("\n5. Training loop: learn y = 2x + 1 from noisy examples")
xs = torch.linspace(-1, 1, 50)  # 50 inputs between -1 and 1
ys = 2 * xs + 1 + 0.1 * torch.randn(50)  # the "true" rule plus a little noise

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(201):
    predictions = model(xs)  # 1) forward pass
    loss = ((predictions - ys) ** 2).mean()  # 2) how wrong are we? (mean squared error)
    optimizer.zero_grad()  # 3) clear old gradients
    loss.backward()  # 4) backward pass: d(loss)/dw and d(loss)/db
    optimizer.step()  # 5) nudge w and b downhill
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  w {model.w.item():.3f}  b {model.b.item():.3f}")
