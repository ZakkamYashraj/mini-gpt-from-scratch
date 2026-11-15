### Chapter 3

**3.1**

```python
t = torch.randn(2, 3, 4)
print(t.shape)              # torch.Size([2, 3, 4])
print(t.dtype)              # torch.float32
print(t.sum(dim=1).shape)   # torch.Size([2, 4])   dimension 1 vanished
```

**3.2** By hand: $\partial g/\partial a = b + \cos(a)$ and $\partial g/\partial b = a$.
At $a = 1$, $b = 2$: $2 + \cos(1) \approx 2.5403$ and $1$.

```python
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
g = a * b + torch.sin(a)
g.backward()
print(a.grad.item(), b.grad.item())   # 2.5403...  1.0
```

**3.3** You see `40.0` instead of `20.0`. The second `backward()` computed 20 again and
**added** it to the 20 already stored in `x.grad`. After `x.grad.zero_()` and one more
forward and backward pass, the value is 20 again. In a training loop this is the job of
`optimizer.zero_grad()`.

**3.4**

```python
class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c

xs = torch.linspace(-1, 1, 50)
ys = 3 * xs**2 - 2 * xs + 1 + 0.1 * torch.randn(50)
model = Quadratic()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(2000):
    loss = ((model(xs) - ys) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(model.a.item(), model.b.item(), model.c.item())   # close to 3, -2, 1
```

Yes, it recovers them (up to the noise in the data). It needs more steps than the
straight line did: `a` and `c` both push the curve upward over the range of the data, so
gradient descent takes longer to tell them apart. Nothing about the loop changed, only
the model and the data. That separation is why the same loop can train a GPT.

**3.5**

```python
lr = 0.1
for step in range(201):
    loss = ((model(xs) - ys) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad      # the gradient-descent update
            p.grad.zero_()        # clear for the next iteration
```

The result is the same as with `torch.optim.SGD`. The `no_grad` block is needed for two
reasons. First, the update itself is not part of the function we want to differentiate,
so it should not be recorded in the computation graph. Second, PyTorch refuses to modify
a parameter in place while it is being tracked, and raises an error if you try.
