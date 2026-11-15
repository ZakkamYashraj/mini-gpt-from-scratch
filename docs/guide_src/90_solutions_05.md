### Chapter 5

**5.1** Training: $250{,}000 \times 0.9 = 225{,}000$ tokens. Validation: $25{,}000$
tokens. The validation set is the **last** 10% of the text, one continuous block.

**5.2** (a) **Overfitting**, and badly: a huge gap between the two losses. Use the
checkpoint with the best validation loss, add dropout, shrink the model, or get more
data. (b) **Underfitting**: the two losses are equal, so nothing is being memorised, but
neither is improving. Try a bigger model, a different learning rate, or check for a bug.
(c) **Fine.** A small gap is normal, and both curves are still improving, so keep
training.

**5.3** One step processes $64 \times 256 = 16{,}384$ tokens. One epoch is therefore
about $1{,}003{,}854 / 16{,}384 \approx 61$ steps, and 5000 steps is about
$5000 / 61 \approx 82$ epochs. The model sees each part of the training text roughly 80
times, which explains why it is able to start memorising it.

**5.4** Measured validation losses for the tiny model:

| learning rate | step 0 | step 100 | step 200 | step 300 |
|---|---|---|---|---|
| `1e-5` | 4.19 | 3.93 | 3.73 | 3.58 |
| `1e-3` | 4.19 | 2.64 | 2.42 | 2.35 |
| `1.0` | 4.19 | 7.15 | 4.82 | 4.11 |

With `1e-5` the loss does fall, but so slowly that 300 steps achieve less than the first
20 steps at `1e-3`. With `1.0` the steps are enormous: the loss first shoots *up* to 7,
far worse than guessing, and then settles near 4.1, barely better than the 4.17 of
uniform guessing. The model has been thrown somewhere it cannot recover from and has
learned essentially nothing. (It did not become `nan` only because gradient clipping
and Adam limit the size of each update.)

**5.5**

```python
w, v, lr = 0.0, 0.0, 0.1
for step in range(30):
    grad = 2 * (w - 3)
    v = 0.9 * v + grad        # velocity: a running total of past gradients
    w = w - lr * v
    print(step, round(w, 3))
```

Plain gradient descent creeps toward 3 from below and never passes it:
`0.6, 1.08, 1.464, ... 2.996`. With momentum, $w$ goes
`0.6, 1.62, 2.814, 3.926, 4.741, 5.127, ...`: it accelerates, **overshoots** to 5.1,
swings back to 1.5, and oscillates around 3 with shrinking swings, like a ball rolling
in a bowl. On this easy problem that is no better. The advantage appears on hard
landscapes: where the gradient is small but consistently in one direction (momentum
builds up speed), and where it zig-zags from step to step (the zig-zags cancel in the
running total). Real loss landscapes are full of both.
