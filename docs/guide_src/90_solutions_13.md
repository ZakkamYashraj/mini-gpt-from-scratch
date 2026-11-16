### Chapter 13

**13.1** Step 50 is in the warmup: $10^{-3} \times 51/100 = 5.1 \times 10^{-4}$.
Step 100 is the first step after the warmup, with `progress` $= 0$: the peak, $10^{-3}$.
Step 3775: `progress` $= 3675/4900 = 0.75$, so the coefficient is
$0.5\,(1 + \cos(0.75\pi)) = 0.5\,(1 - 0.7071) = 0.146$, and the rate is
$10^{-4} + 0.146 \times 9 \times 10^{-4} \approx 2.3 \times 10^{-4}$.

**13.2** Length: $\sqrt{4 + 1 + 4} = 3$. With a maximum of 1.0 every component is
divided by 3: $(0.667, -0.333, 0.667)$, which has length 1 and the same direction. With
a maximum of 5.0 nothing happens, because 3 is already below the threshold.

**13.3**

(a) Without `model.eval()`, dropout stays on during evaluation. The **numbers are
wrong**: both losses come out higher and noisier than the model's true performance.

(b) Without the final `model.train()`, the evaluation's own numbers are fine, but the
model stays in evaluation mode, so dropout is silently off for all the training that
follows. The **later training is wrong**: the model overfits more than it should, and
nothing warns you.

(c) Without `@torch.no_grad()`, the numbers are **correct**. Only the **cost** changes:
PyTorch records a computation graph for every evaluation batch, which uses more memory
and time for no benefit.

**13.4** Measured on the tiny configuration (2,000 steps, one seed, GPU):

| | best validation loss |
|---|---|
| warmup + cosine decay to `1e-4` | 1.831 |
| constant `1e-3` | 1.807 |

The constant rate was very slightly *better*. This is worth thinking about, not
explaining away. The tiny model is still improving steadily when the run ends (it is
underfitting, not overfitting), so reducing its learning rate towards the end only slows
it down. A decaying schedule earns its keep in longer runs of larger models, where the
late, small steps are needed to settle. And the difference, 0.02 from a single seed, is
about the size of the run-to-run noise measured in Chapter 15, so the supported
conclusion is "no clear difference here". Techniques that are standard for large models
do not automatically help small ones: measure.

**13.5** Add a counter next to `best_val`, and update it where the checkpoint is saved:

```python
best_val = float("inf")
evals_without_improvement = 0
patience = 5
...
        if last_losses["val"] < best_val:
            best_val = last_losses["val"]
            evals_without_improvement = 0          # reset: we just improved
            save_checkpoint(...)
        else:
            evals_without_improvement += 1
            if evals_without_improvement >= patience:
                print(f"Stopping early at step {step}: no improvement for {patience} evaluations.")
                break
```

The counter is reset to zero whenever the validation loss improves and incremented
otherwise. With `patience = 5` and an evaluation every 250 steps, the main run would
have stopped at step 3,000 (five evaluations after the best one at step 1,750) instead
of 5,000, saving about 40% of the time with the same saved model. To do it properly,
`patience` should be a field of `TrainConfig` so that it lives in the YAML file.
