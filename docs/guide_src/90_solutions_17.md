### Chapter 17

**17.1**

```python
def test_generate_with_a_batch_of_prompts() -> None:
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    prompt = torch.randint(0, 65, (2, 3))
    out = model.generate(prompt, max_new_tokens=10)
    assert out.shape == (2, 13)
    assert torch.equal(out[:, :3], prompt)     # generation appends; it never alters the prompt
```

**17.2** With the mask replaced by all ones, 3 of the 33 tests fail:

```text
FAILED tests/test_attention.py::test_attention_weights_on_future_tokens_are_zero
FAILED tests/test_attention.py::test_causal_mask_prevents_looking_ahead
FAILED tests/test_overfit.py::test_model_can_memorise_one_batch
3 failed, 30 passed
```

The first two are the tests written for exactly this. The third is more interesting.
The broken model *does* memorise the batch (its loss goes below 0.1), but the last line
of the test, which asks the model to regenerate the sequence, fails: a model that learned
by peeking at the next token cannot produce that token when it is not there. A
whole-system test caught a bug it was not specifically written for.

The 30 tests that still pass include every shape test, "each row of weights sums to 1",
the initial-loss test and all the tokenizer tests. None of those properties depends on
causality. This is why shape tests alone are never enough: a model can have perfect
shapes and be fundamentally wrong.

**17.3**

```python
def test_lr_never_increases_after_warmup() -> None:
    kwargs = dict(max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
    rates = [get_lr(step, **kwargs) for step in range(100, 1001)]
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))
    assert min(rates) >= 1e-4
```

`zip(rates, rates[1:])` pairs each rate with the next one, the same trick the BPE code
uses to list adjacent pairs.

**17.4**

```python
def test_dropout_only_active_in_train_mode() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.5)
    model = GPT(cfg)
    idx = torch.randint(0, 65, (1, 16))

    model.eval()
    a, _ = model(idx)
    b, _ = model(idx)
    assert torch.equal(a, b)            # no randomness in eval mode

    model.train()
    c, _ = model(idx)
    d, _ = model(idx)
    assert not torch.allclose(c, d)     # dropout zeroes different values each time
```

**17.5** With `dropout=0.5` the test fails, but not because the model failed to learn:

| | loss printed during training | same batch, `model.eval()` |
|---|---|---|
| `dropout=0.0` | 0.012 | 0.012 |
| `dropout=0.5` | 0.217 | 0.021 |

In evaluation mode the loss is 0.021: the batch *has* been memorised. But in training
mode half of the activations are zeroed at random on every forward pass, so the loss
measured there stays around 0.2 and never gets under the 0.1 that the test demands.

Two lessons. Dropout deliberately makes memorising hard, which is its purpose in real
training and exactly what you do not want in a test whose point is to memorise; so
sanity checks are run with dropout off. And a loss measured in training mode is not the
model's real performance, which is why `estimate_loss` switches to `eval()`.
