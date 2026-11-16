## Chapter 17: Testing machine-learning code with pytest

> **Key ideas**
>
> - Bugs in machine-learning code rarely crash. The program runs, the loss goes down,
>   and the result is merely *worse than it should be*. You will not notice by looking.
> - Tests catch these bugs by checking facts that **must** be true whatever the weights
>   are: shapes, mathematical properties, round trips, and known values.
> - The single most valuable test is the **overfit test**: a correct model and training
>   loop can memorise one small batch.
> - A test is only worth having if it can fail. Check that it does.

### 17.1 Why this code needs tests more than most

In ordinary programming a bug usually announces itself: a crash, a wrong number on the
screen. Machine learning is different, because a neural network is a machine for
finding *some* way to lower the loss with whatever it is given. Give it a broken
attention layer and it will route around the damage and still learn something. The loss
curve will look plausible. The samples will look like text. And the model will be
quietly worse than it should be, with nothing to tell you so.

Section 10.2 showed the mirror image of this: a bug (no causal mask) that made the loss
look *wonderful* while making the model useless.

So we cannot rely on "it runs and the loss goes down". We need to check, piece by
piece, facts that have to hold. That is what the 33 tests in `tests/` do.

### 17.2 pytest in two minutes

**pytest** is the standard Python testing tool. The rules are simple:

- Tests live in files named `test_*.py`.
- Each test is a function whose name starts with `test_`.
- Inside it, `assert` states something that must be true. If the expression is false,
  the test fails and pytest shows you the values involved.

Run all of them from the repository root:

```bash
python -m pytest                              # everything
python -m pytest -v                           # one line per test
python -m pytest tests/test_attention.py      # one file
python -m pytest -k causal                    # only tests with "causal" in their name
```

```text
.................................                                        [100%]
33 passed in 7.53s
```

Each dot is a passing test. The small file `pytest.ini` tells pytest where the tests are
and puts the repository root on Python's import path, so that `from mini_gpt.model
import GPT` works inside the tests.

All tests use tiny models (two layers, width 32) and run on the CPU, so the whole suite
takes well under a minute. Tests that are slow do not get run.

### 17.3 Shape tests

The cheapest kind: does a layer return a tensor of the shape it promises?

{{code tests/test_attention.py from="def test_single_head_output_shapes" to="assert weights.shape == (B, 4, T, T)"}}

These look almost too simple to be worth writing. But most errors made while *writing*
a model are shape errors, and a shape test fails immediately and tells you exactly
where, instead of leaving you with a broadcasting bug that silently computes the wrong
thing (Section 2.5).

### 17.4 Property tests

A **property** is something that must be true for *any* input and *any* weights. We
cannot say what the attention weights of a random model should be. But we can say that
every row must be a probability distribution, and that nothing may be placed on the
future:

{{code tests/test_attention.py from="def test_attention_weights_are_a_probability_distribution" to="assert torch.allclose(weights[:, 0, 0], torch.ones(B))"}}

`torch.triu(..., diagonal=1)` is the *upper* triangle above the diagonal: exactly the
positions where token $j$ lies in the future of token $i$. The weights there must be
exactly 0.

The most important property test in the repository is the behavioural check of
causality, which you read in Section 10.6: change a token, and no earlier output may
change.

Another useful property test compares our code with a trusted reference:

{{code tests/test_attention.py symbol="test_layernorm_matches_pytorch"}}

We wrote LayerNorm by hand for understanding. PyTorch's version has been used by
millions of people. If the two agree on random inputs, ours is right.

> **Note**
>
> Notice `torch.allclose(a, b, atol=1e-5)` instead of `a == b`. Decimal arithmetic on a
> computer is not exact, and two correct ways of computing the same thing usually
> differ in the last digit. Compare floating-point numbers with a tolerance, except in
> the rare cases (like the causality test) where the results must be bit-for-bit
> identical.

### 17.5 Round-trip tests

For a tokenizer the essential fact is that decoding undoes encoding. This is called a
**round trip**:

{{code tests/test_tokenizer.py symbol="test_bpe_round_trip_on_unseen_and_non_english_text"}}

The inputs are chosen to be awkward on purpose: an empty string, words that never
occurred in the training text, accented letters, Telugu, tabs and double newlines.
Tests should go looking for trouble. Easy inputs rarely reveal a bug; the odd ones at
the edges do.

### 17.6 Known-value tests

Sometimes theory tells us what the answer must be.

{{code tests/test_model.py symbol="test_initial_loss_is_close_to_uniform_guessing"}}

An untrained model should have a loss near $\ln(V)$ (Sections 1.4 and 12.5). If this
test fails, the initialisation is wrong or the loss is being computed wrongly.

For small pure functions we can simply work the answer out by hand and write it down,
as with top-p sampling (compare with the table in Section 14.5):

{{code tests/test_model.py symbol="test_top_p_filter_keeps_smallest_nucleus"}}

### 17.7 The overfit test

This is the test to write first for any model you ever build.

The idea: take **one small batch** and train on it, the same batch, over and over. A
correct model with a correct training loop must be able to drive the loss to nearly
zero, because memorising 4 short sequences is trivially within the capacity of even a
tiny transformer. If it cannot, something is broken.

{{code tests/test_overfit.py symbol="test_model_can_memorise_one_batch"}}

Why is this one test so powerful? Because of how much has to be right for it to pass:

- the targets must line up with the inputs (shifted by exactly one);
- the loss must be computed correctly;
- gradients must reach every layer that matters;
- the optimizer must be updating the parameters;
- attention must really be passing information along, since the right answer at a
  position often depends on earlier characters;
- and `generate` must be consistent with `forward`, or the final check, that greedy
  generation reproduces the memorised sequence exactly, would fail.

It is also the first thing to try when a real training run misbehaves: "can it overfit
one batch?" splits the problem in two. If no, the bug is in the model or the loop. If
yes, look at the data, the learning rate, or the size of the model.

### 17.8 Testing the test

A test that can never fail proves nothing. Here is a quiet danger: suppose the
causality test had a mistake and compared the wrong positions. It would pass, with or
without a working mask, and we would trust it.

So the repository contains a test whose only job is to show that the *method* of
testing can detect a leak:

{{code tests/test_attention.py symbol="test_without_mask_the_future_would_leak"}}

It deliberately breaks the mask (fills it with `True`) and asserts that earlier outputs
now **do** change. Together, the two tests say: with the mask there is no leak, and
without it this kind of check does detect one.

You can do the same by hand for any test: break the code on purpose, confirm that the
test goes red, and restore the code. Exercise 17.2 asks you to.

### 17.9 Habits worth keeping

- **Fix the seed** inside tests that use random numbers (`torch.manual_seed(0)`), so a
  failure can be reproduced.
- **Keep tests small and fast.** Tiny models, CPU, no downloads. `test_tokenizer.py`
  uses a few lines of text written into the file, not the dataset.
- **Use pytest's helpers.** `pytest.raises(ValueError)` checks that bad input is
  rejected; `pytest.approx` compares numbers with a tolerance; the `tmp_path` argument
  gives a test a temporary folder that is deleted afterwards (used by the save/load
  tests).
- **Run the tests after every change**, before you commit. A test that passed yesterday
  and fails today tells you that today's edit is the cause.

### Exercises

**17.1 (easy)** Write a test checking that `GPT.generate` works with a batch of two
prompts: starting from a `(2, 3)` tensor of token IDs and asking for 10 new tokens, the
result must have shape `(2, 13)`, and its first three columns must equal the prompt.

**17.2 (medium)** Break the model on purpose: in `CausalSelfAttentionHead.__init__`,
replace `torch.tril(torch.ones(...))` by just `torch.ones(...)`. Run the tests. Which
ones fail, and which, perhaps surprisingly, still pass? Restore the code afterwards.

**17.3 (medium)** Write a test for `get_lr` asserting that after the warmup the learning
rate never increases from one step to the next, and never goes below `min_lr`.

**17.4 (medium)** Write a test showing that dropout is off in `eval()` mode and on in
`train()` mode: with `dropout=0.5`, the same input must give identical outputs twice in
eval mode, and different outputs in train mode.

**17.5 (hard)** The overfit test uses `dropout=0.0`. Predict what would happen with
`dropout=0.5`, then try it. What does this tell you about when dropout is appropriate?

### Check your understanding

1. Why can you not rely on "the loss goes down" as evidence that the code is correct?
2. Give one example each of a shape test, a property test and a round-trip test.
3. What does the overfit test prove if it passes? What does it suggest if it fails?
4. Why compare floating-point results with `torch.allclose` instead of `==`?
5. What is the purpose of a test that deliberately breaks the code?
