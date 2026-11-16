### Chapter 19

**19.1 to 19.3** The solution is this repository. Use it the way the rules of Section
19.1 describe: as something to check against after you have made your own attempt, not
as something to copy. For each milestone, the corresponding files are:

| Milestone | Where to compare |
|---|---|
| M1 | `mini_gpt/tokenizer.py` (`CharTokenizer`), `mini_gpt/data.py` (`get_batch`) |
| M2 | `examples/ch08_bigram.py` |
| M3, M4 | `mini_gpt/model.py` (`CausalSelfAttentionHead`), `tests/test_attention.py` |
| M5 | `mini_gpt/model.py` (`MultiHeadAttention`, `FeedForward`, `LayerNorm`, `Block`) |
| M6 | `mini_gpt/model.py` (`GPT`), `tests/test_overfit.py` |
| M7, M9 | `train.py`, `mini_gpt/utils.py`, `mini_gpt/config.py` |
| M8 | `mini_gpt/model.py` (`generate` and the sampling functions), `generate.py` |
| M10 | `mini_gpt/tokenizer.py` (`BPETokenizer`), `tests/test_tokenizer.py` |
| M11 | `scripts/` |

If your numbers at a check differ slightly from those in the chapter (a validation loss
of 1.49 instead of 1.47, say), that is expected: different random seeds and small
differences in the code give slightly different results. If they differ a lot (2.3
instead of 1.47), something is wrong.

**19.4** Suggestions, roughly from easier to harder, are in Section 21.3.
