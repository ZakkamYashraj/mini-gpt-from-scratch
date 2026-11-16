### Chapter 14

**14.1** At temperature 1: $e^2 = 7.389$, $e^1 = 2.718$, $e^0 = 1$, sum $11.107$, so the
probabilities are $(0.665, 0.245, 0.090)$. At temperature 0.5 the logits become
$(4, 2, 0)$: $e^4 = 54.60$, $e^2 = 7.389$, $e^0 = 1$, sum $62.99$, giving
$(0.867, 0.117, 0.016)$. The favourite went from 67% to 87%.

**14.2**

- Top-k with $k = 2$: keep 0.4 and 0.3 and rescale by their sum, 0.7:
  $(0.571, 0.429, 0, 0)$.
- Top-p with $p = 0.6$: the probability *before* each token is $0$, $0.4$, $0.7$, $0.9$.
  The third token is dropped because 0.7 already covers 0.6. Two tokens remain:
  $(0.571, 0.429, 0, 0)$, the same as top-k here.
- Top-p with $p = 0.75$: the third token is now kept (0.7 is below 0.75) and the fourth
  is dropped. Rescale by 0.9: $(0.444, 0.333, 0.222, 0)$.

**14.3** The model would fail as soon as the sequence passed 256 tokens:

```text
ValueError: Sequence length 257 exceeds block_size 256
```

There is no row in the positional embedding table for position 256, and the causal mask
is only $256 \times 256$. Cropping to the most recent `block_size` tokens is what makes
it possible to generate text of any length; the price is that the model forgets
everything older than its window.

**14.4** The real outputs are shown in Section 14.7. At 0.2 the text is grammatical but
repeats itself ("the state of the sea" again and again). At 0.8 it is varied and mostly
made of real words. At 1.5 it produces non-words and broken punctuation.

Adding `--top_k 20` to the 1.5 run gave this for the same seed:

```text
MENENIUS:
Well; the, master, tell them all; well, if as touch of
the pasque or arrone's thing.
```

Somewhat more coherent than before (compare "Well; the, early, O, / Draw."), because the
most improbable characters can no longer be chosen, but still with invented words
(`pasque`, `arrone's`). Top-k removes the worst choices; it does not undo the flattening
that a high temperature applies to the 20 candidates that remain.

**14.5**

```python
@torch.no_grad()
def generate_stream(model, tokenizer, prompt, n, temperature=0.8, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    print(prompt, end="", flush=True)
    for _ in range(n):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        next_id = sample_next_token(logits[:, -1, :], temperature, top_k)
        print(tokenizer.decode(next_id[0].tolist()), end="", flush=True)   # show it immediately
        idx = torch.cat([idx, next_id], dim=1)
    print()
```

`flush=True` forces Python to display each character at once instead of collecting
output in a buffer.

With the character tokenizer every token is one complete character, so it can be
printed the moment it is produced. A BPE token is a sequence of *bytes*, and a single
character (a Telugu letter, for instance) can be split across two tokens. Decoding the
first of them alone gives an incomplete character, which shows up as a replacement
symbol. A correct streaming version must collect bytes and print only complete UTF-8
characters.
