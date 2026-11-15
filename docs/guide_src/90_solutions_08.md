### Chapter 8

**8.1** The neighbouring pairs of `abababba` are `ab, ba, ab, ba, ab, bb, ba`.

| | next: `a` | next: `b` |
|---|---|---|
| current: `a` | 0 | 3 |
| current: `b` | 3 | 1 |

After `a`, the next character was `b` three times out of three, so the model gives
$P(b \mid a) = 3/3 = 1$. (And $P(a \mid b) = 3/4$.) A probability of exactly 1 or 0 is
dangerous: if `aa` ever appears in new text the loss is $-\ln(0) = \infty$. That is what
the `+1` smoothing in the example code prevents.

**8.2** $e^{2.48} \approx 11.9$ and $e^{1.47} \approx 4.3$. The bigram model is as
uncertain as if it were choosing among about 12 equally likely characters at each step;
the GPT, among about 4. Guessing uniformly corresponds to 65.

**8.3**

```python
import torch, torch.nn as nn, torch.nn.functional as F

table = nn.Embedding(65, 65)
x, y = val_ids[:4000], val_ids[1:4001]
print(F.cross_entropy(table(x), y).item())           # about 4.7
print(F.cross_entropy(table(x) * 0.01, y).item())    # about 4.17
```

With logits of typical size 1, the softmax gives some characters several times the
probability of others, at random. The model is opinionated and wrong, which costs more
than admitting ignorance. Multiplying the logits by 0.01 makes them nearly equal, the
softmax becomes nearly uniform, and the loss becomes $\ln 65 \approx 4.17$.

**8.4** No. The loss still ends at about 2.48. In a bigram model each prediction depends
only on the single token at that position, so giving it longer sequences does not give
it more context. A longer block only means more predictions per batch: the printed
training loss is less noisy, and each step costs more, but the best achievable loss is
the same.

**8.5**

```python
V = tokenizer.vocab_size
counts = torch.zeros(V, V, V)
counts.index_put_((train_ids[:-2], train_ids[1:-1], train_ids[2:]),
                  torch.ones(len(train_ids) - 2), accumulate=True)
probs = (counts + 1) / (counts + 1).sum(dim=-1, keepdim=True)
loss = -torch.log(probs[val_ids[:-2], val_ids[1:-1], val_ids[2:]]).mean()
print(loss.item())    # about 2.07
```

Two characters of context lower the validation loss from 2.48 to about 2.07 (the
training loss is 1.95, so a gap is already opening). But the table has
$65^3 = 274{,}625$ entries. A context of 10 characters would need $65^{11} \approx
10^{20}$ entries, and almost none of them would ever be observed in a million characters
of text: most rows would hold no information at all. Counting treats every context as
unrelated to every other. To go further, a model must *share* what it learns between
similar contexts, which is what a neural network with attention does.
