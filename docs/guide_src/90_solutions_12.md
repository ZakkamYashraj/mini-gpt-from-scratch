### Chapter 12

**12.1** Token IDs `(64, 256)`. After the embeddings `(64, 256, 384)`. After each of the
six blocks `(64, 256, 384)`. After the final LayerNorm `(64, 256, 384)`. After the LM
head `(64, 256, 65)`.

**12.2** With `C = 96`:

- token embedding: $65 \times 96 = 6{,}240$ (shared with the LM head, counted once)
- positional embedding: $64 \times 96 = 6{,}144$
- one block: attention $3 \times 96 \times 96 + 96 \times 96 = 36{,}864$ (the three
  heads of size 32 together hold three `(96, 96)` matrices' worth of weights, plus the
  projection); feed-forward $2 \times 96 \times 384 = 73{,}728$; two LayerNorm gains
  $192$. Total $110{,}784$.
- three blocks: $332{,}352$
- final LayerNorm: $96$

Sum: $6{,}240 + 6{,}144 + 332{,}352 + 96 = 344{,}832$, which is what `train.py` prints.

**12.3** $50{,}257 \times 768 = 38{,}597{,}376$ parameters, about 31% of the model. With
a large vocabulary, weight tying is a major saving, not a detail.

**12.4**

```python
model = GPT(cfg)
for module in model.modules():
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=1.0)
_, loss = model(idx, targets)
print(loss.item())      # about 18
```

A loss of 18 means the model gives the correct token a probability of around
$e^{-18} \approx 10^{-8}$ on average. With large weights the logits are huge, so the
softmax puts essentially all the probability on one (random) token. When that token is
wrong, which is almost always, the correct one is left with almost nothing, and
$-\ln$ of almost nothing is very large. This is "confidently wrong" taken to the
extreme.

**12.5**

```python
import torch
import torch.nn.functional as F
from generate import load_model

def top_predictions(model, tokenizer, text, k=5):
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(text)], device=device)
    with torch.no_grad():
        logits, _ = model(idx)                    # (1, T, V)
    probs = F.softmax(logits[0, -1], dim=-1)      # distribution after the last token
    top = torch.topk(probs, k)
    return [(tokenizer.token_to_str(i), round(p, 3))
            for i, p in zip(top.indices.tolist(), top.values.tolist())]

model, tokenizer, _ = load_model("checkpoints/small/best.pt", torch.device("cpu"))
print(top_predictions(model, tokenizer, "First Citi"))
print(top_predictions(model, tokenizer, "ROMEO:\n"))
```

With the trained model of this repository:

```text
[('z', 1.0), ('x', 0.0), ('Z', 0.0), ('m', 0.0), ('g', 0.0)]
[('I', 0.147), ('T', 0.144), ('W', 0.104), ('A', 0.083), ('H', 0.072)]
```

After `First Citi` the model gives `z` a probability of 0.9999. After `ROMEO:\n` (the
start of a speech) the probability is spread over many capital letters, led by `I`, `T`
and `W`, because a speech can begin with almost any word. The model is confident exactly
where the text is predictable, and appropriately unsure where it is not.
