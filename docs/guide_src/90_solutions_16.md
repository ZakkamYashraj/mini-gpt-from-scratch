### Chapter 16

**16.1** The first row has a single cell with weight 1, in the first column, and zeros
everywhere else: the first token has nothing before it, so the mask leaves it only
itself to attend to. The top-right cell is "the first token attending to the last
token", the furthest possible look into the future, so it is always exactly 0.

**16.2** Mean distance: position 10 looks 1 token back with weight 0.5 and 5 tokens back
with weight 0.5, so $0.5 \times 1 + 0.5 \times 5 = 3$ tokens. Effective number of
tokens: the entropy is $-2 \times 0.5 \ln 0.5 = \ln 2$, and $e^{\ln 2} = 2$. The weight
is spread evenly over two tokens, as you can see directly.

**16.3** The offset heads of layer 0 do not change. On
`the king and the queen and the king`, heads 0 and 4 attend one position back at every
token and head 2 two positions back, exactly as on the Shakespeare sentence. Their
behaviour depends on position, not on content. (In this sentence without punctuation,
even the exceptions seen at `:` and `,` disappear.) The heads of layer 5, by contrast,
rearrange themselves completely for the new sentence, because what they attend to
depends on what the text says.

**16.4** All scores should be nearly equal, so each token should spread its attention
almost evenly over every token it is allowed to see. The reason is the initialisation
(Section 12.5): with weights of size 0.02, the queries and keys are tiny, every dot
product is close to 0, and a softmax of equal scores is uniform.

```python
import sys
sys.path.insert(0, "scripts")
from attention_stats import head_statistics

model = GPT(load_config("configs/small.yaml").model).eval()
maps = model.attention_maps(torch.randint(0, 65, (4, 128)))
print([round(s["effective_tokens"], 1) for s in head_statistics(maps[0])])
# [63.2, 63.3, 63.3, 63.3, 63.2, 63.3]
```

Every head has an effective number of about 63 (the rows measured can see between 17
and 128 tokens, and uniform attention over those averages out to about that), and less
than 2% of the weight falls on the previous character. Compare 1.0 and 95% for the
trained heads.

**16.5** Wrap one head's `forward` so that it returns zeros, measure, and restore it:

```python
def zero_head(model, layer, head):
    module = model.blocks[layer].attn.heads[head]
    original = module.forward
    def forward(x):
        out, weights = original(x)
        return torch.zeros_like(out), weights      # the head contributes nothing
    module.forward = forward
    return lambda: setattr(module, "forward", original)   # call this to undo
```

The repository's version of this experiment is
`python scripts/attention_stats.py --head_removal`. Its measurements on the trained
model:

{{include results/attention_stats.md from="| Heads removed" to="| 0 and 4 |"}}

Removing either previous-character head alone (0 or 4) does almost nothing: the other
one covers for it. Removing both costs 0.0118, about three times the sum of the two
separate effects (0.0026 + 0.0013). That is direct evidence of **redundancy**: the two
heads carry the same information, and the model needs it once. No single head of this
layer matters much by itself; the largest single effect is 0.0037.

Even with both gone the damage is modest (0.012). Heads 1 and 3 also put 43% to 49% of
their weight on the previous character (see the table in Section 16.4), so the
information still gets through. A trained network is rarely a tidy machine in which
each part has one job and each job has one part.

Notice what this experiment did that the heatmaps could not: it *intervened* on the
model and measured the consequence. That is the difference between describing where a
head looks and testing what it is for.
