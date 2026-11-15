### Chapter 7

**7.1** `token_emb(idx)`: `(64, 256, 384)`. `pos_emb(torch.arange(256))`: `(256, 384)`.
Their sum: `(64, 256, 384)`, with the positional rows broadcast over the 64 sequences.

**7.2** Token table: $65 \times 384 = 24{,}960$ parameters. Position table:
$256 \times 384 = 98{,}304$ parameters. The position table is about four times larger,
simply because there are more positions (256) than characters (65). With a BPE
vocabulary of 512 the order would reverse. Both are tiny next to the roughly 10.6
million parameters inside the transformer blocks.

**7.3**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

emb = nn.Embedding(3, 4)
idx = torch.tensor([2, 0, 2, 1])
one_hot = F.one_hot(idx, num_classes=3).float()    # shape (4, 3)
print(one_hot)
print(torch.allclose(one_hot @ emb.weight, emb(idx)))   # True
```

Each row of `one_hot` has a single 1. Multiplying by the table adds up "1 times that row
plus 0 times every other row", which is that row.

**7.4**

```python
emb = nn.Embedding(10, 4)
out = emb(torch.tensor([1, 1, 7]))
out.sum().backward()
print(emb.weight.grad)
```

Row 1 is `[2, 2, 2, 2]`, row 7 is `[1, 1, 1, 1]`, and the other eight rows are zero. Only
rows that were looked up took part in the computation, so only they receive a gradient.
Row 1 was looked up twice, and gradients from the two uses add together. This is why a
rare token learns slowly: its row is updated only when it appears in a batch.

**7.5**

```python
import math
import torch
import torch.nn.functional as F

def sinusoidal(block_size, C):
    pe = torch.zeros(block_size, C)
    position = torch.arange(block_size).unsqueeze(1)             # (block_size, 1)
    frequency = torch.exp(-math.log(10000.0) * torch.arange(0, C, 2) / C)   # (C/2,)
    pe[:, 0::2] = torch.sin(position * frequency)                # even dimensions
    pe[:, 1::2] = torch.cos(position * frequency)                # odd dimensions
    return pe

pe = F.normalize(sinusoidal(256, 384), dim=-1)
similarity = pe @ pe.T          # plot with plt.imshow(similarity)
```

(`torch.exp(-log(10000) * 2i / C)` is $1 / 10000^{2i/C}$, computed in a numerically safer
way.) In the plot, the similarity depends only on the *distance* between two positions:
you see a bright band along the diagonal that fades smoothly on both sides, perfectly
regular. The learned table shows the same general idea, nearby positions are similar,
but less regularly, because it was shaped by data instead of a formula.
