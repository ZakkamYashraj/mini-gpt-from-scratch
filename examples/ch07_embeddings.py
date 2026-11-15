"""Learning Guide, Chapter 7: token embeddings and positional embeddings.

Run it from the repository root:  python examples/ch07_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)

# --- 1. An embedding is a lookup table ---
print("1. Lookup")
V, C = 5, 4  # vocabulary of 5 tokens, vectors of 4 numbers
token_emb = nn.Embedding(V, C)
print("the table, shape", tuple(token_emb.weight.shape))
print(token_emb.weight.data)
idx = torch.tensor([3, 1, 3])  # three token IDs
x = token_emb(idx)
print("token_emb([3, 1, 3]) has shape", tuple(x.shape))
print(x.data)
print("row 0 of the output is row 3 of the table:", torch.equal(x[0], token_emb.weight[3]))

# --- 2. The same thing as a matrix multiplication ---
print("\n2. One-hot vectors")
one_hot = F.one_hot(idx, num_classes=V).float()  # a 1 in the position of the ID, 0 elsewhere
print(one_hot)
print("one_hot @ table gives the same result:", torch.allclose(one_hot @ token_emb.weight, x))

# --- 3. Token embeddings alone cannot see order ---
print("\n3. Order")
a = token_emb(torch.tensor([0, 1, 2]))  # think: "dog bites man"
b = token_emb(torch.tensor([2, 1, 0]))  # think: "man bites dog"
print("sum of vectors, order 0-1-2:", a.sum(dim=0).data)
print("sum of vectors, order 2-1-0:", b.sum(dim=0).data)

# --- 4. Positional embeddings ---
print("\n4. Positions")
block_size = 8
pos_emb = nn.Embedding(block_size, C)  # one learned vector per position 0..7
positions = torch.arange(3)  # tensor([0, 1, 2])
a_pos = token_emb(torch.tensor([0, 1, 2])) + pos_emb(positions)
b_pos = token_emb(torch.tensor([2, 1, 0])) + pos_emb(positions)
print("token 2 at position 2:", a_pos[2].data)
print("token 2 at position 0:", b_pos[0].data)
print("same token, different position -> different vector:", not torch.equal(a_pos[2], b_pos[0]))

# --- 5. Broadcasting over a batch ---
print("\n5. Shapes with a batch")
batch = torch.randint(0, V, (2, 6))  # B = 2 sequences of T = 6 token IDs
tok = token_emb(batch)  # (B, T, C)
pos = pos_emb(torch.arange(6))  # (T, C)
print("token embeddings:", tuple(tok.shape), " positional embeddings:", tuple(pos.shape),
      " sum:", tuple((tok + pos).shape))

# --- 6. What the trained model learned ---
print("\n6. Nearest neighbours in the trained embedding table")
checkpoint = Path("checkpoints/small/best.pt")
if not checkpoint.exists():
    print("(train the model first: python train.py --config configs/small.yaml)")
else:
    from generate import load_model

    model, tokenizer, _ = load_model(checkpoint, torch.device("cpu"))
    table = F.normalize(model.token_emb.weight.detach(), dim=-1)  # make every row length 1
    similarity = table @ table.T  # cosine similarity between every pair of tokens
    for ch in ["a", "t", "A", "."]:
        i = tokenizer.encode(ch)[0]
        nearest = similarity[i].topk(5).indices.tolist()[1:]  # skip the token itself
        print(f"closest to {ch!r}: {[tokenizer.decode([j]) for j in nearest]}")
