"""Learning Guide, Chapter 8: the bigram model, our first working language model.

A bigram model predicts the next character from the CURRENT character only.
It is the simplest possible baseline, built twice here:
  1. by counting (no learning at all)
  2. as a one-layer neural network trained with gradient descent

Run it from the repository root:  python examples/ch08_bigram.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mini_gpt.config import DataConfig  # noqa: E402
from mini_gpt.data import get_batch, load_dataset  # noqa: E402

torch.manual_seed(1337)
train_ids, val_ids, tokenizer, _ = load_dataset(DataConfig())  # Tiny Shakespeare, char-level
V = tokenizer.vocab_size
print(f"vocab size {V}, {len(train_ids):,} training tokens, {len(val_ids):,} validation tokens")

# --- 1. Bigram model by counting ---
# counts[a, b] = how many times token b came right after token a in the training text
counts = torch.zeros(V, V)
current, following = train_ids[:-1], train_ids[1:]
counts.index_put_((current, following), torch.ones(len(current)), accumulate=True)

# Turn each row of counts into probabilities (the +1 avoids probabilities of exactly 0).
probs = (counts + 1) / (counts + 1).sum(dim=1, keepdim=True)

t_row = probs[tokenizer.encode("t")[0]]
top = torch.topk(t_row, 5)
print("\n1. Counting")
print("after 't' the most likely next characters are:",
      [(tokenizer.decode([i]), round(p, 3)) for i, p in zip(top.indices.tolist(), top.values.tolist())])

# The loss: average of -log(probability given to the character that really came next).
val_loss = -torch.log(probs[val_ids[:-1], val_ids[1:]]).mean()
print(f"validation loss of the counting model: {val_loss.item():.4f}")
print(f"(for comparison, guessing uniformly would give ln({V}) = {torch.log(torch.tensor(float(V))).item():.4f})")


# --- 2. The same model as a neural network ---
class BigramLM(nn.Module):
    """A table with one row of V scores (logits) for every possible current token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.table(idx)  # (B, T) -> (B, T, V): look up the row for each token
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # distribution after the last token
            next_id = torch.multinomial(probs, num_samples=1)  # sample one token from it
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# --- 3. Train it with the standard loop ---
print("\n3. Training the neural bigram model")
model = BigramLM(V)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
for step in range(3001):
    x, y = get_batch(train_ids, block_size=8, batch_size=32)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"step {step:4d}  train loss {loss.item():.4f}")

with torch.no_grad():
    _, val_loss_nn = model(val_ids[None, :-1], val_ids[None, 1:])
print(f"validation loss of the neural bigram model: {val_loss_nn.item():.4f}")

# --- 4. Generate text ---
print("\n4. A sample from the bigram model")
start = torch.tensor([tokenizer.encode("\n")])
print(tokenizer.decode(model.generate(start, 200)[0].tolist()))
