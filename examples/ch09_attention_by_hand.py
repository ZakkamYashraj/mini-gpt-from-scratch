"""Learning Guide, Chapters 9 and 10: self-attention on three tokens, with numbers small
enough to check by hand, then the same numbers from the repo's attention class.

Run it from the repository root:  python examples/ch09_attention_by_hand.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mini_gpt.model import CausalSelfAttentionHead  # noqa: E402

torch.set_printoptions(precision=4, sci_mode=False)

# --- 1. Three tokens, each a 2-dimensional vector ---
X = torch.tensor([[1.0, 0.0],   # token 0
                  [0.0, 1.0],   # token 1
                  [1.0, 1.0]])  # token 2

# The three "learned" matrices. Here we simply choose easy numbers by hand.
W_q = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
W_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
W_v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# --- 2. Queries, keys and values ---
Q = X @ W_q  # what each token is looking for
K = X @ W_k  # what each token contains
V = X @ W_v  # what each token hands over
print("Q =\n", Q)
print("K =\n", K)
print("V =\n", V)

# --- 3. Scores: every query dotted with every key, scaled ---
head_size = 2
scores = Q @ K.T / math.sqrt(head_size)
print("\nscores = Q @ K.T / sqrt(2) =\n", scores)

# --- 4. Without a mask (every token sees every token) ---
weights_unmasked = F.softmax(scores, dim=-1)
print("\nweights WITHOUT mask (each row sums to 1) =\n", weights_unmasked)

# --- 5. Causal mask: no looking at later tokens ---
mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
print("\nmask =\n", mask.int())
masked_scores = scores.masked_fill(~mask, float("-inf"))
print("masked scores =\n", masked_scores)
weights = F.softmax(masked_scores, dim=-1)
print("weights WITH mask =\n", weights)

# --- 6. Output: weighted average of the values ---
out = weights @ V
print("\noutput = weights @ V =\n", out)

# --- 7. The same computation with the class from mini_gpt/model.py ---
head = CausalSelfAttentionHead(n_embd=2, head_size=2, block_size=3)
with torch.no_grad():
    # nn.Linear computes x @ weight.T, so we store the transposes of our matrices.
    head.query.weight.copy_(W_q.T)
    head.key.weight.copy_(W_k.T)
    head.value.weight.copy_(W_v.T)
    class_out, class_weights = head(X[None])  # add a batch dimension: (1, 3, 2)
print("\nCausalSelfAttentionHead output =\n", class_out[0])
print("matches the hand computation:", torch.allclose(class_out[0], out))
