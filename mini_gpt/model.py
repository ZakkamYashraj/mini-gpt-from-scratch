"""The GPT model, built from scratch one small piece at a time.

Reading order (each class uses the ones above it):

    CausalSelfAttentionHead   one attention head: Q, K, V, mask, softmax
    MultiHeadAttention        several heads side by side, then a projection
    FeedForward               a small 2-layer network applied to each position
    LayerNorm                 normalise each token vector
    Block                     attention + feed-forward, with residual connections
    GPT                       embeddings -> N blocks -> final norm -> next-token scores

Shape conventions used in every comment:

    B = batch size        (how many sequences are processed at once)
    T = time / sequence   (how many tokens in each sequence, T <= block_size)
    C = channels          (the embedding width n_embd)
    V = vocabulary size

No ``nn.Transformer`` or ``nn.MultiheadAttention`` is used anywhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_gpt.config import ModelConfig


class CausalSelfAttentionHead(nn.Module):
    """A single head of causal (masked) self-attention.

    Intuition: every token asks a question (its *query*), every token advertises
    what it contains (its *key*), and every token has some information to hand
    over (its *value*). A token collects the values of the earlier tokens whose
    keys best match its query.

    Math, for one sequence X of shape (T, C):

        Q = X W_q,   K = X W_k,   V = X W_v            each (T, head_size)
        scores  = Q K^T / sqrt(head_size)              (T, T)
        scores[i, j] = -inf  wherever  j > i           (no looking at the future)
        weights = softmax(scores, over each row)       each row sums to 1
        output  = weights V                            (T, head_size)
    """

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        """
        Args:
            n_embd: width C of the incoming token vectors.
            head_size: width of the query/key/value vectors of this head.
            block_size: maximum sequence length (size of the causal mask).
            dropout: dropout probability applied to the attention weights.
            bias: whether the Q/K/V projections have a bias term.
        """
        super().__init__()
        self.head_size = head_size
        # Three learned linear maps. Each turns a C-dimensional token vector
        # into a head_size-dimensional query, key or value vector.
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # Causal mask: a lower-triangular matrix of True values.
        #   mask[i, j] is True  when j <= i  (token i MAY look at token j)
        #   mask[i, j] is False when j >  i  (token j is in the future of token i)
        # A buffer is a tensor that belongs to the module (it moves to the GPU
        # with it) but is not a learned parameter.
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: token vectors of shape (B, T, C).

        Returns:
            out: (B, T, head_size) - for each token, a weighted mix of values.
            weights: (B, T, T) - weights[b, i, j] is how much token i attends to token j.
        """
        B, T, C = x.shape

        # 1) Project every token into a query, a key and a value.
        q = self.query(x)  # (B, T, head_size)  "what am I looking for?"
        k = self.key(x)  # (B, T, head_size)  "what do I contain?"
        v = self.value(x)  # (B, T, head_size)  "what will I pass on if chosen?"

        # 2) Scaled dot-product: scores[b, i, j] = (q_i . k_j) / sqrt(head_size).
        #    A large dot product means query i and key j point the same way.
        #    Dividing by sqrt(head_size) keeps the scores at a sensible size so
        #    that the softmax below does not become too "peaky" at initialisation.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)  # (B, T, T)

        # 3) Causal mask: put -inf wherever a token would look into the future.
        #    exp(-inf) = 0, so after the softmax those positions get weight 0.
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))

        # 4) Softmax over the last dimension turns each row of scores into
        #    positive weights that sum to 1 (a probability distribution over
        #    the tokens this position is allowed to see).
        weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # 5) Output: for each token, the weighted average of the value vectors.
        #    Dropout randomly zeroes some weights during training (regularisation).
        out = self.dropout(weights) @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out, weights
