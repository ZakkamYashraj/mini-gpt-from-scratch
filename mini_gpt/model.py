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


class MultiHeadAttention(nn.Module):
    """Several attention heads running in parallel on the same input.

    One head can only compute one "kind" of relationship between tokens. With
    n_head heads, each of width C / n_head, the model can track several kinds
    at once (for example: "the previous character", "the start of this word",
    "who is speaking") for the same total cost as one big head.

        head outputs:  n_head tensors of shape (B, T, C / n_head)
        concatenate :  (B, T, C)
        project     :  one Linear layer mixes the heads' results together
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        head_size = n_embd // n_head
        # Built directly from the single-head class above.
        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(n_embd, head_size, block_size, dropout, bias)
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)  # mixes information across heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C)

        Returns:
            out: (B, T, C)
            weights: (B, n_head, T, T) - the attention weights of every head.
        """
        head_outputs = []
        head_weights = []
        for head in self.heads:
            out, weights = head(x)  # (B, T, head_size), (B, T, T)
            head_outputs.append(out)
            head_weights.append(weights)

        # Glue the heads' outputs side by side along the channel dimension:
        # n_head tensors of width head_size -> one tensor of width C.
        out = torch.cat(head_outputs, dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))  # (B, T, C)
        return out, torch.stack(head_weights, dim=1)  # (B, n_head, T, T)


class FeedForward(nn.Module):
    """A two-layer network applied to every token position independently.

    Attention lets tokens *gather* information from each other; the
    feed-forward block lets each token *think about* what it gathered.

        out = W_2 . GELU(W_1 . x)       widths: C -> 4C -> C
    """

    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)  # expand to 4x the width
        self.act = nn.GELU()  # smooth non-linearity (a softer version of ReLU)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)  # project back to width C
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C). The same weights are used at every position."""
        return self.dropout(self.proj(self.act(self.fc(x))))


class LayerNorm(nn.Module):
    """Layer normalisation, written out by hand.

    For each token vector x (length C) separately:

        x_hat = (x - mean(x)) / sqrt(var(x) + eps)     now mean 0, variance 1
        out   = gain * x_hat + bias                    learned re-scale and shift

    Keeping every vector at a standard size stops the numbers from growing or
    shrinking as they pass through many layers, which makes training stable.
    """

    def __init__(self, n_embd: int, bias: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps  # tiny constant that avoids dividing by zero
        self.gain = nn.Parameter(torch.ones(n_embd))  # starts as "do nothing" (multiply by 1)
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C), normalising over the last dimension only."""
        mean = x.mean(dim=-1, keepdim=True)  # (B, T, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, T, 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gain * x_hat
        if self.bias is not None:
            out = out + self.bias
        return out


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (feed-forward).

        x = x + attention(norm(x))
        x = x + feed_forward(norm(x))

    * ``x + ...`` is a *residual connection*: each sub-layer only has to learn
      a correction to add to x. Gradients can also flow straight through the
      "+" during backpropagation, which is what makes deep stacks trainable.
    * The norm is applied *before* each sub-layer ("pre-LayerNorm", as in GPT-2).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = MultiHeadAttention(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout, cfg.bias)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.ffwd = FeedForward(cfg.n_embd, cfg.dropout, cfg.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, C) -> (B, T, C), plus this block's attention weights (B, n_head, T, T)."""
        attn_out, weights = self.attn(self.ln1(x))
        x = x + attn_out  # residual connection around attention
        x = x + self.ffwd(self.ln2(x))  # residual connection around the feed-forward net
        return x, weights


class GPT(nn.Module):
    """The full language model.

        token IDs (B, T)
          -> token embedding + positional embedding      (B, T, C)
          -> n_layer transformer Blocks                  (B, T, C)
          -> final LayerNorm                             (B, T, C)
          -> linear "LM head"                            (B, T, V)  scores for the next token

    The output at position t is a score (a *logit*) for every token in the
    vocabulary: how likely the model thinks each one is to come next.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token embedding: a table with one learned C-dimensional row per token ID.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Positional embedding: one learned row per position 0 .. block_size-1.
        # Attention itself has no notion of order, so we add "where am I?" to
        # every token vector. (Disabled in the no-positional-embedding ablation.)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.use_pos_emb else None
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)  # final normalisation

        # LM head: turns each C-dimensional vector into V scores, one per token.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # Weight tying: the embedding table (V, C) and the output layer's
            # weight (V, C) have the same shape and do mirror-image jobs
            # (ID -> vector, vector -> ID scores), so we make them the SAME
            # tensor. This saves V*C parameters and usually improves results.
            self.lm_head.weight = self.token_emb.weight

        # Initialise weights with small random numbers (as in GPT-2).
        self.apply(self._init_weights)
        # The last Linear of each residual branch gets an even smaller start,
        # so that at initialisation every block is close to "add nothing".
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Normal(0, 0.02) for weight matrices and embeddings, zeros for biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Number of learned parameters (shared tensors are counted once)."""
        return sum(p.numel() for p in self.parameters())

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """Token IDs (B, T) -> input vectors (B, T, C) = token embedding + position embedding."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.token_emb(idx)  # (B, T, C): look up one row per token ID
        if self.pos_emb is not None:
            positions = torch.arange(T, device=idx.device)  # [0, 1, ..., T-1]
            x = x + self.pos_emb(positions)  # (T, C) is broadcast across the batch
        return self.drop(x)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: token IDs, shape (B, T).
            targets: the correct next tokens, shape (B, T). Optional.

        Returns:
            logits: (B, T, V) - unnormalised scores for the next token at every position.
            loss: the average cross-entropy if ``targets`` was given, otherwise None.
        """
        x = self.embed(idx)  # (B, T, C)
        for block in self.blocks:
            x, _ = block(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            # Cross-entropy = -log(probability the model gave to the correct token),
            # averaged over all B*T positions. F.cross_entropy applies the softmax
            # itself and wants 2-D logits, so we flatten (B, T, V) -> (B*T, V).
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def attention_maps(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Run the model and return every block's attention weights (for visualisation).

        Returns:
            A list with one tensor per block, each of shape (B, n_head, T, T).
        """
        x = self.embed(idx)
        maps = []
        for block in self.blocks:
            x, weights = block(x)
            maps.append(weights)
        return maps

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Extend the sequence ``idx`` (B, T) by ``max_new_tokens`` sampled tokens.

        This is *autoregressive* generation: predict one token, append it to
        the input, and repeat. Call ``model.eval()`` first to switch off dropout.
        """
        for _ in range(max_new_tokens):
            # The model can only see block_size tokens, so keep just the most recent ones.
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)  # (B, T, V)
            logits = logits[:, -1, :]  # (B, V): only the prediction after the last token
            next_id = sample_next_token(logits, temperature, top_k, top_p)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)  # (B, T + 1)
        return idx


# --------------------------------------------------------------------------- #
# Sampling: choosing the next token from the model's scores
# --------------------------------------------------------------------------- #


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the ``k`` highest-scoring tokens; set every other logit to -inf."""
    k = min(k, logits.size(-1))
    kth_largest = torch.topk(logits, k, dim=-1).values[..., -1, None]  # (B, 1)
    return logits.masked_fill(logits < kth_largest, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of top tokens whose probabilities sum to >= p.

    Unlike top-k, the number of tokens kept adapts: few when the model is
    confident, many when it is unsure.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Probability mass that comes BEFORE each token in the sorted order. A token
    # is dropped once the tokens ahead of it already cover p (the best token has
    # nothing ahead of it, so it is always kept).
    mass_before = cumulative - sorted_probs
    sorted_logits = sorted_logits.masked_fill(mass_before >= p, float("-inf"))
    # Put the logits back into their original (unsorted) vocabulary order.
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Pick the next token ID from ``logits`` of shape (B, V). Returns shape (B, 1).

    * temperature < 1 sharpens the distribution (safer, more repetitive);
      temperature > 1 flattens it (more surprising, more mistakes);
      temperature == 0 always takes the single most likely token ("greedy").
    * top_k and top_p cut off the unlikely tail before sampling.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        logits = top_k_filter(logits, top_k)
    if top_p is not None and 0.0 < top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # (B, V), each row sums to 1
    return torch.multinomial(probs, num_samples=1)  # draw one token per row
