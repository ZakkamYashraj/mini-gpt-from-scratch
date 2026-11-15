"""Tests for the attention layers: output shapes and, above all, causality."""

from __future__ import annotations

import torch

from mini_gpt.config import ModelConfig
from mini_gpt.model import GPT, CausalSelfAttentionHead, LayerNorm, MultiHeadAttention

B, T, C = 2, 8, 32  # batch, time, channels used throughout


def test_single_head_output_shapes() -> None:
    head = CausalSelfAttentionHead(n_embd=C, head_size=16, block_size=T)
    out, weights = head(torch.randn(B, T, C))
    assert out.shape == (B, T, 16)
    assert weights.shape == (B, T, T)


def test_multi_head_output_shapes() -> None:
    attn = MultiHeadAttention(n_embd=C, n_head=4, block_size=T)
    out, weights = attn(torch.randn(B, T, C))
    assert out.shape == (B, T, C)  # same shape in and out, so blocks can be stacked
    assert weights.shape == (B, 4, T, T)


def test_attention_handles_sequences_shorter_than_block_size() -> None:
    head = CausalSelfAttentionHead(n_embd=C, head_size=16, block_size=T)
    out, weights = head(torch.randn(B, 5, C))
    assert out.shape == (B, 5, 16)
    assert weights.shape == (B, 5, 5)


def test_attention_weights_are_a_probability_distribution() -> None:
    head = CausalSelfAttentionHead(n_embd=C, head_size=16, block_size=T)
    _, weights = head(torch.randn(B, T, C))
    assert (weights >= 0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, T), atol=1e-6)  # each row sums to 1


def test_attention_weights_on_future_tokens_are_zero() -> None:
    """weights[i, j] must be exactly 0 whenever j > i (the upper triangle)."""
    head = CausalSelfAttentionHead(n_embd=C, head_size=16, block_size=T)
    _, weights = head(torch.randn(B, T, C))
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert (weights[:, future] == 0).all()
    # ... and the very first token can only attend to itself.
    assert torch.allclose(weights[:, 0, 0], torch.ones(B))


def test_causal_mask_prevents_looking_ahead() -> None:
    """Changing a token must not change the output at any EARLIER position.

    This is the behavioural definition of causality, tested on the full model:
    we alter the token at position 5 and check that the logits at positions
    0..4 are bit-for-bit unchanged while position 5 onward does change.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, block_size=T, n_layer=2, n_head=2, n_embd=C, dropout=0.0)
    model = GPT(cfg).eval()

    idx = torch.randint(0, 50, (1, T))
    changed = idx.clone()
    changed[0, 5] = (changed[0, 5] + 1) % 50  # a different token at position 5

    logits_a, _ = model(idx)
    logits_b, _ = model(changed)

    assert torch.equal(logits_a[:, :5], logits_b[:, :5]), "an earlier position saw the future!"
    assert not torch.allclose(logits_a[:, 5:], logits_b[:, 5:]), "later positions should change"


def test_without_mask_the_future_would_leak() -> None:
    """Sanity check of the test above: remove the mask and information DOES leak."""
    torch.manual_seed(0)
    head = CausalSelfAttentionHead(n_embd=C, head_size=16, block_size=T)
    head.mask.fill_(True)  # allow every token to see every other token

    x = torch.randn(1, T, C)
    changed = x.clone()
    changed[0, 5] += 1.0

    out_a, _ = head(x)
    out_b, _ = head(changed)
    assert not torch.allclose(out_a[:, :5], out_b[:, :5])


def test_layernorm_matches_pytorch() -> None:
    """Our hand-written LayerNorm must agree with torch.nn.LayerNorm."""
    x = torch.randn(B, T, C) * 3 + 1
    ours = LayerNorm(C, bias=True)
    reference = torch.nn.LayerNorm(C)
    assert torch.allclose(ours(x), reference(x), atol=1e-5)
    # after normalisation each token vector has mean 0 and variance 1
    assert torch.allclose(ours(x).mean(dim=-1), torch.zeros(B, T), atol=1e-5)
    assert torch.allclose(ours(x).var(dim=-1, unbiased=False), torch.ones(B, T), atol=1e-3)
