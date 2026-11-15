"""Tests for the full GPT model, sampling helpers, LR schedule and config loading."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from mini_gpt.config import ModelConfig, load_config
from mini_gpt.data import get_batch
from mini_gpt.model import GPT, sample_next_token, top_k_filter, top_p_filter
from mini_gpt.utils import get_lr

CFG = ModelConfig(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_forward_shapes_and_loss() -> None:
    model = GPT(CFG)
    idx = torch.randint(0, 65, (4, 16))
    logits, loss = model(idx)
    assert logits.shape == (4, 16, 65)
    assert loss is None  # no targets given

    _, loss = model(idx, idx)
    assert loss.shape == ()  # a single number


def test_initial_loss_is_close_to_uniform_guessing() -> None:
    """An untrained model should be about as good as guessing uniformly: loss = ln(V)."""
    torch.manual_seed(0)
    model = GPT(CFG)
    idx = torch.randint(0, 65, (32, 16))
    targets = torch.randint(0, 65, (32, 16))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(65)) < 0.3


def test_weight_tying_shares_one_tensor() -> None:
    tied = GPT(CFG)
    assert tied.lm_head.weight is tied.token_emb.weight

    untied = GPT(ModelConfig(**{**CFG.__dict__, "tie_weights": False}))
    assert untied.lm_head.weight is not untied.token_emb.weight
    # untying adds exactly one extra (V x C) matrix
    assert untied.num_params() - tied.num_params() == 65 * 32


def test_no_positional_embedding_option() -> None:
    model = GPT(ModelConfig(**{**CFG.__dict__, "use_pos_emb": False}))
    assert model.pos_emb is None
    logits, _ = model(torch.randint(0, 65, (2, 16)))
    assert logits.shape == (2, 16, 65)


def test_sequence_longer_than_block_size_raises() -> None:
    model = GPT(CFG)
    with pytest.raises(ValueError):
        model(torch.randint(0, 65, (1, 17)))


def test_attention_maps_shape() -> None:
    model = GPT(CFG).eval()
    maps = model.attention_maps(torch.randint(0, 65, (1, 10)))
    assert len(maps) == CFG.n_layer
    assert all(m.shape == (1, CFG.n_head, 10, 10) for m in maps)


def test_generate_extends_the_sequence() -> None:
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=40, temperature=1.0, top_k=10, top_p=0.9)
    assert out.shape == (1, 41)  # longer than block_size: the context gets cropped
    assert out.min() >= 0 and out.max() < 65


def test_top_k_filter_keeps_k_tokens() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = top_k_filter(logits, k=2)
    assert torch.isfinite(filtered).sum() == 2
    assert filtered[0, 1] == 5.0 and filtered[0, 4] == 4.0


def test_top_p_filter_keeps_smallest_nucleus() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    assert torch.isfinite(top_p_filter(logits, p=0.7)).sum() == 2  # 0.5 + 0.3 >= 0.7
    assert torch.isfinite(top_p_filter(logits, p=0.4)).sum() == 1  # 0.5 alone covers 0.4
    assert torch.isfinite(top_p_filter(logits, p=0.9)).sum() == 3
    # works when the input is not already sorted
    shuffled = torch.tensor([[0.05, 0.5, 0.15, 0.3]]).log()
    kept = torch.isfinite(top_p_filter(shuffled, p=0.7))
    assert kept.tolist() == [[False, True, False, True]]


def test_temperature_zero_is_greedy() -> None:
    logits = torch.tensor([[0.1, 2.0, 0.3], [3.0, 0.0, 1.0]])
    assert sample_next_token(logits, temperature=0.0).tolist() == [[1], [0]]


def test_get_batch_targets_are_inputs_shifted_by_one() -> None:
    data = torch.arange(1000)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)  # because data is 0, 1, 2, ...


def test_lr_schedule_warmup_then_cosine() -> None:
    kwargs = dict(max_lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
    assert get_lr(0, **kwargs) < get_lr(50, **kwargs) < get_lr(99, **kwargs)  # warming up
    assert get_lr(99, **kwargs) == pytest.approx(1e-3)  # peak at the end of warmup
    assert get_lr(100, **kwargs) > get_lr(500, **kwargs) > get_lr(900, **kwargs)  # decaying
    assert get_lr(1000, **kwargs) == pytest.approx(1e-4)  # ends at min_lr


def test_configs_load() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ["small.yaml", "tiny_cpu.yaml", "ablation.yaml"]:
        cfg = load_config(root / "configs" / name)
        assert cfg.model.n_embd % cfg.model.n_head == 0
        assert isinstance(cfg.train.learning_rate, float)

    cfg = load_config(root / "configs" / "small.yaml", ["train.max_steps=7", "model.n_head=2"])
    assert cfg.train.max_steps == 7 and cfg.model.n_head == 2

    with pytest.raises(ValueError):
        load_config(root / "configs" / "small.yaml", ["model.not_an_option=1"])
