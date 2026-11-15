"""The classic ML sanity check: a correct model + training loop can memorise one batch.

If the loss does not go to ~0 on a single small batch, something is broken
(wrong targets, gradients not flowing, a bug in attention, ...). This catches a
whole class of bugs that shape tests cannot.
"""

from __future__ import annotations

import torch

from mini_gpt.config import ModelConfig
from mini_gpt.data import get_batch
from mini_gpt.model import GPT
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.utils import configure_optimizer, set_seed

TEXT = "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer. " * 4


def test_model_can_memorise_one_batch() -> None:
    set_seed(0)
    tokenizer = CharTokenizer.train(TEXT)
    data = torch.tensor(tokenizer.encode(TEXT), dtype=torch.long)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0
    )
    model = GPT(cfg)
    optimizer = configure_optimizer(model, learning_rate=3e-3, weight_decay=0.0, betas=(0.9, 0.99))

    x, y = get_batch(data, cfg.block_size, batch_size=4)  # ONE fixed batch, reused every step
    _, first_loss = model(x, y)

    for _ in range(300):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert first_loss.item() > 2.5  # starts near ln(vocab_size)
    assert loss.item() < 0.1, f"model failed to memorise one batch (loss {loss.item():.3f})"

    # Having memorised the batch, greedy generation should reproduce it exactly.
    model.eval()
    # Start from the first 8 tokens of sequence 0 and let the model write the other 24.
    generated = model.generate(x[:1, :8], max_new_tokens=24, temperature=0.0)
    assert torch.equal(generated[0], x[0])
