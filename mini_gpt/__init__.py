"""mini_gpt: a GPT-style language model written from scratch in PyTorch.

Modules
-------
config     dataclasses that hold every hyperparameter, loaded from YAML
tokenizer  character-level and byte-pair-encoding (BPE) tokenizers
data       turn a .txt file into token-ID arrays and sample training batches
model      the transformer itself (attention, blocks, GPT)
utils      seeding, device detection, LR schedule, checkpoints, logging
"""

__version__ = "0.1.0"
