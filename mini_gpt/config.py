"""Configuration: every hyperparameter of the project lives here.

A *hyperparameter* is a number we choose by hand (model size, learning rate,
...), as opposed to a *parameter*, which the model learns during training.

The settings are grouped into three dataclasses (model / data / train) and are
loaded from a YAML file such as ``configs/small.yaml``. Keeping them in one
place means an experiment is fully described by one small text file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Shape of the transformer."""

    vocab_size: int = 65  # number of distinct tokens (set automatically from the tokenizer)
    block_size: int = 256  # context length: how many tokens the model can look back at
    n_layer: int = 6  # number of transformer blocks stacked on top of each other
    n_head: int = 6  # attention heads per block
    n_embd: int = 384  # embedding dimension (the width of every token vector)
    dropout: float = 0.2  # probability of zeroing an activation during training
    bias: bool = False  # use bias terms in Linear / LayerNorm layers?
    use_pos_emb: bool = True  # add learned positional embeddings? (False = ablation)
    tie_weights: bool = True  # share the token-embedding matrix with the output layer?


@dataclass
class DataConfig:
    """Where the text comes from and how it becomes token IDs."""

    input_file: str = "data/tinyshakespeare.txt"  # any UTF-8 .txt file works
    tokenizer: str = "char"  # "char" or "bpe"
    bpe_vocab_size: int = 512  # only used when tokenizer == "bpe"
    val_fraction: float = 0.1  # last 10% of the text is held out for validation


@dataclass
class TrainConfig:
    """Everything about the optimisation loop."""

    seed: int = 1337  # random seed, for reproducibility
    device: str = "auto"  # "auto", "cuda" or "cpu"
    amp: bool = True  # mixed precision (float16) on GPU; ignored on CPU
    batch_size: int = 64  # sequences per training step
    max_steps: int = 5000  # total number of optimizer steps
    learning_rate: float = 1e-3  # peak learning rate
    min_lr: float = 1e-4  # learning rate at the end of the cosine decay
    warmup_steps: int = 100  # linear warmup length
    weight_decay: float = 0.1  # AdamW weight decay (applied to weight matrices only)
    beta1: float = 0.9  # AdamW momentum coefficients
    beta2: float = 0.99
    grad_clip: float = 1.0  # clip the gradient norm to this value (0 disables)
    eval_interval: int = 250  # evaluate train/val loss every N steps
    eval_batches: int = 100  # batches averaged per evaluation
    sample_interval: int = 500  # print a generated sample every N steps
    sample_tokens: int = 300  # length of those samples
    sample_prompt: str = "\n"  # text the samples start from
    out_dir: str = "checkpoints/small"  # checkpoint, log.csv and samples go here


@dataclass
class Config:
    """The full configuration of one training run."""

    name: str = "small"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain nested dicts (so it can be saved inside a checkpoint)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        """Build a Config from nested dicts, rejecting unknown keys (catches typos)."""
        return cls(
            name=raw.get("name", "run"),
            model=_build(ModelConfig, raw.get("model", {})),
            data=_build(DataConfig, raw.get("data", {})),
            train=_build(TrainConfig, raw.get("train", {})),
        )


def _build(cls: type, values: dict[str, Any]) -> Any:
    """Create dataclass ``cls`` from ``values``, converting each value to the field's type."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} option(s): {sorted(unknown)}")
    converted = {}
    for key, value in values.items():
        default = known[key].default
        # YAML reads "1e-3" as a string; convert using the type of the default value.
        if isinstance(default, bool):
            converted[key] = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
        elif isinstance(default, (int, float)):
            converted[key] = type(default)(value)
        else:
            converted[key] = value
    return cls(**converted)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config file and apply optional command-line overrides.

    Args:
        path: path to a YAML file, e.g. ``configs/small.yaml``.
        overrides: strings like ``"train.max_steps=200"`` or ``"model.n_head=1"``.

    Returns:
        A fully populated :class:`Config`.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"Override must look like section.key=value, got: {item!r}")
        dotted_key, value = item.split("=", 1)
        section, key = dotted_key.split(".", 1)
        raw.setdefault(section, {})[key] = yaml.safe_load(value)

    return Config.from_dict(raw)
