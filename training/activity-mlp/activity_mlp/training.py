"""Deterministic fixed-architecture activity-MLP training."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

from pipeline.activity_features import ACTIVITY_CLASSES, FEATURE_DIM


@dataclass(frozen=True)
class TrainingConfig:
    """The single reviewed configuration; issue #34 forbids broad search."""

    seed: int = 3407
    hidden_sizes: tuple[int, int] = (128, 64)
    dropout: float = 0.15
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    max_epochs: int = 500
    early_stopping_patience: int = 50
    minimum_delta: float = 0.0001


def model_spec(config: TrainingConfig) -> dict:
    """Describe the frozen architecture without importing a training framework."""
    return {
        "input_dimension": FEATURE_DIM,
        "hidden_layers": [
            {"units": units, "activation": "relu", "dropout": config.dropout}
            for units in config.hidden_sizes
        ],
        "output_dimension": len(ACTIVITY_CLASSES),
        "output_activation": "softmax",
    }


def seed_everything(seed: int, *, torch_module: object | None = None) -> None:
    """Seed every training RNG and request deterministic CUDA kernels."""
    if torch_module is None:
        import torch as torch_module

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
