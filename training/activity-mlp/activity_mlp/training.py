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


def build_model(
    config: TrainingConfig,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    torch_module: object | None = None,
) -> object:
    """Build the fixed two-layer model with normalization inside its graph."""
    if torch_module is None:
        import torch as torch_module

    nn = torch_module.nn

    class ActivityMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            safe_std = np.where(feature_std > 1e-6, feature_std, 1.0).astype(np.float32)
            self.register_buffer("feature_mean", torch_module.as_tensor(feature_mean))
            self.register_buffer("feature_std", torch_module.as_tensor(safe_std))
            first_hidden, second_hidden = config.hidden_sizes
            self.hidden = nn.Sequential(
                nn.Linear(FEATURE_DIM, first_hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(first_hidden, second_hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.output = nn.Linear(second_hidden, len(ACTIVITY_CLASSES))

        def forward_logits(self, features):
            normalized = (features - self.feature_mean) / self.feature_std
            return self.output(self.hidden(normalized))

        def forward(self, features):
            return torch_module.softmax(self.forward_logits(features), dim=1)

    return ActivityMLP()


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
