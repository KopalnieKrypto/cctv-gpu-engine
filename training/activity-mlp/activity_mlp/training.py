"""Deterministic fixed-architecture activity-MLP training."""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

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


@dataclass
class TrainingOutcome:
    model: object
    best_epoch: int
    best_validation_accuracy: float
    best_validation_loss: float
    epochs_ran: int
    history: list[dict]


def print_epoch_heartbeat(metrics: dict, *, stream: TextIO = sys.stderr) -> None:
    """Emit one unbuffered progress line per completed training epoch."""
    print(
        f"training epoch={metrics['epoch']} "
        f"train_loss={metrics['train_loss']:.6f} "
        f"validation_loss={metrics['validation_loss']:.6f} "
        f"validation_accuracy={metrics['validation_accuracy']:.6f}",
        file=stream,
        flush=True,
    )


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


def train_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: TrainingConfig,
    *,
    progress: Callable[[dict], None] | None = None,
) -> TrainingOutcome:
    """Train only on train rows and select/stop only on validation rows."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for activity-MLP training; no CPU fallback")
    seed_everything(config.seed, torch_module=torch)
    device = torch.device("cuda")
    feature_mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    model = build_model(
        config,
        feature_mean=feature_mean,
        feature_std=feature_std,
        torch_module=torch,
    ).to(device)

    train_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(train_features), torch.as_tensor(train_labels)
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_x = torch.as_tensor(validation_features, device=device)
    validation_y = torch.as_tensor(validation_labels, device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_epoch = 0
    best_accuracy = -1.0
    best_loss = float("inf")
    best_state: dict[str, object] | None = None
    epochs_without_improvement = 0
    history: list[dict] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.forward_logits(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(batch_labels)
            train_count += len(batch_labels)

        model.eval()
        with torch.no_grad():
            validation_logits = model.forward_logits(validation_x)
            validation_loss = float(loss_fn(validation_logits, validation_y))
            validation_predictions = validation_logits.argmax(dim=1)
            validation_accuracy = float((validation_predictions == validation_y).float().mean())
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_metrics)
        if progress is not None:
            progress(epoch_metrics)

        improved = validation_accuracy > best_accuracy + config.minimum_delta or (
            abs(validation_accuracy - best_accuracy) <= config.minimum_delta
            and validation_loss < best_loss - config.minimum_delta
        )
        if improved:
            best_epoch = epoch
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingOutcome(
        model=model,
        best_epoch=best_epoch,
        best_validation_accuracy=best_accuracy,
        best_validation_loss=best_loss,
        epochs_ran=len(history),
        history=history,
    )


def export_onnx(model: object, output_path: str | Path) -> None:
    """Export the frozen model with dynamic batch and softmax probabilities."""
    import torch

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    example = torch.zeros((1, FEATURE_DIM), dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["features"],
        output_names=["probabilities"],
        dynamic_axes={"features": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )


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
