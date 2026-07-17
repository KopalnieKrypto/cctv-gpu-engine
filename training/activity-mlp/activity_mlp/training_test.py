"""Reproducibility contract for the deliberately small fixed MLP."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from activity_mlp.training import (
    TrainingConfig,
    build_model,
    export_onnx,
    model_spec,
    seed_everything,
    train_model,
)
from pipeline.activity_features import FEATURE_DIM


def test_training_configuration_is_frozen_without_a_hyperparameter_search() -> None:
    assert asdict(TrainingConfig()) == {
        "seed": 3407,
        "hidden_sizes": (128, 64),
        "dropout": 0.15,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "max_epochs": 500,
        "early_stopping_patience": 50,
        "minimum_delta": 0.0001,
    }


def test_training_seed_controls_numpy_and_all_torch_cuda_generators() -> None:
    class FakeCuda:
        seeds: list[int] = []

        @classmethod
        def manual_seed_all(cls, seed: int) -> None:
            cls.seeds.append(seed)

    class FakeCudnn:
        benchmark = True
        deterministic = False

    class FakeBackends:
        cudnn = FakeCudnn()

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()
        seeds: list[int] = []
        deterministic_flags: list[bool] = []

        @classmethod
        def manual_seed(cls, seed: int) -> None:
            cls.seeds.append(seed)

        @classmethod
        def use_deterministic_algorithms(cls, enabled: bool) -> None:
            cls.deterministic_flags.append(enabled)

    seed_everything(3407, torch_module=FakeTorch)
    first = np.random.random(3)
    seed_everything(3407, torch_module=FakeTorch)
    second = np.random.random(3)

    np.testing.assert_array_equal(first, second)
    assert FakeTorch.seeds == [3407, 3407]
    assert FakeCuda.seeds == [3407, 3407]
    assert FakeTorch.deterministic_flags == [True, True]
    assert FakeTorch.backends.cudnn.benchmark is False
    assert FakeTorch.backends.cudnn.deterministic is True


def test_model_spec_is_two_small_relu_dropout_layers_and_softmax_output() -> None:
    assert model_spec(TrainingConfig()) == {
        "input_dimension": 115,
        "hidden_layers": [
            {"units": 128, "activation": "relu", "dropout": 0.15},
            {"units": 64, "activation": "relu", "dropout": 0.15},
        ],
        "output_dimension": 4,
        "output_activation": "softmax",
    }


@pytest.mark.gpu
def test_model_runs_the_frozen_softmax_architecture_on_cuda() -> None:
    import torch

    assert torch.cuda.is_available()
    model = build_model(
        TrainingConfig(),
        feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        feature_std=np.ones(FEATURE_DIM, dtype=np.float32),
    ).cuda()

    probabilities = model(torch.zeros((2, FEATURE_DIM), device="cuda"))

    assert tuple(probabilities.shape) == (2, 4)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2, device="cuda"))


@pytest.mark.gpu
def test_seeded_cuda_training_is_reproducible() -> None:
    import torch

    features = np.zeros((32, FEATURE_DIM), dtype=np.float32)
    labels = np.arange(32, dtype=np.int64) % 4
    features[np.arange(32), labels] = 1.0
    config = replace(
        TrainingConfig(),
        max_epochs=4,
        early_stopping_patience=4,
        batch_size=8,
    )

    first = train_model(features[:24], labels[:24], features[24:], labels[24:], config)
    second = train_model(features[:24], labels[:24], features[24:], labels[24:], config)
    validation = torch.as_tensor(features[24:], device="cuda")
    first.model.eval()
    second.model.eval()

    with torch.no_grad():
        first_probabilities = first.model(validation)
        second_probabilities = second.model(validation)
    torch.testing.assert_close(first_probabilities, second_probabilities, rtol=0, atol=0)
    assert first.best_epoch == second.best_epoch
    assert first.best_validation_accuracy == second.best_validation_accuracy


@pytest.mark.gpu
def test_onnx_export_matches_torch_softmax_and_is_under_10_mb(tmp_path) -> None:
    import onnx
    import onnxruntime as ort
    import torch

    model = build_model(
        TrainingConfig(),
        feature_mean=np.linspace(0.0, 1.0, FEATURE_DIM, dtype=np.float32),
        feature_std=np.linspace(1.0, 2.0, FEATURE_DIM, dtype=np.float32),
    ).cuda()
    model.eval()
    sample = np.random.default_rng(3407).normal(size=(3, FEATURE_DIM)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.as_tensor(sample, device="cuda")).cpu().numpy()
    model_path = tmp_path / "activity-mlp.onnx"

    export_onnx(model, model_path)

    assert model_path.stat().st_size <= 10 * 1024 * 1024
    onnx.checker.check_model(onnx.load(model_path))
    ort.preload_dlls(cuda=True, cudnn=True)
    session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    assert "CUDAExecutionProvider" in session.get_providers()
    actual = session.run(None, {"features": sample})[0]
    # CUDA Torch and CUDA ORT select different FP32 GEMM kernels. Keep the
    # tolerance narrow enough that it cannot alter a meaningful class margin.
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)
