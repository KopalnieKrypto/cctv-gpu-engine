"""Reproducibility contract for the deliberately small fixed MLP."""

from dataclasses import asdict

import numpy as np

from activity_mlp.training import TrainingConfig, model_spec, seed_everything


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
