"""Serializable run configuration; architecture options belong to registered models."""

import math
from dataclasses import dataclass, field

from ml.data.schema import integer, require


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "mlp"
    seed: int = 42
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 20
    validation_split: float = 0.2
    validation_dataset: str | None = None
    weight_decay: float = 0.01
    patience: int = 5
    min_delta: float = 0.0
    model_options: dict = field(default_factory=dict)
    device: str = "cpu"
    num_workers: int = 0
    num_threads: int = 1

    def validate(self):
        integer(self.seed, "seed", 0, 2**32 - 1)
        for name in ("batch_size", "epochs", "num_threads"):
            integer(getattr(self, name), name, 1)
        integer(self.num_workers, "num_workers", 0)
        integer(self.patience, "patience", 0)
        for name in ("learning_rate", "weight_decay", "min_delta", "validation_split"):
            value = getattr(self, name)
            require(type(value) in (int, float) and math.isfinite(value), f"Invalid {name}")
        require(
            self.learning_rate > 0 and self.weight_decay >= 0 and self.min_delta >= 0,
            "Learning rate must be positive; weight decay and min_delta must be non-negative",
        )
        require(0 < self.validation_split < 1, "validation_split must lie between zero and one")
        require(isinstance(self.model, str) and bool(self.model), "Model ID is required")
        require(isinstance(self.model_options, dict), "model_options must be an object")
        require(
            self.validation_dataset is None
            or (isinstance(self.validation_dataset, str) and self.validation_dataset.strip()),
            "validation_dataset must be a nonempty path",
        )
        require(
            isinstance(self.device, str) and self.device in ("cpu", "cuda"),
            "device must be cpu or cuda",
        )
