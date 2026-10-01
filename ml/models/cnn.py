"""Compact temporal convolutions over an ordered, flat feature vector."""

import math

from torch import nn

from ml.data.schema import integer, require

from .registry import ModelDefinition, register_model


class CNN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        require(
            set(config)
            == {"input_shape", "num_classes", "channels", "kernel_size", "pool_size", "dropout"},
            "Invalid CNN config",
        )
        shape = config["input_shape"]
        require(isinstance(shape, list) and len(shape) == 1, "CNN expects a flat feature vector")
        self.input_size = integer(shape[0], "input_size", 1)
        output_size = integer(config["num_classes"], "num_classes", 2)
        channels = config["channels"]
        require(
            isinstance(channels, list) and len(channels) > 0, "channels must be a nonempty list"
        )
        for width in channels:
            integer(width, "channel width", 1)
        kernel_size = integer(config["kernel_size"], "kernel_size", 1)
        require(kernel_size % 2 == 1, "kernel_size must be odd to preserve sequence length")
        pool_size = integer(config["pool_size"], "pool_size", 1, self.input_size)
        dropout = config["dropout"]
        require(
            type(dropout) in (int, float) and math.isfinite(dropout) and 0 <= dropout < 1,
            "dropout must be finite and in [0, 1)",
        )

        layers = []
        previous = 1
        for width in channels:
            layers.extend([
                nn.Conv1d(previous, width, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ])
            previous = width
        self.convolutions = nn.Sequential(*layers)
        # Retain coarse position information while bounding the classifier's size.
        # Dropout is active during training only; checkpoint loading selects eval mode.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(pool_size),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(previous * pool_size, output_size),
        )

    def forward(self, inputs):
        require(
            inputs.ndim == 2 and inputs.shape[1] == self.input_size,
            f"Expected [batch, {self.input_size}] input",
        )
        # Dataset v1 and the shared standardizer operate on [batch, features].
        # Only this architecture adds Conv1d's channel axis; column order is preserved.
        return self.classifier(self.convolutions(inputs.unsqueeze(1)))  # Raw class logits.


def configure(metadata: dict, options: dict) -> dict:
    require(
        set(options) <= {"channels", "kernel_size", "pool_size", "dropout"},
        "Unknown CNN options",
    )
    return {
        "input_shape": metadata["inputShape"],
        "num_classes": metadata["numClasses"],
        "channels": options.get("channels", [8, 16]),
        "kernel_size": options.get("kernel_size", 3),
        "pool_size": options.get("pool_size", min(4, metadata["inputShape"][0])),
        "dropout": options.get("dropout", 0.2),
    }


register_model("cnn", ModelDefinition(version=1, configure=configure, create=CNN))
