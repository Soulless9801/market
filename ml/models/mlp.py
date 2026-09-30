"""Small baseline used to validate the training pipeline, not a TypeScript port."""

from torch import nn

from ml.data.schema import integer, require

from .registry import ModelDefinition, register_model


class MLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        require(set(config) == {"input_shape", "num_classes", "hidden_sizes"}, "Invalid MLP config")
        shape = config["input_shape"]
        require(isinstance(shape, list) and len(shape) == 1, "MLP expects a flat feature vector")
        self.input_size = integer(shape[0], "input_size", 1)
        output_size = integer(config["num_classes"], "num_classes", 2)
        hidden = config["hidden_sizes"]
        require(
            isinstance(hidden, list) and len(hidden) > 0, "hidden_sizes must be a nonempty list"
        )
        for width in hidden:
            integer(width, "hidden width", 1)
        layers = []
        previous = self.input_size
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        require(
            inputs.ndim == 2 and inputs.shape[1] == self.input_size,
            f"Expected [batch, {self.input_size}] input",
        )
        return self.layers(inputs)  # Logits; CrossEntropyLoss applies log-softmax.


def configure(metadata: dict, options: dict) -> dict:
    # Architecture defaults live here, outside shared training and inference. This
    # MLP can train on any registered flat layout; the layout ID need not be "mlp".
    require(set(options) <= {"hidden_sizes"}, "Unknown MLP options")
    return {
        "input_shape": metadata["inputShape"],
        "num_classes": metadata["numClasses"],
        "hidden_sizes": options.get("hidden_sizes", [32, 16]),
    }


register_model("mlp", ModelDefinition(version=1, configure=configure, create=MLP))
