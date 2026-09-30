"""Versioned, self-contained inference checkpoints with exclusive atomic publication."""

import os
import tempfile
from pathlib import Path

import torch

from ml.data.preprocessing import Predictor, Standardizer
from ml.data.schema import require, validate_metadata
from ml.models import build_model

CHECKPOINT_VERSION = 1


def save_checkpoint(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link publishes atomically and refuses to overwrite an existing model.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> tuple[Predictor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(payload, dict)
        and set(payload)
        == {
            "checkpoint_version",
            "model_type",
            "model_version",
            "model_config",
            "state_dict",
            "training_config",
            "metadata",
            "validation_metadata",
            "split",
            "best_epoch",
            "history",
            "metrics",
            "environment",
            "output_type",
            "preprocessing_version",
        },
        "Unexpected checkpoint fields",
    )
    require(
        type(payload["checkpoint_version"]) is int
        and payload["checkpoint_version"] == CHECKPOINT_VERSION,
        "Unsupported checkpoint version",
    )
    require(payload["preprocessing_version"] == 1, "Unsupported preprocessing version")
    require(payload["output_type"] == "logits", "Unsupported checkpoint output type")
    metadata = validate_metadata(payload["metadata"])
    validate_metadata(payload["validation_metadata"])
    model = build_model(payload["model_type"], payload["model_version"], payload["model_config"])
    predictor = Predictor(Standardizer(metadata["inputShape"][0]), model)
    predictor.load_state_dict(payload["state_dict"], strict=True)
    require(
        all(torch.isfinite(t).all() for t in predictor.state_dict().values()),
        "Checkpoint contains non-finite parameters",
    )
    require(
        (predictor.normalizer.scale > 0).all(), "Checkpoint contains invalid normalization scales"
    )
    predictor.eval()
    with torch.no_grad():
        result = predictor(torch.zeros((1, metadata["inputShape"][0])))
    require(
        result.shape == (1, metadata["numClasses"]) and torch.isfinite(result).all(),
        "Checkpoint output differs from its metadata",
    )
    return predictor, payload
