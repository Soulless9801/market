"""Deployment aliases are independent of checkpoint architecture and feature IDs."""

import re
from collections.abc import Mapping
from pathlib import Path

import torch

from ml.data.schema import parse_json, require
from ml.training.checkpoint import load_checkpoint


def validate_alias(alias: object) -> str:
    require(
        isinstance(alias, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", alias),
        "Model alias must be 1-128 letters, digits, dots, underscores or hyphens",
    )
    return alias


def read_registry(path: str | Path) -> dict[str, Path]:
    """Read alias-to-path JSON; relative checkpoints are relative to this file."""
    path = Path(path).resolve()
    entries = parse_json(path.read_text(encoding="utf-8"))
    require(isinstance(entries, dict) and bool(entries), "Registry must be a nonempty object")
    checkpoints = {}
    for alias, checkpoint in entries.items():
        validate_alias(alias)
        require(isinstance(checkpoint, str) and bool(checkpoint.strip()), "Invalid checkpoint path")
        checkpoints[alias] = path.parent / checkpoint
    return checkpoints


class CheckpointRegistry:
    """Load each distinct checkpoint once at startup; serve raw public feature vectors."""

    def __init__(self, checkpoints: Mapping[str, str | Path]):
        require(bool(checkpoints), "At least one checkpoint is required")
        self._models = {}
        loaded = {}
        for alias, checkpoint in checkpoints.items():
            validate_alias(alias)
            path = Path(checkpoint).resolve()
            if path not in loaded:
                loaded[path] = load_checkpoint(path)
            self._models[alias] = loaded[path]

    def predict(self, alias: str, inputs: object) -> list[float]:
        if alias not in self._models:
            raise ValueError(f"Unknown model: {alias}")
        predictor, payload = self._models[alias]
        metadata = payload["metadata"]
        # Dataset contract v1 is flat, including layouts consumed by future CNNs.
        width = metadata["inputShape"][0]
        require(isinstance(inputs, list) and len(inputs) == width, f"Expected {width} input values")
        require(all(type(value) in (int, float) for value in inputs), "Input must contain numbers")
        try:
            tensor = torch.tensor([inputs], dtype=torch.float32)
        except (ValueError, TypeError, OverflowError, RuntimeError) as error:
            raise ValueError("Input cannot be represented as float32") from error
        require(torch.isfinite(tensor).all(), "Input must contain finite float32 values")
        with torch.inference_mode():
            prediction = predictor(tensor)
        require(
            isinstance(prediction, torch.Tensor)
            and prediction.is_floating_point()
            and prediction.shape == (1, metadata["numClasses"])
            and torch.isfinite(prediction).all(),
            "Model returned invalid prediction values or shape",
        )
        # Preserve the checkpoint's logits and class ordering; do not convert to probabilities.
        return prediction[0].tolist()
