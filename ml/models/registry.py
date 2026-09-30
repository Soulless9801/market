"""Explicit factories, with versioned configurations saved alongside their weights."""

import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from torch import nn

from ml.data.schema import integer, require


@dataclass(frozen=True)
class ModelDefinition:
    version: int
    configure: Callable[[dict, dict], dict]
    create: Callable[[dict], nn.Module]


_REGISTRY: dict[str, ModelDefinition] = {}


def register_model(name: str, definition: ModelDefinition) -> None:
    require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_-]*", name), "Invalid model ID")
    require(name not in _REGISTRY, f"Model already registered: {name}")
    integer(definition.version, "model version", 1)
    require(callable(definition.configure) and callable(definition.create), "Invalid model factory")
    _REGISTRY[name] = definition


def model_names() -> list[str]:
    return sorted(_REGISTRY)


def definition_for(name: str) -> ModelDefinition:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {', '.join(model_names())}")
    return _REGISTRY[name]


def configure_model(name: str, metadata: dict, options: dict) -> dict:
    return definition_for(name).configure(deepcopy(metadata), deepcopy(options))


def build_model(name: str, version: int, config: dict) -> nn.Module:
    definition = definition_for(name)
    require(type(version) is int and version == definition.version, f"Unsupported {name} version")
    model = definition.create(deepcopy(config))
    require(isinstance(model, nn.Module), "Model factory must return torch.nn.Module")
    return model
