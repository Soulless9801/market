"""Built-in registrations. Add new architecture modules here, not in the trainer."""

from . import mlp as mlp
from .registry import (
    ModelDefinition,
    build_model,
    configure_model,
    definition_for,
    model_names,
    register_model,
)

__all__ = [
    "ModelDefinition",
    "build_model",
    "configure_model",
    "definition_for",
    "model_names",
    "register_model",
]
