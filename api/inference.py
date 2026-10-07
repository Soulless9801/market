"""Vercel entry point. The shared HTTP adapter owns validation and checkpoint loading."""

import os
from pathlib import Path

from ml.inference.http import HostedInference, InferenceHandler

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models/deployment/market.pt"
DEFAULT_REGISTRY = DEFAULT_CHECKPOINT.parent / "registry.json"


class handler(InferenceHandler):
    backend = HostedInference(
        os.environ.get("MARKET_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
        os.environ.get("MARKET_MODEL_ALIAS", "custom-model"),
    ) if os.environ.get("MARKET_CHECKPOINT") else HostedInference(
        registry_path=os.environ.get("MARKET_REGISTRY", str(DEFAULT_REGISTRY)),
        alias=os.environ.get("MARKET_MODEL_ALIAS"),
    )
