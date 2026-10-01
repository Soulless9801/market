"""Vercel entry point. The shared HTTP adapter owns validation and checkpoint loading."""

import os
from pathlib import Path

from ml.inference.http import HostedInference, InferenceHandler

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models/deployment/market.pt"


class handler(InferenceHandler):
    backend = HostedInference(
        os.environ.get("MARKET_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
        os.environ.get("MARKET_MODEL_ALIAS", "market-side-v1"),
    )
