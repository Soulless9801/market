"""Exercise the actual Node client against a freshly trained Python checkpoint."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ml.data.dataset import MarketDataset
from ml.training.config import TrainingConfig
from ml.training.train import train

ROOT = Path(__file__).resolve().parents[2]


def test_typescript_client_uses_real_python_process(exports, tmp_path):
    checkpoint = tmp_path / "checkpoint with spaces.pt"
    model, _ = train(exports / "mlp", checkpoint, TrainingConfig(epochs=1))
    inputs, _ = next(iter(MarketDataset(exports / "mlp")))
    with torch.inference_mode():
        expected = model(inputs.unsqueeze(0))[0].tolist()
    fixture = tmp_path / "prediction.json"
    fixture.write_text(json.dumps({"input": inputs.tolist(), "expected": expected}))
    result = subprocess.run(
        [
            "node",
            "--import",
            "tsx",
            "tests/fixtures/python-client-smoke.ts",
            sys.executable,
            str(checkpoint),
            str(fixture),
        ],
        cwd=ROOT,
        env={**os.environ, "TSX_TSCONFIG_PATH": "tsconfig.scripts.json"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Python client integration passed" in result.stdout


@pytest.mark.parametrize("mode", ["dev", "preview"])
def test_local_marketview_uses_python_through_vite(exports, tmp_path, mode):
    checkpoint = tmp_path / "local-market.pt"
    train(exports / "mlp", checkpoint, TrainingConfig(epochs=1))
    result = subprocess.run(
        ["node", "--import", "tsx", "tests/fixtures/local-market-smoke.ts", mode],
        cwd=ROOT,
        env={
            **os.environ,
            "TSX_TSCONFIG_PATH": "tsconfig.scripts.json",
            "MARKET_CHECKPOINT": str(checkpoint),
            "MARKET_PYTHON": sys.executable,
            "MARKET_MODEL_ALIAS": "independent-ui-deployment",
        },
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Local MarketView Python integration passed" in result.stdout
