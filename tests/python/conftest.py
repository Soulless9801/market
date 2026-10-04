"""Small real TypeScript exports exercise the language boundary without training artifacts."""

import os
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def exports(tmp_path_factory):
    root = tmp_path_factory.mktemp("typescript-exports")
    for layout in ("mlp", "cnn", "orderbook"):
        subprocess.run(
            [
                "node",
                "--import",
                "tsx",
                "scripts/dataset.ts",
                layout,
                "--samples",
                "32",
                "--trajectories",
                "4",
                "--output",
                str(root / layout),
            ],
            cwd=ROOT,
            env={**os.environ, "TSX_TSCONFIG_PATH": "tsconfig.scripts.json"},
            check=True,
            capture_output=True,
            text=True,
        )
    return root


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)
