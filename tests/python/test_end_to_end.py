"""Phase 8: exercise public CLIs and real process boundaries with disposable artifacts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ml.data.dataset import MarketDataset
from ml.training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def run(command, *, env=None):
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.skipif(os.name != "posix", reason="Real stop/kill failure test requires POSIX signals")
@pytest.mark.parametrize(("architecture", "layout"), [("mlp", "mlp"), ("cnn", "cnn")])
def test_complete_workflow_is_reproducible_isolated_and_fails_closed(
    tmp_path, architecture, layout
):
    env = {**os.environ, "TSX_TSCONFIG_PATH": "tsconfig.scripts.json"}
    datasets = [tmp_path / name for name in ("export-a", "export-b")]
    for directory in datasets:
        run(
            [
                "node",
                "--import",
                "tsx",
                "scripts/dataset.ts",
                layout,
                "--seed",
                "42",
                "--samples",
                "256",
                "--trajectories",
                "8",
                "--output",
                str(directory),
            ],
            env=env,
        )
    for name in ("dataset.jsonl", "metadata.json"):
        assert (datasets[0] / name).read_bytes() == (datasets[1] / name).read_bytes()

    checkpoints = [tmp_path / name for name in ("trained-a.pt", "trained-b.pt")]
    for checkpoint in checkpoints:
        run(
            [
                sys.executable,
                "-m",
                "ml.training.train",
                "--model",
                architecture,
                "--dataset",
                str(datasets[0]),
                "--output",
                str(checkpoint),
                "--epochs",
                "3",
                "--seed",
                "42",
                "--batch-size",
                "32",
                "--num-threads",
                "1",
            ]
        )
    first, first_payload = load_checkpoint(checkpoints[0])
    repeated, repeated_payload = load_checkpoint(checkpoints[1])
    assert first_payload["history"] == repeated_payload["history"]
    assert first_payload["split"] == repeated_payload["split"]
    assert set(first_payload["split"]["train_trajectories"]).isdisjoint(
        first_payload["split"]["validation_trajectories"]
    )
    for key, value in first.state_dict().items():
        assert torch.equal(value, repeated.state_dict()[key]), key
    dataset = MarketDataset(datasets[0])
    inputs = torch.stack([inputs for inputs, _ in dataset])
    with torch.inference_mode():
        expected = first(inputs)
        assert torch.equal(expected, repeated(inputs))
        # Match the server's single-row batch shape. Batched BLAS kernels can round
        # differently from single-row matrix products even with identical weights.
        single_prediction = first(inputs[:1])[0].tolist()
    fixture = tmp_path / "predictions.json"
    fixture.write_text(json.dumps({"input": inputs[0].tolist(), "expected": single_prediction}))

    # Test-only interpreter startup hook captures the actual child PID. No production
    # process handles, alternate models or altered predictions are needed for crash injection.
    hook = tmp_path / "python-hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['MARKET_TEST_PID_FILE']).write_text(str(os.getpid()))\n"
    )
    pid_file = tmp_path / "python.pid"
    result = run(
        [
            "node",
            "--import",
            "tsx",
            "tests/fixtures/end-to-end.ts",
            sys.executable,
            str(checkpoints[0]),
            str(fixture),
            str(pid_file),
        ],
        env={
            **env,
            "MARKET_TEST_PID_FILE": str(pid_file),
            "PYTHONPATH": str(hook) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
    )
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["mlTrades"] > 0
    assert report["predictionsChecked"] >= 100
    assert report["pythonCrashRejected"] is True
    assert report["sameProcess"] is True
    assert report["runtimeReplayEqual"] is True
    (tmp_path / "validation-report.json").write_text(
        json.dumps(
            {
                **report,
                "datasetSha256": dataset.metadata["sha256"],
                "datasetRows": len(dataset),
                "trainingRepeatEqual": True,
                "checkpointPredictionsEqual": True,
                "architecture": architecture,
                "featureLayout": layout,
            },
            indent=2,
        )
        + "\n"
    )
