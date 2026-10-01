import copy
import io
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from ml.data.dataset import MarketDataset
from ml.inference.protocol import respond
from ml.inference.registry import CheckpointRegistry, read_registry
from ml.inference.server import MAX_REQUEST_BYTES, serve
from ml.models import ModelDefinition, register_model
from ml.training.checkpoint import load_checkpoint, save_checkpoint
from ml.training.config import TrainingConfig
from ml.training.train import train

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def checkpoint(exports, tmp_path_factory):
    path = tmp_path_factory.mktemp("inference") / "saved model.pt"
    torch.set_num_threads(1)
    train(exports / "mlp", path, TrainingConfig(epochs=1))
    return path


@pytest.fixture
def registry(checkpoint):
    return CheckpointRegistry({"deployment-v1": checkpoint})


@pytest.fixture
def predict_request(exports):
    inputs, _ = next(iter(MarketDataset(exports / "mlp")))
    return {"id": 0, "type": "predict", "model": "deployment-v1", "input": inputs.tolist()}


def test_repeated_predictions_match_checkpoint_and_load_once(
    checkpoint, predict_request, monkeypatch
):
    calls = []

    def load(path):
        calls.append(path)
        return load_checkpoint(path)

    monkeypatch.setattr("ml.inference.registry.load_checkpoint", load)
    registry = CheckpointRegistry({"deployment-v1": checkpoint, "other-alias": checkpoint})
    predictor, _ = load_checkpoint(checkpoint)
    with torch.inference_mode():
        expected = predictor(torch.tensor([predict_request["input"]]))[0].tolist()
    for alias in ("deployment-v1", "other-alias", "deployment-v1"):
        response = respond(json.dumps({**predict_request, "model": alias}), registry)
        assert response == {"id": 0, "type": "prediction", "prediction": expected}
    assert calls == [checkpoint.resolve()]


def test_distinct_architectures_coexist_in_one_registry(checkpoint, exports, tmp_path):
    temporal_path = tmp_path / "temporal.pt"
    train(exports / "cnn", temporal_path, TrainingConfig(model="cnn", epochs=1))
    deployments = [
        ("baseline-deployment", checkpoint, "mlp"),
        ("temporal-deployment", temporal_path, "cnn"),
    ]
    registry = CheckpointRegistry({alias: path for alias, path, _layout in deployments})
    for alias, path, layout in deployments:
        dataset = MarketDataset(exports / layout)
        inputs, _ = next(iter(dataset))
        predictor, _ = load_checkpoint(path)
        with torch.inference_mode():
            expected = predictor(inputs.unsqueeze(0))[0].tolist()
        assert registry.describe_models()[alias]["inputShape"] == dataset.metadata["inputShape"]
        for _ in range(3):
            assert registry.predict(alias, inputs.tolist()) == expected


@pytest.mark.parametrize(
    "line",
    [
        "",
        "{",
        "[]",
        "null",
        "true",
        "42",
        '{"id":0,"id":1}',
        '{"id":0,"input":[NaN]}',
        "[" * 2000,
    ],
)
def test_malformed_json_does_not_terminate_session(registry, predict_request, line):
    response = respond(line, registry)
    assert response["type"] == "error" and response["id"] is None
    assert respond(json.dumps(predict_request), registry)["type"] == "prediction"


@pytest.mark.parametrize(
    "changes",
    [
        {"id": None},
        {"id": True},
        {"id": -1},
        {"id": 1.0},
        {"id": 2**53},
        {"id": "1"},
        {"type": "train"},
        {"model": "unknown"},
        {"model": []},
        {"model": ""},
        {"input": []},
        {"input": None},
        {"input": {}},
        {"input": [[0]]},
        {"privatePortfolio": {}},
    ],
)
def test_invalid_request_is_correlated_when_id_is_valid(registry, predict_request, changes):
    response = respond(json.dumps({**predict_request, **changes}), registry)
    assert response["type"] == "error"
    assert response["id"] == (None if "id" in changes else predict_request["id"])


@pytest.mark.parametrize("value", [True, "0", None, [], {}, 1e100, 10**100, float("inf")])
def test_invalid_input_numbers_are_rejected(registry, predict_request, value):
    predict_request["input"][0] = value
    assert respond(json.dumps(predict_request), registry)["type"] == "error"


def test_missing_fields_and_largest_valid_id(registry, predict_request):
    for field in predict_request:
        response = respond(
            json.dumps({k: v for k, v in predict_request.items() if k != field}), registry
        )
        assert response["type"] == "error"
    predict_request["id"] = 2**53 - 1
    assert respond(json.dumps(predict_request), registry)["id"] == predict_request["id"]


def test_stream_resynchronizes_after_oversize_invalid_utf8_and_blank_lines(
    registry, predict_request
):
    valid = json.dumps(predict_request).encode()
    source = io.BytesIO(b"x" * (MAX_REQUEST_BYTES + 12) + b"\n\xff\n\n" + valid)
    destination = io.StringIO()
    serve(registry, source, destination, io.StringIO())
    responses = [json.loads(line) for line in destination.getvalue().splitlines()]
    assert [r["type"] for r in responses] == ["error", "error", "error", "prediction"]
    assert "1 MiB" in responses[0]["error"]
    assert "UTF-8" in responses[1]["error"]


def test_third_architecture_uses_checkpoint_shape_eval_mode_and_preprocessing(exports, tmp_path):
    observations = []

    class AdditionalArchitecture(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.layers = nn.Sequential(nn.Dropout(0.8), nn.Linear(config["width"], 3))

        def forward(self, inputs):
            observations.append(
                (self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled())
            )
            return self.layers(inputs)

    register_model(
        "inference-third",
        ModelDefinition(
            version=1,
            configure=lambda metadata, options: {"width": metadata["inputShape"][0]},
            create=AdditionalArchitecture,
        ),
    )
    path = tmp_path / "third.pt"
    train(exports / "cnn", path, TrainingConfig(model="inference-third", epochs=1))
    registry = CheckpointRegistry({"unrelated-deployment": path})
    inputs, _ = next(iter(MarketDataset(exports / "cnn")))
    loaded, payload = load_checkpoint(path)
    assert payload["model_type"] != payload["metadata"]["features"]["kind"]
    with torch.inference_mode():
        expected = loaded(inputs.unsqueeze(0))[0].tolist()
    observations.clear()
    for _ in range(3):
        assert registry.predict("unrelated-deployment", inputs.tolist()) == expected
    assert observations == [(False, False, True)] * 3
    assert all(p.grad is None for p in loaded.parameters())


def test_prediction_failures_and_model_logging_do_not_corrupt_protocol(
    registry, predict_request, monkeypatch
):
    predictor, _ = registry._models["deployment-v1"]
    original = predictor.forward

    def failure(inputs):
        print("architecture diagnostic")
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(predictor, "forward", failure)
    diagnostics = io.StringIO()
    response = respond(json.dumps(predict_request), registry, diagnostics)
    assert response["type"] == "error" and response["id"] == 0
    assert "backend unavailable" in diagnostics.getvalue()
    assert "architecture diagnostic" in diagnostics.getvalue()
    for output in (
        torch.zeros(3),
        torch.full((1, 3), float("nan")),
        torch.ones((1, 3), dtype=torch.complex64),
        torch.ones((1, 3), dtype=torch.int64),
        [1, 2, 3],
    ):
        monkeypatch.setattr(predictor, "forward", lambda inputs, output=output: output)
        assert respond(json.dumps(predict_request), registry)["type"] == "error"
    monkeypatch.setattr(predictor, "forward", original)
    assert respond(json.dumps(predict_request), registry)["type"] == "prediction"


def test_registry_paths_are_relative_to_file_and_duplicate_aliases_fail(tmp_path, checkpoint):
    config = tmp_path / "registry.json"
    config.write_text(json.dumps({"release": os.path.relpath(checkpoint, tmp_path)}))
    assert read_registry(config)["release"].resolve() == checkpoint
    for contents in ("{}", "[]", '{"release":42}', '{"release":"a","release":"b"}'):
        config.write_text(contents)
        with pytest.raises(ValueError):
            read_registry(config)


def read_response(process):
    # A response must arrive while stdin remains OPEN: this catches missing flushes/EOF buffering.
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=20), "Server did not flush a response within 20 seconds"
        line = process.stdout.readline()
    assert line.endswith(b"\n"), f"Server stopped without a complete response: {line!r}"
    return json.loads(line)


def test_real_process_stays_alive_flushes_and_recovers(checkpoint, predict_request, tmp_path):
    config = tmp_path / "registry.json"
    config.write_text(json.dumps({"deployment-v1": str(checkpoint)}))
    with (tmp_path / "stderr.log").open("w+") as diagnostics:
        process = subprocess.Popen(
            [sys.executable, "-m", "ml.inference.server", "--registry", str(config)],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=diagnostics,
        )
        try:
            responses = []
            for line in [json.dumps(predict_request), "broken-json", json.dumps(predict_request)]:
                process.stdin.write((line + "\n").encode())
                process.stdin.flush()
                responses.append(read_response(process))
                assert process.poll() is None
            assert responses[0] == responses[2]
            assert responses[1]["type"] == "error"
            predictor, _ = load_checkpoint(checkpoint)
            with torch.inference_mode():
                assert (
                    responses[0]["prediction"]
                    == predictor(torch.tensor([predict_request["input"]]))[0].tolist()
                )
            process.stdin.close()
            assert process.wait(timeout=20) == 0
            assert process.stdout.read() == b""
            diagnostics.seek(0)
            assert "Inference ready" in diagnostics.read()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--model", "no-equals"],
        ["--model", "a=/missing/checkpoint.pt"],
        ["--model", "a=x", "--model", "a=y"],
        ["--num-threads", "0"],
    ],
)
def test_startup_errors_are_nonzero_and_stderr_only(arguments):
    result = subprocess.run(
        [sys.executable, "-m", "ml.inference.server", *arguments],
        cwd=ROOT,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode != 0 and result.stdout == b""
    assert b"Inference startup failed" in result.stderr


def test_corrupt_and_incompatible_checkpoints_fail_startup(checkpoint, tmp_path):
    _, payload = load_checkpoint(checkpoint)
    incompatible = copy.deepcopy(payload)
    incompatible["checkpoint_version"] = 999
    broken = tmp_path / "broken.pt"
    for contents in (None, incompatible):
        if contents is None:
            broken.write_bytes(b"not a checkpoint")
        else:
            broken.unlink()
            save_checkpoint(broken, contents)
        result = subprocess.run(
            [sys.executable, "-m", "ml.inference.server", "--model", f"release={broken}"],
            cwd=ROOT,
            capture_output=True,
            timeout=20,
        )
        assert result.returncode == 1 and result.stdout == b""
        assert b"Inference startup failed" in result.stderr
