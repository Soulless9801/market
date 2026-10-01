"""Run the deployed Python handler over HTTP, with the same checkpoint as local inference."""

import json
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from api.inference import DEFAULT_CHECKPOINT, handler
from ml.inference.http import HostedInference
from ml.inference.registry import CheckpointRegistry
from ml.training.config import TrainingConfig
from ml.training.train import train


@pytest.fixture(scope="module", params=["deployment", "cnn"])
def hosted_checkpoint(request, exports, tmp_path_factory):
    if request.param == "deployment":
        return DEFAULT_CHECKPOINT
    path = tmp_path_factory.mktemp("hosted-cnn") / "checkpoint.pt"
    train(exports / "cnn", path, TrainingConfig(model="cnn", epochs=1))
    return path


@pytest.fixture
def server(hosted_checkpoint):
    class TestHandler(handler):
        backend = HostedInference(hosted_checkpoint, "third-hosted-deployment")

        def log_message(self, *_args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield http, TestHandler.backend
    http.shutdown()
    http.server_close()
    thread.join(timeout=5)


def request(server, method="GET", operation="model", body=None, headers=None, path=None):
    http, _ = server
    connection = HTTPConnection(*http.server_address, timeout=5)
    try:
        connection.request(
            method,
            path or f"/api/inference?operation={operation}",
            body,
            headers or {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read()), response.getheader("Cache-Control")
    finally:
        connection.close()


def test_deployed_handler_matches_checkpoint_and_reuses_registry(server, hosted_checkpoint):
    status, descriptor, cache = request(server)
    assert status == 200
    assert cache == "no-store"
    assert descriptor["alias"] == "third-hosted-deployment"
    inputs = [0.0] * descriptor["metadata"]["inputShape"][0]
    expected = CheckpointRegistry({"reference": hosted_checkpoint}).predict("reference", inputs)
    registry = server[1].registry()
    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(
            lambda _: request(server, "POST", "predict", json.dumps({"input": inputs})),
            range(3),
        ))
    assert all(response[0] == 200 for response in responses)
    assert all(response[1]["prediction"] == expected for response in responses)
    assert server[1].registry() is registry
    assert request(server, path="/__market/inference/model")[1] == descriptor


@pytest.mark.parametrize("body", ["null", "{", '{}', '{"input":[true]}', '{"input":[NaN]}'])
def test_deployed_handler_rejects_invalid_inputs_without_poisoning_model(server, body):
    assert request(server, "POST", "predict", body)[0] == 400
    assert request(server)[0] == 200


def test_deployed_handler_rejects_oversize_and_cross_origin_requests(server):
    # An oversized upload is rejected from its headers, before reading its body.
    assert request(server, "POST", "predict", headers={
        "Content-Type": "application/json", "Content-Length": "1048577",
    })[0] == 413
    assert request(server, headers={"Origin": "https://another.example"})[0] == 403
    assert request(server, "POST", "predict", "{}", {"Content-Type": "text/plain"})[0] == 415
    assert request(server, operation="missing")[0] == 404


def test_missing_checkpoint_is_a_visible_service_error(server, tmp_path):
    server[1].checkpoint = tmp_path / "missing.pt"
    status, body, _ = request(server)
    assert status == 503
    assert "Model unavailable" in body["error"]
    # Do not leak the deployment's filesystem paths to public callers.
    assert str(tmp_path) not in body["error"]
