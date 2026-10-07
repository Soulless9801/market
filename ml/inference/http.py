"""HTTP transport for the existing checkpoint registry; no architecture-specific inference."""

import json
import logging
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

import torch

from ml.data.schema import parse_json, require
from ml.inference.registry import CheckpointRegistry, read_registry, validate_alias

MAX_BODY_BYTES = 1_048_576


class HostedInference:
    """Lazy, thread-safe loading once per warm function instance (not once per request)."""

    def __init__(
        self, checkpoint: Path | str | None = None, alias: str | None = None,
        *, registry_path: Path | str | None = None,
    ):
        require((checkpoint is None) != (registry_path is None), "Select a checkpoint or registry")
        self.alias = validate_alias(alias) if alias is not None else None
        require(checkpoint is None or self.alias is not None, "A single checkpoint needs an alias")
        self.checkpoint = checkpoint
        self.registry_path = registry_path
        self._registry = None
        self._lock = Lock()

    def registry(self) -> CheckpointRegistry:
        with self._lock:
            if self._registry is None:
                torch.set_num_threads(1)
                checkpoints = (
                    read_registry(self.registry_path) if self.registry_path is not None
                    else {self.alias: self.checkpoint}
                )
                default_alias = self.alias or sorted(checkpoints)[0]
                require(default_alias in checkpoints, "Default model is not in the catalog")
                self._registry = CheckpointRegistry(checkpoints)
                self.alias = default_alias
            return self._registry


class InferenceHandler(BaseHTTPRequestHandler):
    """Same JSON metadata/prediction API as the local Node bridge, served on Vercel."""

    # Each hosting entry point supplies its deployment selection; the adapter is generic.
    backend: HostedInference

    def _send_json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _operation(self) -> str | None:
        url = urlsplit(self.path)
        # Support both Vercel's rewritten query and the original browser route.
        if url.path == "/api/inference":
            operations = parse_qs(url.query).get("operation", [])
            return operations[0] if len(operations) == 1 else None
        prefix = "/__market/inference/"
        return url.path.removeprefix(prefix) if url.path.startswith(prefix) else None

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        try:
            allowed = urlsplit(origin).netloc == self.headers.get("Host")
        except ValueError:
            allowed = False
        if not allowed:
            self._send_json(403, {"error": "Cross-origin inference is not allowed"})
        return allowed

    def do_GET(self) -> None:
        if not self._same_origin():
            return
        operation = self._operation()
        if operation not in ("model", "models"):
            self._send_json(404, {"error": "Unknown inference endpoint"})
            return
        try:
            models = self.backend.registry().describe_models()
        except Exception:
            logging.exception("Could not load inference checkpoint")
            self._send_json(503, {"error": "Model unavailable; check inference server logs"})
            return
        if operation == "models":
            self._send_json(200, {
                "defaultAlias": self.backend.alias,
                "models": [
                    {"alias": alias, "metadata": models[alias]} for alias in sorted(models)
                ],
            })
            return
        aliases = parse_qs(urlsplit(self.path).query, keep_blank_values=True).get("alias", [])
        alias = aliases[0] if aliases else self.backend.alias
        if len(aliases) > 1 or alias not in models:
            self._send_json(400, {"error": "Unknown model alias"})
            return
        self._send_json(200, {"alias": alias, "metadata": models[alias]})

    def do_POST(self) -> None:
        if not self._same_origin():
            return
        if self._operation() != "predict":
            self._send_json(404, {"error": "Unknown inference endpoint"})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "Expected application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 1:
            self._send_json(400, {"error": "A nonempty JSON body is required"})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "Inference request is too large"})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body")
            body = parse_json(raw.decode("utf-8"))
            if (not isinstance(body, dict) or "input" not in body
                    or not set(body) <= {"input", "model"}):
                raise ValueError("Expected an input feature vector")
        except (ValueError, UnicodeError, RecursionError) as error:
            self._send_json(400, {"error": str(error)})
            return
        try:
            registry = self.backend.registry()
        except Exception:
            logging.exception("Could not load inference checkpoint")
            self._send_json(503, {"error": "Model unavailable; check inference server logs"})
            return
        try:
            alias = validate_alias(body.get("model", self.backend.alias))
            prediction = registry.predict(alias, body["input"])
            self._send_json(200, {"prediction": prediction})
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
        except Exception:
            logging.exception("Model prediction failed")
            self._send_json(503, {"error": "Prediction failed; check inference server logs"})
