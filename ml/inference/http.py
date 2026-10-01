"""HTTP transport for the existing checkpoint registry; no architecture-specific inference."""

import json
import logging
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

import torch

from ml.data.schema import parse_json
from ml.inference.registry import CheckpointRegistry, validate_alias

MAX_BODY_BYTES = 1_048_576


class HostedInference:
    """Lazy, thread-safe loading once per warm function instance (not once per request)."""

    def __init__(self, checkpoint: Path | str, alias: str):
        self.alias = validate_alias(alias)
        self.checkpoint = checkpoint
        self._registry = None
        self._lock = Lock()

    def registry(self) -> CheckpointRegistry:
        with self._lock:
            if self._registry is None:
                torch.set_num_threads(1)
                self._registry = CheckpointRegistry({self.alias: self.checkpoint})
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
        if self._operation() != "model":
            self._send_json(404, {"error": "Unknown inference endpoint"})
            return
        try:
            metadata = self.backend.registry().describe_models()[self.backend.alias]
            self._send_json(200, {"alias": self.backend.alias, "metadata": metadata})
        except Exception:
            logging.exception("Could not load inference checkpoint")
            self._send_json(503, {"error": "Model unavailable; check inference server logs"})

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
            if not isinstance(body, dict) or set(body) != {"input"}:
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
            prediction = registry.predict(self.backend.alias, body["input"])
            self._send_json(200, {"prediction": prediction})
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
        except Exception:
            logging.exception("Model prediction failed")
            self._send_json(503, {"error": "Prediction failed; check inference server logs"})
