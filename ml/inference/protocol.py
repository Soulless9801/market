"""One request produces one response; invalid requests never terminate the session."""

import sys
import traceback
from contextlib import redirect_stdout
from typing import TextIO

from ml.data.schema import parse_json, require

from .registry import CheckpointRegistry, validate_alias

MAX_SAFE_ID = 2**53 - 1


def valid_request_id(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_SAFE_ID


def error_response(request_id: int | None, message: str) -> dict:
    return {"id": request_id, "type": "error", "error": message}


def respond(line: str, registry: CheckpointRegistry, diagnostics: TextIO | None = None) -> dict:
    diagnostics = diagnostics if diagnostics is not None else sys.stderr
    request_id = None
    try:
        request = parse_json(line)
        require(isinstance(request, dict), "Request must be an object")
        if valid_request_id(request.get("id")):
            request_id = request["id"]
        require(request_id is not None, "id must be a non-negative JavaScript-safe integer")
        require(request.get("type") == "predict", "Unknown request type; expected predict")
        require(set(request) == {"id", "type", "model", "input"}, "Unexpected request fields")
        alias = validate_alias(request["model"])
    except (ValueError, RecursionError) as error:
        return error_response(request_id, str(error))

    try:
        # Registered architectures may print diagnostics; stdout belongs exclusively to JSONL.
        with redirect_stdout(diagnostics):
            prediction = registry.predict(alias, request["input"])
        return {"id": request_id, "type": "prediction", "prediction": prediction}
    except ValueError as error:
        return error_response(request_id, str(error))
    except Exception:
        traceback.print_exc(file=diagnostics)
        return error_response(request_id, "Prediction failed; see server stderr for details")
