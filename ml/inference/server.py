"""JSONL stdin/stdout server. Run with python -m ml.inference.server --help."""

import argparse
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import BinaryIO, TextIO

import torch

from ml.data.schema import require

from .protocol import error_response, respond
from .registry import CheckpointRegistry, read_registry, validate_alias

MAX_REQUEST_BYTES = 1024 * 1024


def serve(
    registry: CheckpointRegistry,
    source: BinaryIO,
    destination: TextIO,
    diagnostics: TextIO,
) -> None:
    """Bound each UTF-8 line to 1 MiB (including newline); drain oversize lines before resuming."""
    while line := source.readline(MAX_REQUEST_BYTES + 1):
        if len(line) > MAX_REQUEST_BYTES:
            # Drain the rest of the same bad line. Treating chunks as new requests
            # would lose framing and associate errors with unrelated predictions.
            while not line.endswith(b"\n"):
                line = source.readline(MAX_REQUEST_BYTES + 1)
                if not line:
                    break
            response = error_response(None, "Request exceeds 1 MiB limit")
        else:
            try:
                response = respond(line.decode("utf-8"), registry, diagnostics)
            except UnicodeDecodeError:
                response = error_response(None, "Request must be UTF-8")
        destination.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
        destination.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="JSON mapping of aliases to checkpoint paths")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="Register a checkpoint (repeatable); paths are relative to the working directory",
    )
    parser.add_argument("--num-threads", type=int, default=1, help="Positive CPU thread count")
    parser.add_argument(
        "--ready-message",
        action="store_true",
        help="Emit a versioned JSON ready message after all checkpoints load (for process clients)",
    )
    args = parser.parse_args(argv)
    try:
        require(args.num_threads > 0, "num-threads must be positive")
        checkpoints = read_registry(args.registry) if args.registry else {}
        for entry in args.model:
            alias, separator, path = entry.partition("=")
            require(separator and bool(path.strip()), "Expected --model ALIAS=PATH")
            validate_alias(alias)
            require(alias not in checkpoints, f"Duplicate model alias: {alias}")
            checkpoints[alias] = Path(path)
        # A bad checkpoint fails startup; never advertise a partially ready server.
        with redirect_stdout(sys.stderr):
            torch.set_num_threads(args.num_threads)
            registry = CheckpointRegistry(checkpoints)
        print(
            f"Inference ready: {', '.join(checkpoints)} (logits, CPU)", file=sys.stderr, flush=True
        )
    except Exception:
        print("Inference startup failed", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    try:
        if args.ready_message:
            # Process clients opt into readiness plus public feature contracts.
            # Plain Phase 4 callers still receive exactly one reply per request.
            print(
                json.dumps(
                    {"type": "ready", "protocolVersion": 1, "models": registry.describe_models()}
                ),
                flush=True,
            )
        serve(registry, sys.stdin.buffer, sys.stdout, sys.stderr)
    except BrokenPipeError:
        # Prevent another failing flush during interpreter shutdown when the client closes stdout.
        sys.stdout = sys.stderr
        return 0
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
