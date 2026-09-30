"""The existing TypeScript schema-v1 contract, independent of model architecture."""

import json
import math
import re
from pathlib import Path

MAX_SAFE_INTEGER = 2**53 - 1
CLASS_NAMES = ["BUY", "SELL", "HOLD"]


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value, name: str, minimum: int = 0, maximum: int = MAX_SAFE_INTEGER) -> int:
    require(
        type(value) is int and minimum <= value <= maximum,
        f"{name} must be an integer in [{minimum}, {maximum}]",
    )
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def parse_json(text: str | bytes):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def validate_metadata(m: dict) -> dict:
    require(isinstance(m, dict), "Metadata must be an object")
    require(
        set(m)
        == {
            "schemaVersion",
            "datasetType",
            "inputShape",
            "targetType",
            "targetEncoding",
            "numClasses",
            "classNames",
            "features",
            "label",
            "generation",
            "datasetFile",
            "sampleCount",
            "classCounts",
            "sha256",
        },
        "Unexpected or missing metadata fields",
    )
    integer(m["schemaVersion"], "schemaVersion", 1, 1)
    require(m["datasetType"] == "market-side-classification", "Unsupported dataset type")
    require(m["targetType"] == "classification", "Unsupported target type")
    require(m["targetEncoding"] == "class-index", "Unsupported target encoding")
    integer(m["numClasses"], "numClasses", len(CLASS_NAMES), len(CLASS_NAMES))
    require(m["classNames"] == CLASS_NAMES, "Unexpected class order")
    shape = m["inputShape"]
    require(isinstance(shape, list) and len(shape) == 1, "Schema v1 requires a flat inputShape")
    width = integer(shape[0], "inputShape[0]", 1)
    integer(m["sampleCount"], "sampleCount", 1)
    require(m["datasetFile"] == "dataset.jsonl", "datasetFile must be dataset.jsonl")
    require(
        isinstance(m["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", m["sha256"]),
        "Invalid SHA-256",
    )
    counts = m["classCounts"]
    require(isinstance(counts, dict) and set(counts) == set(CLASS_NAMES), "Invalid classCounts")
    for value in counts.values():
        integer(value, "class count")
    require(sum(counts.values()) == m["sampleCount"], "Class counts do not sum to sampleCount")
    f = m["features"]
    require(
        isinstance(f, dict)
        and set(f)
        == {
            "version",
            "kind",
            "names",
            "normalization",
            "historyOrder",
            "priceHistoryLimit",
            "tradeHistoryLimit",
            "bookDepth",
            "padding",
            "description",
        },
        "Invalid feature metadata",
    )
    integer(f["version"], "feature version", 1)
    names = f["names"]
    require(
        isinstance(names, list)
        and len(names) == width
        and all(isinstance(n, str) and n.strip() for n in names)
        and len(set(names)) == width,
        "Feature names must be unique and match inputShape",
    )
    require(f["normalization"] == "none", "Expected raw, unnormalized features")
    for key in ("kind", "historyOrder", "padding", "description"):
        require(isinstance(f[key], str) and f[key].strip(), f"Invalid features.{key}")
    require(re.fullmatch(r"[a-z][a-z0-9_-]*", f["kind"]), "Invalid feature kind")
    for key in ("priceHistoryLimit", "tradeHistoryLimit"):
        integer(f[key], key, 0, 1000)
    integer(f["bookDepth"], "bookDepth", 1)
    g = m["generation"]
    require(
        isinstance(g, dict)
        and set(g)
        == {
            "version",
            "model",
            "seed",
            "warmupSteps",
            "horizonSteps",
            "sampleCount",
            "trajectoryCount",
            "samplesPerTrajectory",
            "actualTrajectoryCount",
            "trajectorySeedStride",
            "seedArithmetic",
            "agentPopulation",
            "referencePrice",
            "observationTiming",
            "historyTiming",
            "rowOrder",
        },
        "Invalid generation metadata",
    )
    integer(g["version"], "generation version", 1, 1)
    integer(g["seed"], "seed", 0, 2**32 - 1)
    integer(g["warmupSteps"], "warmupSteps")
    for key in ("horizonSteps", "sampleCount", "samplesPerTrajectory", "actualTrajectoryCount"):
        integer(g[key], key, 1)
    integer(g["trajectoryCount"], "trajectoryCount", 1, 2**32)
    require(g["model"] == f["kind"], "Generation layout differs from feature kind")
    require(g["sampleCount"] == m["sampleCount"], "Inconsistent sample counts")
    per_trajectory = (g["sampleCount"] + g["trajectoryCount"] - 1) // g["trajectoryCount"]
    actual = (g["sampleCount"] + per_trajectory - 1) // per_trajectory
    require(
        g["samplesPerTrajectory"] == per_trajectory and g["actualTrajectoryCount"] == actual,
        "Inconsistent trajectory partition",
    )
    integer(g["warmupSteps"] + per_trajectory * g["horizonSteps"], "final label step")
    for key, expected in {
        "trajectorySeedStride": 9973,
        "seedArithmetic": "uint32",
        "agentPopulation": "buildDefaultAgents-v1-no-ml-agent",
        "referencePrice": 100,
        "observationTiming": "after-completed-step-before-label-horizon",
        "historyTiming": "one-pre-order-midprice-per-completed-step",
        "rowOrder": "trajectory-index-then-observation-step",
    }.items():
        require(g[key] == expected, f"Unsupported generation.{key}")
    require(
        m["label"]
        == {
            "quantity": "future-midprice-minus-current-midprice",
            "threshold": 0.01,
            "thresholdUnit": "absolute-price",
            "horizonSteps": g["horizonSteps"],
            "buy": "change > threshold",
            "sell": "change < -threshold",
            "hold": "-threshold <= change <= threshold",
        },
        "Unsupported label semantics",
    )
    return m


def read_metadata(directory: Path) -> dict:
    try:
        return validate_metadata(parse_json((directory / "metadata.json").read_bytes()))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid metadata in {directory}: {error}") from error


def trajectory_seed(metadata: dict, trajectory: int) -> int:
    g = metadata["generation"]
    return (g["seed"] + trajectory * g["trajectorySeedStride"]) % 2**32


def validate_row(row, metadata: dict, index: int) -> None:
    require(
        isinstance(row, dict)
        and set(row)
        == {
            "simulationSeed",
            "trajectoryIndex",
            "step",
            "input",
            "target",
        },
        "Row must contain exactly the five public dataset fields",
    )
    g = metadata["generation"]
    require(index < metadata["sampleCount"], "More rows than sampleCount")
    trajectory, sample = divmod(index, g["samplesPerTrajectory"])
    integer(row["trajectoryIndex"], "trajectoryIndex", trajectory, trajectory)
    seed = trajectory_seed(metadata, trajectory)
    integer(row["simulationSeed"], "simulationSeed", seed, seed)
    step = g["warmupSteps"] + sample * g["horizonSteps"]
    integer(row["step"], "step", step, step)
    integer(row["target"], "target", 0, metadata["numClasses"] - 1)
    values = row["input"]
    require(
        isinstance(values, list) and len(values) == metadata["inputShape"][0],
        "Input dimensions differ from inputShape",
    )
    require(
        all(
            type(v) in (int, float) and math.isfinite(v) and abs(v) <= 3.4028234663852886e38
            for v in values
        ),
        "Inputs must be finite float32-representable numbers",
    )
