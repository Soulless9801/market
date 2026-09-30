import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from ml.data.dataset import MarketDataset, split_trajectories
from ml.data.preprocessing import Standardizer
from ml.data.schema import parse_json
from ml.training.config import TrainingConfig
from ml.training.train import make_loader, validation_data


def copied_export(exports, tmp_path):
    output = tmp_path / "export"
    shutil.copytree(exports / "mlp", output)
    return output


def rewrite(directory, rows=None, metadata=None):
    if metadata is None:
        metadata = json.loads((directory / "metadata.json").read_text())
    if rows is not None:
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (directory / "dataset.jsonl").write_bytes(raw)
        metadata["sha256"] = hashlib.sha256(raw).hexdigest()
    (directory / "metadata.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize(("layout", "width"), [("mlp", 40), ("cnn", 30)])
def test_real_typescript_exports_stream_public_tensors(exports, layout, width, monkeypatch):
    read_bytes = Path.read_bytes

    def guarded_read(path):
        assert path.name != "dataset.jsonl", "JSONL must be streamed"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    dataset = MarketDataset(exports / layout)
    batches = list(DataLoader(dataset, batch_size=7))
    assert len(dataset) == 32
    assert sum(len(targets) for _, targets in batches) == 32
    assert batches[0][0].shape == (7, width)
    assert batches[0][0].dtype == torch.float32
    assert batches[0][1].shape == (7,)
    assert batches[0][1].dtype == torch.long
    assert all(torch.isfinite(inputs).all() for inputs, _ in batches)
    assert all(((targets >= 0) & (targets < 3)).all() for _, targets in batches)
    assert torch.equal(next(iter(dataset))[0], batches[0][0][0])


@pytest.mark.parametrize(
    "damage",
    [
        {"target": True},
        {"target": 3},
        {"input": [1, 2]},
        {"simulationSeed": 7},
        {"step": 43},
        {"trajectoryIndex": 1},
        {"portfolio": {"cash": 1}},
        {"input": [False] * 40},
        {"input": [1e39] * 40},
        {"input": [float("nan")] * 40},
    ],
)
def test_invalid_rows_rejected_with_line_number(exports, tmp_path, damage):
    directory = copied_export(exports, tmp_path)
    rows = [json.loads(line) for line in (directory / "dataset.jsonl").read_text().splitlines()]
    rows[0].update(damage)
    rewrite(directory, rows=rows)
    with pytest.raises(ValueError, match="line 1"):
        MarketDataset(directory)


@pytest.mark.parametrize(
    "damage",
    [
        {"schemaVersion": 2},
        {"schemaVersion": True},
        {"inputShape": [20, 2]},
        {"classNames": ["SELL", "BUY", "HOLD"]},
        {"datasetFile": "../outside.jsonl"},
        {"targetEncoding": "one-hot"},
        {"features": {}},
        {"generation": {}},
        {"sampleCount": 0},
    ],
)
def test_invalid_metadata_rejected(exports, tmp_path, damage):
    directory = copied_export(exports, tmp_path)
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata.update(damage)
    rewrite(directory, metadata=metadata)
    with pytest.raises(ValueError):
        MarketDataset(directory)


def test_integrity_counts_order_and_file_changes(exports, tmp_path):
    directory = copied_export(exports, tmp_path)
    original = (directory / "dataset.jsonl").read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    metadata = json.loads((directory / "metadata.json").read_text())
    rows[0]["input"][0] += 1
    (directory / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="SHA-256"):
        MarketDataset(directory)
    rewrite(directory, rows=rows[:-1])
    with pytest.raises(ValueError, match="row count"):
        MarketDataset(directory)
    rows[1] = rows[0]
    rewrite(directory, rows=rows)
    with pytest.raises(ValueError, match="line 2"):
        MarketDataset(directory)
    (directory / "dataset.jsonl").write_bytes(original)
    rewrite(directory, metadata=metadata)
    dataset = MarketDataset(directory)
    (directory / "dataset.jsonl").write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="changed after validation"):
        next(iter(dataset))


def test_split_is_deterministic_disjoint_and_preprocessing_uses_only_training(exports):
    dataset = MarketDataset(exports / "mlp")
    training, validation = split_trajectories(dataset, 0.25, 42)
    again, _ = split_trajectories(dataset, 0.25, 42)
    assert training.trajectories == again.trajectories
    assert training.trajectories.isdisjoint(validation.trajectories)
    assert training.trajectories | validation.trajectories == dataset.trajectories
    assert len(training) + len(validation) == len(dataset)
    normalizer = Standardizer(40)
    normalizer.fit(DataLoader(training, batch_size=5))
    train_values = torch.stack([x for x, _ in training]).double()
    torch.testing.assert_close(normalizer.mean, train_values.mean(dim=0), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        normalizer.scale, train_values.std(dim=0, correction=0).clamp_min(1e-8)
    )
    all_values = torch.stack([x for x, _ in dataset]).double()
    assert not torch.allclose(normalizer.mean, all_values.mean(dim=0))
    assert torch.isfinite(normalizer(train_values.float())).all()
    with pytest.raises(ValueError, match="two trajectories"):
        split_trajectories(dataset.select([0]), 0.2, 42)
    with pytest.raises(ValueError, match="share simulation seeds"):
        validation_data(dataset, TrainingConfig(validation_dataset=str(exports / "mlp")))
    with pytest.raises(ValueError, match="incompatible"):
        validation_data(dataset, TrainingConfig(validation_dataset=str(exports / "cnn")))


def test_worker_shards_cover_every_row_once(exports, monkeypatch):
    dataset = MarketDataset(exports / "mlp")
    expected = [(x.tolist(), y.item()) for x, y in dataset]
    shards = []
    for worker in range(3):
        monkeypatch.setattr(
            "ml.data.dataset.get_worker_info",
            lambda worker=worker: SimpleNamespace(id=worker, num_workers=3),
        )
        shards.append([(x.tolist(), y.item()) for x, y in dataset])
    combined = [shards[i % 3][i // 3] for i in range(len(expected))]
    assert combined == expected


def test_actual_multiworker_loader_no_duplicates(exports):
    dataset = MarketDataset(exports / "mlp")
    loader = make_loader(dataset, TrainingConfig(batch_size=5, num_workers=2))
    actual = [(tuple(x.tolist()), int(y)) for xs, ys in loader for x, y in zip(xs, ys, strict=True)]
    expected = [(tuple(x.tolist()), int(y)) for x, y in dataset]
    assert sorted(actual) == sorted(expected)


def test_duplicate_keys_and_nonfinite_json_are_rejected():
    for text in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
        with pytest.raises(ValueError):
            parse_json(text)
