import copy
import random
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from ml.data.dataset import MarketDataset, ShuffleBuffer
from ml.models import ModelDefinition, build_model, configure_model, register_model
from ml.training.checkpoint import load_checkpoint, save_checkpoint
from ml.training.config import TrainingConfig
from ml.training.train import train
from ml.utils.reproducibility import seed_everything


def test_explicit_seeding_covers_python_numpy_and_torch():
    def sample(seed):
        seed_everything(seed)
        return random.random(), np.random.random(), torch.rand(3)

    first, second = sample(42), sample(42)
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert not torch.equal(first[2], sample(43)[2])


@pytest.mark.parametrize("layout", ["mlp", "cnn"])
def test_mlp_uses_dataset_dimensions_and_small_architecture(exports, layout):
    metadata = MarketDataset(exports / layout).metadata
    config = configure_model("mlp", metadata, {})
    seed_everything(3)
    model = build_model("mlp", 1, config)
    seed_everything(3)
    duplicate = build_model("mlp", 1, config)
    width = metadata["inputShape"][0]
    values = torch.ones(4, width)
    assert model(values).shape == (4, metadata["numClasses"])
    assert torch.equal(model(values), duplicate(values))
    assert sum(p.numel() for p in model.parameters()) < 2100
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(2, width + 1))
    with pytest.raises(ValueError, match="Unknown MLP options"):
        configure_model("mlp", metadata, {"input_shape": [1]})


@pytest.mark.parametrize(("architecture", "layout", "options"), [
    ("mlp", "mlp", {"hidden_sizes": [8, 4]}),
    ("cnn", "cnn", {"channels": [4, 8], "kernel_size": [5, 3], "pool_size": 3, "dropout": 0.3}),
    ("mlp", "orderbook", {"hidden_sizes": [8, 4]}),
    ("cnn", "orderbook", {"channels": [4], "kernel_size": [3], "pool_size": 8, "dropout": 0.1}),
])
def test_training_is_repeatable_and_checkpoint_predictions_match(
    exports, tmp_path, architecture, layout, options
):
    config = TrainingConfig(model=architecture, epochs=3, batch_size=7, model_options=options)
    first, payload = train(exports / layout, tmp_path / "first.pt", config)
    second, repeated = train(exports / layout, tmp_path / "second.pt", config)
    reloaded, saved = load_checkpoint(tmp_path / "first.pt")
    values = torch.stack([x for x, _ in MarketDataset(exports / layout)])
    assert saved["model_type"] == architecture
    assert all(saved["model_config"][key] == value for key, value in options.items())
    assert payload["history"] == repeated["history"]
    assert (
        payload["best_epoch"]
        == min(payload["history"], key=lambda e: e["validation"]["loss"])["epoch"]
    )
    assert not first.training and not reloaded.training
    with torch.no_grad():
        assert torch.equal(first(values), reloaded(values))
        assert torch.equal(first(values), second(values))
    for name, value in first.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name])
    assert saved["metadata"]["classNames"] == ["BUY", "SELL", "HOLD"]
    assert saved["training_config"] == payload["training_config"]
    assert saved["metrics"]["validation"]["sample_count"] == saved["split"]["validation_samples"]
    original = (tmp_path / "first.pt").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        train(exports / layout, tmp_path / "first.pt", config)
    assert (tmp_path / "first.pt").read_bytes() == original
    broken = copy.deepcopy(payload)
    broken["checkpoint_version"] = 999
    save_checkpoint(tmp_path / "invalid.pt", broken)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(tmp_path / "invalid.pt")
    broken = copy.deepcopy(payload)
    broken["state_dict"]["normalizer.scale"][0] = 0
    save_checkpoint(tmp_path / "scale.pt", broken)
    with pytest.raises(ValueError, match="scales"):
        load_checkpoint(tmp_path / "scale.pt")


def test_future_model_registration_uses_same_training_and_checkpoint_path(exports, tmp_path):
    def configure(metadata, options):
        assert options == {}
        return {"width": metadata["inputShape"][0], "classes": metadata["numClasses"]}

    definition = ModelDefinition(
        version=2, configure=configure, create=lambda c: nn.Linear(c["width"], c["classes"])
    )
    register_model("test-linear", definition)
    config = TrainingConfig(model="test-linear", epochs=1)
    model, payload = train(exports / "cnn", tmp_path / "linear.pt", config)
    loaded, _ = load_checkpoint(tmp_path / "linear.pt")
    values = torch.ones(2, 30)
    assert torch.equal(model(values), loaded(values))
    assert payload["model_type"] == "test-linear"
    assert payload["metadata"]["features"]["kind"] == "cnn"
    with pytest.raises(ValueError, match="already registered"):
        register_model("test-linear", definition)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown", 1, {})
    with pytest.raises(ValueError, match="version"):
        build_model("test-linear", 3, payload["model_config"])


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_early_stopping_and_best_weights_restored(exports, tmp_path, monkeypatch, architecture):
    import ml.training.train as training

    original = training.evaluate
    calls = 0
    best = {}

    def controlled(model, batches, classes, device):
        nonlocal calls
        metrics = original(model, batches, classes, device)
        calls += 1
        if calls % 2 == 0:
            metrics["loss"] = float(calls)
            if calls == 2:
                best.update({k: v.clone() for k, v in model.state_dict().items()})
        return metrics

    monkeypatch.setattr(training, "evaluate", controlled)
    model, payload = train(
        exports / "mlp", tmp_path / "stopped.pt",
        TrainingConfig(model=architecture, epochs=10, patience=2),
    )
    assert len(payload["history"]) == 3
    assert payload["best_epoch"] == 1
    assert all(torch.equal(v, best[k]) for k, v in model.state_dict().items())


def test_checkpoint_failure_cleans_partial_files(tmp_path, monkeypatch):
    def fail(_payload, stream):
        stream.write(b"partial")
        raise OSError("write failed")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="write failed"):
        save_checkpoint(tmp_path / "failed.pt", {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"seed": -1},
        {"batch_size": True},
        {"validation_split": 1},
        {"learning_rate": float("nan")},
        {"patience": -1},
        {"weight_decay": -1},
        {"model_options": []},
        {"device": "mps"},
        {"optimizer": "adam"},
        {"scheduler": "step"},
        {"label_smoothing": 1},
        {"label_smoothing": -0.1},
        {"shuffle_buffer": -1},
    ],
)
def test_invalid_training_config_fails(overrides):
    with pytest.raises(ValueError):
        replace(TrainingConfig(), **overrides).validate()


def test_cli_import_safe_and_trains_real_export(exports, tmp_path):
    imported = subprocess.run(
        [sys.executable, "-c", "import ml.training.train"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert imported.stdout == ""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.training.train",
            "--dataset",
            str(exports / "cnn"),
            "--output",
            str(tmp_path / "cli.pt"),
            "--epochs",
            "1",
            "--batch-size",
            "8",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Best Epoch: 1\n" in result.stdout
    assert (tmp_path / "cli.pt").exists()


def test_independent_validation_dataset_trains_and_records_provenance(exports, tmp_path):
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    validation = tmp_path / "validation"
    subprocess.run(
        [
            "node",
            "--import",
            "tsx",
            "scripts/dataset.ts",
            "mlp",
            "--seed",
            "700",
            "--samples",
            "8",
            "--trajectories",
            "2",
            "--output",
            str(validation),
        ],
        cwd=root,
        env={**os.environ, "TSX_TSCONFIG_PATH": "tsconfig.scripts.json"},
        check=True,
        capture_output=True,
    )
    _, checkpoint = train(
        exports / "mlp",
        tmp_path / "independent.pt",
        TrainingConfig(epochs=1, validation_dataset=str(validation)),
    )
    assert checkpoint["split"]["method"] == "independent-dataset"
    assert checkpoint["split"]["training_samples"] == 32
    assert checkpoint["split"]["validation_samples"] == 8
    assert checkpoint["validation_metadata"]["generation"]["seed"] == 700


def test_atomic_checkpoint_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "existing.pt"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        save_checkpoint(path, {"example": 1})
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_shuffle_buffer_permutes_rows_reproducibly_per_epoch():
    rows = list(range(50))
    shuffled = ShuffleBuffer(rows, buffer_size=8, seed=7)
    first = list(shuffled)
    assert sorted(first) == rows and first != rows
    assert list(shuffled) == first
    shuffled.set_epoch(1)
    assert sorted(list(shuffled)) == rows and list(shuffled) != first
    with pytest.raises(ValueError):
        ShuffleBuffer(rows, buffer_size=0, seed=7)


@pytest.mark.parametrize("scheduler", ["cosine", "plateau"])
def test_optimization_options_train_repeatably_and_are_recorded(exports, tmp_path, scheduler):
    config = TrainingConfig(
        epochs=3,
        batch_size=7,
        optimizer="sgd",
        scheduler=scheduler,
        label_smoothing=0.1,
        shuffle_buffer=16,
    )
    _, payload = train(exports / "mlp", tmp_path / "first.pt", config)
    _, repeated = train(exports / "mlp", tmp_path / "second.pt", config)
    default = TrainingConfig(epochs=3, batch_size=7)
    _, baseline = train(exports / "mlp", tmp_path / "baseline.pt", default)
    assert payload["history"] == repeated["history"]
    assert payload["history"] != baseline["history"]
    _, saved = load_checkpoint(tmp_path / "first.pt")
    assert saved["training_config"]["optimizer"] == "sgd"
    assert saved["training_config"]["scheduler"] == scheduler
    assert saved["training_config"]["shuffle_buffer"] == 16
