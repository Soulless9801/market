"""Run with python -m ml.training.train; imports never trigger training."""

import argparse
import copy
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.data.dataset import MarketDataset, ShuffleBuffer, split_trajectories
from ml.data.preprocessing import Predictor, Standardizer
from ml.data.schema import parse_json, require, trajectory_seed
from ml.models import build_model, configure_model, definition_for, model_names
from ml.training.checkpoint import CHECKPOINT_VERSION, save_checkpoint
from ml.training.config import OPTIMIZERS, SCHEDULERS, TrainingConfig
from ml.training.evaluate import evaluate, validate_logits
from ml.utils.reproducibility import seed_everything, seed_worker, seeded_generator


def make_loader(dataset, config: TrainingConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=seeded_generator(config.seed),
        multiprocessing_context="spawn" if config.num_workers else None,
    )


def create_optimizer(parameters, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=config.weight_decay,
        )
    return torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """Epoch-level schedules; early stopping may end training before a cosine cycle completes."""
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=max(1, config.patience // 2)
        )
    return None


def validation_data(dataset: MarketDataset, config: TrainingConfig):
    """Use whole held-out trajectories, or a compatible export with disjoint seeds."""
    if config.validation_dataset is None:
        return split_trajectories(dataset, config.validation_split, config.seed)
    validation = MarketDataset(config.validation_dataset)
    for key in (
        "schemaVersion",
        "inputShape",
        "classNames",
        "targetType",
        "targetEncoding",
        "numClasses",
        "features",
        "label",
    ):
        require(
            dataset.metadata[key] == validation.metadata[key],
            f"Validation dataset has incompatible {key}",
        )
    train_seeds = {trajectory_seed(dataset.metadata, t) for t in dataset.trajectories}
    validation_seeds = {trajectory_seed(validation.metadata, t) for t in validation.trajectories}
    require(
        train_seeds.isdisjoint(validation_seeds),
        "Training and validation share simulation seeds; use independent trajectories",
    )
    return dataset, validation


def train(
    dataset_directory: str | Path, output: str | Path, config: TrainingConfig
) -> tuple[Predictor, dict]:
    """Train, select by validation loss, restore the best state, and publish a new checkpoint."""
    config.validate()
    definition = definition_for(config.model)
    require(not Path(output).exists(), "Checkpoint already exists; choose a new output path")
    if config.device == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    seed_everything(config.seed)
    torch.set_num_threads(config.num_threads)
    dataset = MarketDataset(dataset_directory)
    training, validation = validation_data(dataset, config)
    train_loader, validation_loader = make_loader(training, config), make_loader(validation, config)
    # Fitting and evaluation keep file order so their float sums stay deterministic;
    # only optimizer steps draw from the optional shuffle buffer.
    shuffled = (
        ShuffleBuffer(training, config.shuffle_buffer, config.seed)
        if config.shuffle_buffer
        else None
    )
    step_loader = make_loader(shuffled, config) if shuffled else train_loader
    model_config = configure_model(config.model, dataset.metadata, config.model_options)
    backbone = build_model(config.model, definition.version, model_config)
    normalizer = Standardizer(dataset.metadata["inputShape"][0])
    # Validation statistics must not influence the transform learned from training.
    normalizer.fit(train_loader)
    predictor = Predictor(normalizer, backbone).to(config.device)
    optimizer = create_optimizer(predictor.parameters(), config)
    scheduler = create_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    num_classes = dataset.metadata["numClasses"]
    history = []
    best_loss = float("inf")
    stopping_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        predictor.train()
        if shuffled:
            shuffled.set_epoch(epoch)
        for inputs, targets in step_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = predictor(inputs)
            validate_logits(logits, targets, num_classes)
            loss = criterion(logits, targets)
            require(torch.isfinite(loss), "Non-finite training loss")
            loss.backward()
            require(
                all(p.grad is None or torch.isfinite(p.grad).all() for p in predictor.parameters()),
                "Non-finite training gradient",
            )
            optimizer.step()
        # Report a fresh evaluation of the same epoch's model on both partitions.
        train_metrics = evaluate(predictor, train_loader, num_classes, config.device)
        val_metrics = evaluate(predictor, validation_loader, num_classes, config.device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["loss"])
        elif scheduler:
            scheduler.step()

        # format number of digits in epoch to max digits of total epochs
        epoch_digits = len(str(config.epochs))

        # print inline
        print(
            f"epoch {epoch:0{epoch_digits}d}: "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        # early stopping based on validation loss
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
        if val_metrics["loss"] < stopping_loss - config.min_delta:
            stopping_loss = val_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if config.patience and stale_epochs >= config.patience:
            break
    require(best_state is not None, "Training did not produce a checkpoint")
    # Discard the last epoch if an earlier one generalized better. Inference must
    # restore these selected weights rather than whichever epoch happened to run last.
    predictor.load_state_dict(best_state)
    predictor.cpu().eval()
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_type": config.model,
        "model_version": definition.version,
        "model_config": model_config,
        "state_dict": best_state,
        "training_config": asdict(config),
        "metadata": copy.deepcopy(dataset.metadata),
        "validation_metadata": copy.deepcopy(validation.metadata),
        "split": {
            "method": "independent-dataset" if config.validation_dataset else "trajectory",
            "train_trajectories": sorted(training.trajectories),
            "validation_trajectories": sorted(validation.trajectories),
            "training_samples": len(training),
            "validation_samples": len(validation),
        },
        "best_epoch": best_epoch,
        "history": history,
        "metrics": history[best_epoch - 1],
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "device": config.device,
        },
        "output_type": "logits",
        "preprocessing_version": 1,
    }
    save_checkpoint(output, payload)
    return predictor, payload


def main(argv=None) -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description="train a market model"
    )
    parser.add_argument(
        "--dataset", required=True, help="directory containing metadata.json and dataset.jsonl"
    )
    parser.add_argument(
        "--output", required=True, help="path to write a new checkpoint"
    )
    parser.add_argument("--model", choices=model_names(), default=defaults.model)
    parser.add_argument(
        "--model-options", default="{}", help="json object of architecture-specific options"
    )
    for flag in (
        "seed", "batch_size", "epochs", "patience", "num_workers", "num_threads", "shuffle_buffer"
    ):
        parser.add_argument(
            "--" + flag.replace("_", "-"), type=int, default=getattr(defaults, flag)
        )
    for flag in ("learning_rate", "weight_decay", "min_delta", "label_smoothing"):
        parser.add_argument(
            "--" + flag.replace("_", "-"), type=float, default=getattr(defaults, flag)
        )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validation-split", type=float, default=defaults.validation_split)
    group.add_argument("--validation-dataset")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=defaults.device)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default=defaults.optimizer)
    parser.add_argument("--scheduler", choices=SCHEDULERS, default=defaults.scheduler)
    options = vars(parser.parse_args(argv))
    dataset, output = options.pop("dataset"), options.pop("output")
    try:
        options["model_options"] = parse_json(options["model_options"])
        _, payload = train(dataset, output, TrainingConfig(**options))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Training Failed: {error}\n")

    print("\nTRAINING COMPLETE")

    # print inline
    validation = payload["metrics"]["validation"]
    print(
        f"\nCheckpoint: {Path(output).resolve()}\n"
        f"Best Epoch: {payload['best_epoch']}\n"
        f"Validation Loss: {validation['loss']:.4f}\n"
        f"Validation Accuracy: {validation['accuracy']:.4f}"
    )

    # print confusion matrix inline
    print(f"\nCONFUSION MATRIX\n\n{np.array(validation['confusion_matrix'])}")
    # print per class metrics inline
    print("\nPER CLASS METRICS\n")
    for i, class_metrics in enumerate(validation["per_class"]):
        print(
            f"class {i}: support={class_metrics['support']} | "
            f"precision={class_metrics['precision']:.4f} | "
            f"recall={class_metrics['recall']:.4f} | "
            f"f1={class_metrics['f1']:.4f}"
        )

if __name__ == "__main__":
    main()
