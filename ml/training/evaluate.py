"""Evaluation always uses inference mode and sample-weighted metrics."""

import torch
from torch import nn

from ml.data.schema import require


def validate_logits(logits, targets, num_classes: int):
    require(logits.shape == (len(targets), num_classes), "Model output shape differs from classes")
    require(torch.isfinite(logits).all(), "Model produced non-finite logits")


@torch.no_grad()
def evaluate(model: nn.Module, batches, num_classes: int, device="cpu") -> dict:
    """Weight metrics by sample count so a short final batch is not over-represented."""
    model.eval()
    loss_sum = 0.0
    count = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        validate_logits(logits, targets, num_classes)
        loss = criterion(logits, targets)
        require(torch.isfinite(loss), "Non-finite evaluation loss")
        loss_sum += loss.item()
        predicted = logits.argmax(dim=1)
        indices = (targets * num_classes + predicted).cpu()
        confusion += torch.bincount(indices, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
        count += len(targets)
    require(count > 0, "Cannot evaluate an empty dataset")
    per_class = []
    for index in range(num_classes):
        correct = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = correct / predicted_count if predicted_count else 0.0
        recall = correct / support if support else 0.0
        per_class.append(
            {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    return {
        "loss": loss_sum / count,
        "accuracy": int(confusion.diag().sum()) / count,
        "sample_count": count,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }