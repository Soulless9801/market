"""Train-only feature standardization with streaming, float64 running statistics."""

import torch
from torch import nn

from .schema import require


class Standardizer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(width, dtype=torch.float64))

    @torch.no_grad()
    def fit(self, batches) -> None:
        """Fit only the training partition supplied by train(); never fit at inference time."""
        count = 0
        mean = torch.zeros_like(self.mean)
        squared_deviations = torch.zeros_like(self.mean)
        for inputs, _ in batches:
            values = inputs.to(dtype=torch.float64, device=mean.device)
            batch_count = len(values)
            batch_mean = values.mean(dim=0)
            batch_deviations = ((values - batch_mean) ** 2).sum(dim=0)
            total = count + batch_count
            delta = batch_mean - mean
            # Merge each batch's statistics with Welford's stable parallel update.
            # float64 avoids loss of precision when features have very different scales.
            squared_deviations += batch_deviations + delta.square() * count * batch_count / total
            mean += delta * batch_count / total
            count = total
        require(count > 0, "Cannot fit a normalizer on empty training data")
        scale = (squared_deviations / count).clamp_min(0).sqrt().clamp_min(1e-8)
        require(torch.isfinite(mean).all() and torch.isfinite(scale).all(), "Non-finite statistics")
        self.mean.copy_(mean)
        self.scale.copy_(scale)

    def forward(self, inputs):
        require(
            inputs.ndim == 2 and inputs.shape[1] == self.mean.numel(),
            "Input shape differs from fitted preprocessing",
        )
        return ((inputs.to(torch.float64) - self.mean) / self.scale).to(torch.float32)


class Predictor(nn.Module):
    """The persisted prediction path accepts raw public features and returns logits."""

    def __init__(self, normalizer: Standardizer, model: nn.Module):
        super().__init__()
        # Saving one state_dict captures both fitted buffers and model weights, so
        # training evaluation and the inference server use exactly the same transform.
        self.normalizer = normalizer
        self.model = model

    def forward(self, inputs):
        return self.model(self.normalizer(inputs))
