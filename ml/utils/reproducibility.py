"""Explicit random state for initialization and deterministic DataLoader workers."""

import os
import random

import numpy as np
import torch

from ml.data.schema import integer


def seed_everything(seed: int) -> None:
    integer(seed, "seed", 0, 2**32 - 1)
    # Required by deterministic CUDA matrix multiplication; harmless on CPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def seeded_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)
