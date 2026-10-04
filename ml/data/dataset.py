"""Re-iterable JSONL datasets with bounded memory and explicit trajectory selection."""

import copy
import hashlib
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .schema import parse_json, read_metadata, require, validate_row


class MarketDataset(IterableDataset):
    """Validate once, then stream one row at a time. Files must remain immutable."""

    def __init__(self, directory: str | Path):
        super().__init__()
        self.directory = Path(directory).resolve()
        self.metadata = read_metadata(self.directory)
        self.path = self.directory / self.metadata["datasetFile"]
        self._signature = self._file_signature()
        self.trajectories = frozenset(range(self.metadata["generation"]["actualTrajectoryCount"]))
        # Scan once to reject corrupt/truncated exports before any optimizer step.
        # Later epochs reopen the same immutable file instead of storing all rows in RAM.
        checksum = hashlib.sha256()
        counts = [0] * self.metadata["numClasses"]
        count = 0
        with self.path.open("rb") as stream:
            for index, line in enumerate(stream):
                row = self._parse(line, index)
                checksum.update(line)
                counts[row["target"]] += 1
                count += 1
        require(count == self.metadata["sampleCount"], "Dataset row count differs from metadata")
        require(checksum.hexdigest() == self.metadata["sha256"], "Dataset SHA-256 mismatch")
        require(
            counts == [self.metadata["classCounts"][n] for n in self.metadata["classNames"]],
            "Dataset class counts differ from metadata",
        )
        self._assert_unchanged()

    def _file_signature(self):
        stat = self.path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _assert_unchanged(self):
        require(
            self._file_signature() == self._signature,
            "Dataset changed after validation; create a new dataset instance",
        )

    def _parse(self, line: bytes, index: int):
        try:
            row = parse_json(line)
            validate_row(row, self.metadata, index)
            return row
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise ValueError(f"{self.path}, line {index + 1}: {error}") from error

    def select(self, trajectories) -> "MarketDataset":
        selected = frozenset(trajectories)
        require(
            bool(selected) and selected <= self.trajectories, "Invalid or empty trajectory subset"
        )
        result = copy.copy(self)
        result.trajectories = selected
        return result

    def __len__(self):
        g = self.metadata["generation"]
        return sum(
            min(g["samplesPerTrajectory"], g["sampleCount"] - t * g["samplesPerTrajectory"])
            for t in self.trajectories
        )

    def __iter__(self):
        self._assert_unchanged()
        worker = get_worker_info()
        workers, worker_id = (worker.num_workers, worker.id) if worker else (1, 0)
        selected_index = 0
        with self.path.open("rb") as stream:
            for index, line in enumerate(stream):
                trajectory = index // self.metadata["generation"]["samplesPerTrajectory"]
                if trajectory not in self.trajectories:
                    continue
                # IterableDataset workers each read the file: assign disjoint rows
                # explicitly so adding workers does not silently duplicate examples.
                assigned = selected_index % workers == worker_id
                selected_index += 1
                if not assigned:
                    continue
                row = self._parse(line, index)
                yield (
                    torch.tensor(row["input"], dtype=torch.float32),
                    torch.tensor(row["target"], dtype=torch.long),
                )
        self._assert_unchanged()


def split_trajectories(dataset: MarketDataset, fraction: float, seed: int):
    """Keep whole seeded simulations apart; nearby rows share market history and labels."""
    require(0 < fraction < 1, "validation_split must lie strictly between zero and one")
    trajectories = sorted(dataset.trajectories)
    require(
        len(trajectories) >= 2,
        "Validation split needs at least two trajectories; "
        "supply an independent validation dataset",
    )
    random.Random(seed).shuffle(trajectories)
    count = max(1, min(len(trajectories) - 1, round(len(trajectories) * fraction)))
    return dataset.select(trajectories[count:]), dataset.select(trajectories[:count])


class ShuffleBuffer(IterableDataset):
    """Approximate streaming shuffle; rows are otherwise ordered trajectory by trajectory.

    Memory stays bounded by buffer_size. Each epoch draws from a generator seeded by
    (seed, epoch, worker), so repeated training runs see identical batch orders.
    """

    def __init__(self, source: IterableDataset, buffer_size: int, seed: int):
        super().__init__()
        require(buffer_size >= 1, "Shuffle buffer must hold at least one row")
        self.source = source
        self.buffer_size = buffer_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        generator = random.Random(f"{self.seed}:{self.epoch}:{worker_id}")
        buffer = []
        for row in self.source:
            if len(buffer) < self.buffer_size:
                buffer.append(row)
                continue
            index = generator.randrange(self.buffer_size)
            yield buffer[index]
            buffer[index] = row
        generator.shuffle(buffer)
        yield from buffer
