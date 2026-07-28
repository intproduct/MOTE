from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import content_hash, stable_json_bytes


SAMPLER_STATE_FORMAT = "fitmotn_source_sampler_state_v1"


def _derived_seed(seed: int, namespace: str, epoch: int) -> int:
    payload = f"{int(seed)}\0{namespace}\0{int(epoch)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def sequence_fingerprint(records: Sequence[Any]) -> str:
    explicit = getattr(records, "_fingerprint", None)
    if explicit:
        return content_hash({"count": len(records), "fingerprint": str(explicit)})
    digest = hashlib.sha256()
    digest.update(str(len(records)).encode("ascii"))
    digest.update(b"\n")
    for record in records:
        digest.update(stable_json_bytes(record))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class ShardContext:
    rank: int
    world_size: int
    worker_id: int
    worker_count: int

    @property
    def shard_id(self) -> int:
        return self.rank * self.worker_count + self.worker_id

    @property
    def shard_count(self) -> int:
        return self.world_size * self.worker_count


def resolve_shard_context(worker_info=None) -> ShardContext:
    worker_id = int(getattr(worker_info, "id", 0) if worker_info is not None else 0)
    worker_count = int(getattr(worker_info, "num_workers", 1) if worker_info is not None else 1)
    rank = 0
    world_size = 1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
    except (ImportError, RuntimeError):
        pass
    return ShardContext(rank=rank, world_size=world_size, worker_id=worker_id, worker_count=worker_count)


class DeterministicSequenceSampler:
    """Deterministic, epoch-aware, without-replacement traversal of a finite source."""

    def __init__(
        self,
        records: Sequence[Any],
        *,
        seed: int,
        namespace: str,
        shard_id: int = 0,
        shard_count: int = 1,
        max_samples: int | None = None,
        max_epochs: int | None = None,
        shuffle: bool = True,
        dataset_fingerprint: str | None = None,
    ) -> None:
        if not hasattr(records, "__len__") or not hasattr(records, "__getitem__"):
            raise TypeError("deterministic without-replacement sampling requires a finite indexable source")
        if len(records) <= 0:
            raise ValueError("source sampler requires at least one record")
        if int(shard_count) <= 0 or not 0 <= int(shard_id) < int(shard_count):
            raise ValueError(f"invalid shard {shard_id}/{shard_count}")
        self.records = records
        self.seed = int(seed)
        self.namespace = str(namespace)
        self.shard_id = int(shard_id)
        self.shard_count = int(shard_count)
        self.max_samples = None if max_samples is None or int(max_samples) <= 0 else min(int(max_samples), len(records))
        self.max_epochs = None if max_epochs is None or int(max_epochs) <= 0 else int(max_epochs)
        self.shuffle = bool(shuffle)
        self.dataset_fingerprint = dataset_fingerprint or sequence_fingerprint(records)
        self.epoch = 0
        self.position = 0
        self.total_draws = 0
        self._draw_counts: Counter[int] = Counter()
        self.order = self._build_order(self.epoch)
        if not self.order:
            raise ValueError(
                f"source shard {self.shard_id}/{self.shard_count} is empty; "
                "reduce shard_count or increase max_samples"
            )

    def _build_global_order(self, epoch: int) -> list[int]:
        order = list(range(len(self.records)))
        if self.shuffle:
            random.Random(_derived_seed(self.seed, self.namespace, epoch)).shuffle(order)
        if self.max_samples is not None:
            order = order[: self.max_samples]
        return order

    def _build_order(self, epoch: int) -> list[int]:
        return self._build_global_order(epoch)[self.shard_id :: self.shard_count]

    @property
    def selected_count(self) -> int:
        return len(self._build_global_order(self.epoch))

    def next_record(self) -> Any:
        if self.position >= len(self.order):
            if self.max_epochs is not None and self.epoch + 1 >= self.max_epochs:
                raise RuntimeError(
                    f"source {self.namespace} reached configured max_epochs={self.max_epochs}; "
                    "refusing additional repeated exposure"
                )
            self.epoch += 1
            self.position = 0
            self.order = self._build_order(self.epoch)
            if not self.order:
                raise RuntimeError("source shard became empty at an epoch boundary")
        record_index = int(self.order[self.position])
        self.position += 1
        self.total_draws += 1
        self._draw_counts[record_index] += 1
        return self.records[record_index]

    def __iter__(self) -> Iterator[Any]:
        while True:
            yield self.next_record()

    def statistics(self) -> dict[str, Any]:
        unique = len(self._draw_counts)
        return {
            "namespace": self.namespace,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_record_count": len(self.records),
            "selected_record_count": self.selected_count,
            "shard_id": self.shard_id,
            "shard_count": self.shard_count,
            "epoch": self.epoch,
            "position": self.position,
            "total_draws": self.total_draws,
            "unique_draws": unique,
            "repeat_count": max(0, self.total_draws - unique),
            "max_repeat_per_sample": max(self._draw_counts.values(), default=0),
            "coverage_rate": float(unique / max(1, len(self.records))),
            "current_epoch_coverage_rate": float(min(self.position, len(self.order)) / max(1, len(self.order))),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": SAMPLER_STATE_FORMAT,
            "seed": self.seed,
            "namespace": self.namespace,
            "dataset_fingerprint": self.dataset_fingerprint,
            "record_count": len(self.records),
            "max_samples": self.max_samples,
            "max_epochs": self.max_epochs,
            "shuffle": self.shuffle,
            "shard_id": self.shard_id,
            "shard_count": self.shard_count,
            "epoch": self.epoch,
            "position": self.position,
            "total_draws": self.total_draws,
            "draw_counts": dict(self._draw_counts),
        }

    @classmethod
    def from_state(cls, records: Sequence[Any], state: Mapping[str, Any]) -> "DeterministicSequenceSampler":
        if state.get("format") != SAMPLER_STATE_FORMAT:
            raise RuntimeError("unsupported source sampler state format")
        sampler = cls(
            records,
            seed=int(state["seed"]),
            namespace=str(state["namespace"]),
            shard_id=int(state["shard_id"]),
            shard_count=int(state["shard_count"]),
            max_samples=state.get("max_samples"),
            max_epochs=state.get("max_epochs"),
            shuffle=bool(state.get("shuffle", True)),
        )
        if int(state.get("record_count", -1)) != len(records):
            raise RuntimeError("source sampler record count changed since checkpoint")
        if state.get("dataset_fingerprint") != sampler.dataset_fingerprint:
            raise RuntimeError("source sampler dataset content changed since checkpoint")
        sampler.epoch = int(state.get("epoch", 0))
        sampler.order = sampler._build_order(sampler.epoch)
        sampler.position = int(state.get("position", -1))
        if not 0 <= sampler.position <= len(sampler.order):
            raise RuntimeError("source sampler checkpoint position is invalid")
        sampler.total_draws = int(state.get("total_draws", 0))
        sampler._draw_counts = Counter({int(key): int(value) for key, value in dict(state.get("draw_counts") or {}).items()})
        if any(index < 0 or index >= len(records) or count <= 0 for index, count in sampler._draw_counts.items()):
            raise RuntimeError("source sampler checkpoint draw counts are invalid")
        return sampler
