from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Sequence


SAMPLER_STATE_FORMAT = "fitmotn_rl_sampler_state_v1"


def records_fingerprint(records: Sequence[Dict[str, Any]]) -> str:
    """Return a stable identity for the ordered RL dataset."""
    digest = hashlib.sha256()
    for record in records:
        payload = {
            "idx": record.get("idx"),
            "source": record.get("source"),
            "question": record.get("question"),
            "answer": record.get("answer"),
        }
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class StatefulRLRecordSampler:
    """Epoch-aware sampler whose exact traversal can be checkpointed."""

    def __init__(self, records: Sequence[Dict[str, Any]], *, shuffle: bool, seed: int) -> None:
        if not records:
            raise ValueError("RL sampler requires at least one record")
        self._records = records
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.dataset_fingerprint = records_fingerprint(records)
        self.epoch = 0
        self.position = 0
        self.samples_seen = 0
        self._rng = random.Random(self.seed)
        self.order = list(range(len(records)))
        if self.shuffle:
            self._rng.shuffle(self.order)

    def next_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        batch: List[Dict[str, Any]] = []
        for _ in range(int(batch_size)):
            if self.position >= len(self.order):
                self.epoch += 1
                self.position = 0
                self.order = list(range(len(self._records)))
                if self.shuffle:
                    self._rng.shuffle(self.order)
            batch.append(self._records[self.order[self.position]])
            self.position += 1
            self.samples_seen += 1
        return batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": SAMPLER_STATE_FORMAT,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "record_count": len(self._records),
            "dataset_fingerprint": self.dataset_fingerprint,
            "epoch": self.epoch,
            "position": self.position,
            "samples_seen": self.samples_seen,
            "order": list(self.order),
            "rng_state": self._rng.getstate(),
        }

    @classmethod
    def from_state(
        cls,
        records: Sequence[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> "StatefulRLRecordSampler":
        if not isinstance(state, dict) or state.get("format") != SAMPLER_STATE_FORMAT:
            raise RuntimeError("Unsupported RL sampler state format")
        expected_count = len(records)
        if int(state.get("record_count", -1)) != expected_count:
            raise RuntimeError(
                "RL sampler dataset size changed since checkpoint: "
                f"checkpoint={state.get('record_count')} current={expected_count}"
            )
        fingerprint = records_fingerprint(records)
        if state.get("dataset_fingerprint") != fingerprint:
            raise RuntimeError("RL sampler dataset content/order changed since checkpoint")
        order = [int(index) for index in state.get("order", [])]
        if sorted(order) != list(range(expected_count)):
            raise RuntimeError("RL sampler checkpoint order is not a valid dataset permutation")
        position = int(state.get("position", -1))
        if not 0 <= position <= expected_count:
            raise RuntimeError(f"RL sampler checkpoint position is invalid: {position}")

        sampler = cls(records, shuffle=bool(state.get("shuffle", False)), seed=int(state.get("seed", 0)))
        sampler.epoch = int(state.get("epoch", 0))
        sampler.position = position
        sampler.samples_seen = int(state.get("samples_seen", 0))
        sampler.order = order
        try:
            sampler._rng.setstate(state["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("RL sampler checkpoint RNG state is invalid") from exc
        return sampler

