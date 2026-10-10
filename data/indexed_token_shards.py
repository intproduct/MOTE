from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch as tc


class IndexedTokenShardDataset:
    """Finite, indexable token tensors with an exact-resume fingerprint."""

    INDEX_NAME = "TOKEN_SHARD_INDEX.json"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        index_path = self.path / self.INDEX_NAME if self.path.is_dir() else self.path.with_name(self.INDEX_NAME)
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else None
        if index is not None:
            base = index_path.parent
            self.files = [base / item["path"] for item in index["files"]]
            expected_rows = [int(item["rows"]) for item in index["files"]]
            self._fingerprint = str(index["fingerprint"])
        else:
            if self.path.is_file() and self.path.suffix.lower() in {".pt", ".pth", ".bin"}:
                self.files = [self.path]
            elif self.path.is_dir():
                self.files = sorted(
                    file
                    for pattern in ("*.pt", "*.pth", "*.bin")
                    for file in self.path.rglob(pattern)
                )
            else:
                self.files = []
            expected_rows = []
            digest = hashlib.sha256()
            for file in self.files:
                stat = file.stat()
                relative = file.relative_to(self.path if self.path.is_dir() else self.path.parent)
                digest.update(str(relative).encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
            self._fingerprint = digest.hexdigest()
        if not self.files or any(not file.is_file() for file in self.files):
            raise FileNotFoundError(f"No indexed tensor shard files found under: {self.path}")
        # Keep every row-aligned tensor so masked labels and provenance survive.
        # Input_ids-only releases remain backward compatible.
        self._tensors = []
        self._payloads: list[dict[str, tc.Tensor]] = []
        self._ends = []
        total = 0
        for file_index, file in enumerate(self.files):
            obj = tc.load(file, map_location="cpu", weights_only=True)
            value = obj.get("input_ids") if isinstance(obj, dict) else obj
            if not isinstance(value, tc.Tensor) or value.ndim != 2:
                raise ValueError(f"Indexed token shard must contain rank-2 input_ids: {file}")
            rows = int(value.shape[0])
            if expected_rows and rows != expected_rows[file_index]:
                raise RuntimeError(
                    f"Indexed token shard row mismatch: {file} rows={rows} expected={expected_rows[file_index]}"
                )
            payload: dict[str, tc.Tensor] = {"input_ids": value}
            if isinstance(obj, dict):
                for key, candidate in obj.items():
                    if key == "input_ids" or not isinstance(candidate, tc.Tensor):
                        continue
                    if candidate.ndim < 1 or int(candidate.shape[0]) != rows:
                        continue
                    if key in {"labels", "attention_mask", "loss_weights"} and (
                        candidate.ndim != 2 or tuple(candidate.shape) != tuple(value.shape)
                    ):
                        raise ValueError(f"Indexed token shard {key} shape mismatch: {file}")
                    payload[str(key)] = candidate
            self._tensors.append(value)
            total += rows
            self._payloads.append(payload)
            self._ends.append(total)
        if index is not None and total != int(index["total_rows"]):
            raise RuntimeError(f"Indexed token release row mismatch: rows={total} expected={index['total_rows']}")

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self._ends, index)
        begin = 0 if shard == 0 else self._ends[shard - 1]
        row = index - begin
        result: Dict[str, Any] = {}
        for key, tensor in self._payloads[shard].items():
            value = tensor[row]
            result[key] = value.item() if value.ndim == 0 else value.tolist()
        return result
