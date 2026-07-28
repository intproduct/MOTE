from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch as tc


class IndexedJsonlDataset:
    """Read-only JSONL sequence with deterministic random access by byte offset."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"JSONL path not found: {self.path}")
        self._offsets: list[int] = []
        digest = __import__("hashlib").sha256()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                digest.update(line)
                if line.strip():
                    self._offsets.append(offset)
        self._fingerprint = digest.hexdigest()

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self._offsets)
        if not 0 <= index < len(self._offsets):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            line = handle.readline()
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {index} is not an object")
        return value


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_jsonl_gz(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_local_token_shards(path: str) -> Iterator[Dict[str, Any]]:
    base = Path(path).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Local token shard path not found: {base}")

    files: List[Path] = [base] if base.is_file() else []
    if base.is_dir():
        for pat in ["*.pt", "*.pth", "*.bin", "*.jsonl", "*.jsonl.gz", "*.txt"]:
            files.extend(sorted(base.rglob(pat)))
    if not files:
        raise FileNotFoundError(f"No shard files found under: {base}")

    for fp in files:
        suffix = fp.suffix.lower()
        if suffix in [".pt", ".pth", ".bin"]:
            obj = tc.load(fp, map_location="cpu", weights_only=False)
            if isinstance(obj, tc.Tensor):
                if obj.ndim == 1:
                    yield {"input_ids": obj.tolist()}
                elif obj.ndim == 2:
                    for row in obj:
                        yield {"input_ids": row.tolist()}
                else:
                    raise ValueError(f"Unsupported tensor ndim in {fp}: {obj.ndim}")
            elif isinstance(obj, list):
                if obj and isinstance(obj[0], int):
                    yield {"input_ids": obj}
                else:
                    for item in obj:
                        if isinstance(item, dict):
                            yield item
                        elif isinstance(item, tc.Tensor):
                            if item.ndim == 1:
                                yield {"input_ids": item.tolist()}
                            elif item.ndim == 2:
                                for row in item:
                                    yield {"input_ids": row.tolist()}
                        else:
                            yield {"input_ids": list(item)}
            elif isinstance(obj, dict):
                if "input_ids" in obj:
                    value = obj["input_ids"]
                    if isinstance(value, tc.Tensor):
                        if value.ndim == 1:
                            yield {"input_ids": value.tolist()}
                        elif value.ndim == 2:
                            for row in value:
                                yield {"input_ids": row.tolist()}
                    elif isinstance(value, list):
                        if value and isinstance(value[0], int):
                            yield {"input_ids": value}
                        else:
                            for row in value:
                                yield {"input_ids": list(row)}
                else:
                    for value in obj.values():
                        if isinstance(value, tc.Tensor) and value.ndim == 2:
                            for row in value:
                                yield {"input_ids": row.tolist()}
        elif suffix == ".jsonl":
            yield from iter_jsonl(fp)
        elif suffix == ".gz" and fp.name.endswith(".jsonl.gz"):
            yield from iter_jsonl_gz(fp)
        elif suffix == ".txt":
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.strip():
                        yield {"text": line}
