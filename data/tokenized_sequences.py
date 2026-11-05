"""Indexable, fingerprinted JSONL sequences for distributed SFT.

Each record contains input_ids, labels and optional loss_weights. Sequence
packing/padding is deliberately not inferred; callers control supervision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenizedSequenceDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.offsets = []
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                if line.strip():
                    self.offsets.append(offset)
        self.fingerprint = digest.hexdigest()
        stat = self.path.stat()
        self.file_identity = (stat.st_size, stat.st_mtime_ns)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.file_identity:
            raise RuntimeError("Tokenized dataset changed during training")
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            record = json.loads(stream.readline())
        return {key: torch.tensor(value, dtype=torch.float32 if key == "loss_weights" else torch.long)
                for key, value in record.items()}
