"""Best-effort CUDA diagnostics, including hosts with all devices hidden."""
from __future__ import annotations

import torch


def visible_cuda_device(device=None) -> int | None:
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return None
        if isinstance(device, int):
            index = device
        else:
            resolved = torch.device(device) if device is not None else torch.device("cuda")
            if resolved.type != "cuda":
                return None
            index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        return index if 0 <= index < torch.cuda.device_count() else None
    except (RuntimeError, AssertionError, ValueError):
        return None


def cuda_memory_bytes(device=None) -> dict[str, int | None]:
    result = dict.fromkeys(("allocated", "reserved", "max_allocated", "max_reserved"))
    index = visible_cuda_device(device)
    if index is None:
        return result
    try:
        return {
            "allocated": torch.cuda.memory_allocated(index),
            "reserved": torch.cuda.memory_reserved(index),
            "max_allocated": torch.cuda.max_memory_allocated(index),
            "max_reserved": torch.cuda.max_memory_reserved(index),
        }
    except (RuntimeError, AssertionError, ValueError):
        return result
