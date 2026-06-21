from __future__ import annotations

import os
import time
from pathlib import Path

try:
    from datasets import load_dataset, load_from_disk
except Exception:
    load_dataset = None
    load_from_disk = None


def dataset_ready_flag(path: Path) -> Path:
    return path / "_READY"


def resolve_split(ds, split: str):
    if hasattr(ds, "keys"):
        if split not in ds:
            raise KeyError(f"dataset has splits={list(ds.keys())}, missing split={split}")
        return ds[split]
    return ds


def load_dataset_auto_cached(
    path: str,
    split: str,
    *,
    hf_name: str | None = None,
    hf_config: str | None = None,
):
    path_obj = Path(path).expanduser().resolve()
    ready = dataset_ready_flag(path_obj)

    if load_from_disk is None:
        raise RuntimeError("datasets is not available; cannot use auto cache loading")

    if path_obj.exists() and ready.exists():
        return resolve_split(load_from_disk(str(path_obj)), split)

    if path_obj.exists() and not ready.exists():
        try:
            ds = load_from_disk(str(path_obj))
            ready.write_text("ok", encoding="utf-8")
            return resolve_split(ds, split)
        except Exception:
            pass

    if hf_name is None:
        raise FileNotFoundError(f"Dataset not cached at {path_obj} and hf_name is None")

    return None, ready, path_obj, hf_name, hf_config, split


def download_and_cache_dataset(
    cache_path: Path,
    ready: Path,
    hf_name: str,
    hf_config: str | None,
    split: str,
    timeout_sec: int = 3600,
):
    if load_dataset is None or load_from_disk is None:
        raise RuntimeError("datasets is not available; cannot download/cache dataset")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = cache_path.with_suffix(".lock")

    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        ds = load_dataset(hf_name, hf_config, split=split)
        ds.save_to_disk(str(cache_path))
        ready.write_text("ok", encoding="utf-8")
    except FileExistsError:
        t0 = time.time()
        while not ready.exists():
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(f"Timeout waiting for dataset cache: {cache_path}")
            time.sleep(1.0)
    finally:
        if lock.exists():
            try:
                lock.unlink()
            except Exception:
                pass

    ds = load_from_disk(str(cache_path))
    return resolve_split(ds, split="train") if hasattr(ds, "__len__") else ds
