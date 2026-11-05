from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from fitmotn.audit import cuda_snapshot
from fitmotn.data.release_utils import canonical_json_line, integer_distribution
from fitmotn.utils.files import sha256_file
from fitmotn.utils.logging import build_run_logger
from fitmotn.utils.memory import cuda_memory_bytes


def test_objective_import_avoids_loading_training_and_rollout_controllers():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from fitmotn.train import objectives; "
         "assert 'fitmotn.train.controller' not in sys.modules; "
         "assert 'fitmotn.train.rl_controller' not in sys.modules; "
         "assert 'transformers.trainer' not in sys.modules; "
         "from fitmotn.train import run_fitmotn_training; "
         "from fitmotn.train.controller import run_fitmotn_training as original; "
         "assert run_fitmotn_training is original"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_release_primitives_preserve_serialized_bytes_and_nearest_rank(tmp_path):
    line = canonical_json_line({"z": "中文", "path": Path("data"), "a": [1, None]})
    assert line == '{"a":[1,null],"path":"data","z":"中文"}\n'
    payload = line.encode("utf-8") * 100_000
    path = tmp_path / "records.jsonl"
    path.write_bytes(payload)
    assert sha256_file(str(path)) == hashlib.sha256(payload).hexdigest()
    # Iterator input is supported without changing the established percentile rule.
    assert integer_distribution(iter([4, 1, 9, 2])) == {
        "count": 4, "mean": 4.0, "p50": 2, "p90": 9, "p95": 9, "p99": 9, "max": 9,
    }
    assert integer_distribution(iter([])) == {
        "count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None,
    }


def test_run_logger_releases_previous_file_and_does_not_duplicate_records(tmp_path):
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    logger = build_run_logger("fitmotn.test.repeated_run", first)
    previous_file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    logger.info("first_run")
    logger = build_run_logger(logger.name, second)
    try:
        assert previous_file_handler.stream is None
        logger.info("second_run")
        assert first.read_text(encoding="utf-8").count("first_run") == 1
        assert "second_run" not in first.read_text(encoding="utf-8")
        assert second.read_text(encoding="utf-8").count("second_run") == 1
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


@pytest.mark.parametrize("device_count,device", [(0, None), (1, "cpu"), (1, "cuda:2")])
def test_memory_diagnostics_do_not_query_hidden_or_invalid_devices(monkeypatch, device_count, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def unexpected_query(*args):
        raise AssertionError("invalid device must not reach memory queries")

    monkeypatch.setattr(torch.cuda, "memory_allocated", unexpected_query)
    assert all(v is None for v in cuda_memory_bytes(device).values())
    assert all(v is None for v in cuda_snapshot(device).values())


def test_sft_and_rl_memory_units_and_device_selection_are_preserved(monkeypatch):
    from fitmotn.train.rl_controller import _cuda_memory_snapshot, log_cuda_memory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    for name, value in [("memory_allocated", 1), ("memory_reserved", 2),
                        ("max_memory_allocated", 3), ("max_memory_reserved", 4)]:
        def read_memory(index, value=value):
            assert index == 1
            return value * 1024 ** 3
        monkeypatch.setattr(torch.cuda, name, read_memory)
    assert cuda_snapshot("cuda:1")["cuda_mem_peak_alloc_mb"] == 3072.0
    assert _cuda_memory_snapshot() == {
        "memory_allocated_gib": 1.0, "memory_reserved_gib": 2.0,
        "memory_max_allocated_gib": 3.0, "memory_max_reserved_gib": 4.0,
    }
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    # Reproduces the pre-existing hidden-device failure during RL logging.
    log_cuda_memory(logging.getLogger("test.hidden_device"), "hidden", reset_peak=True)


def test_memory_driver_error_is_non_fatal(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def driver_error(*args):
        raise RuntimeError("driver query failed")

    monkeypatch.setattr(torch.cuda, "memory_allocated", driver_error)
    assert all(v is None for v in cuda_memory_bytes().values())
