from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = ROOT / "scripts" / "analyze_rl_timing.py"
SPEC = importlib.util.spec_from_file_location("analyze_rl_timing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyze_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze_module)
analyze_timing = analyze_module.analyze_timing


def test_analyze_rl_timing_summary_and_last_n(tmp_path):
    path = tmp_path / "rl_train.jsonl"
    records = [
        {"kind": "run_start"},
        {"kind": "train", "tokenize_sec": 1.0, "generate_sec": 3.0, "total_micro_step_sec": 10.0},
        {"kind": "zero_advantage_skip", "tokenize_sec": 2.0, "generate_sec": 4.0, "total_micro_step_sec": 20.0},
        {"kind": "train", "tokenize_sec": "bad", "generate_sec": 100.0},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    summary = analyze_timing(path)
    assert summary["record_count"] == 3
    assert summary["timing"]["tokenize_sec"]["count"] == 2
    assert summary["timing"]["tokenize_sec"]["mean"] == 1.5
    assert summary["timing"]["tokenize_sec"]["median"] == 1.5
    assert summary["timing"]["generate_sec"]["max"] == 100.0
    assert summary["stage_total_mean_ratio"]["generate_sec"] == 0.25

    last = analyze_timing(path, last_n=1)
    assert last["record_count"] == 1
    assert last["timing"]["generate_sec"]["mean"] == 100.0


def test_analyze_rl_timing_json_cli(tmp_path):
    path = tmp_path / "rl_train.jsonl"
    path.write_text('{"kind":"train","tokenize_sec":1.0,"total_micro_step_sec":4.0}\n', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(result.stdout)
    assert summary["timing"]["tokenize_sec"]["mean"] == 1.0
    assert summary["stage_total_mean_ratio"]["tokenize_sec"] == 0.25
