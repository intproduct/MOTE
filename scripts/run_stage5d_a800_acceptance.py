#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitmotn.config import load_config
from fitmotn.train.rl_controller import resolve_rl_output_dir
from fitmotn.rl.vllm_weight_transfer_capabilities import probe_vllm_weight_transfer_capabilities


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(command: list[str], *, log_path: Path, env=None) -> int:
    print(f"[Stage5D] RUN: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return int(process.wait())


def _capture(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, check=False)
        return {
            "command": command,
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        return {"command": command, "returncode": -1, "stdout": "", "stderr": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 5D three-A800 native-sync acceptance")
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence-dir")
    parser.add_argument("--min-updates", type=int, default=20)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--max-sync-p95-sec", type=float, default=60.0)
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    source_config = Path(args.config).expanduser().resolve()
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    if args.max_steps is not None:
        payload.setdefault("rl", {})["max_steps"] = int(args.max_steps)
    if args.run_name:
        payload.setdefault("output", {})["run_name"] = str(args.run_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence = (
        Path(args.evidence_dir).expanduser().resolve()
        if args.evidence_dir
        else source_config.parent / "stage5d_evidence" / stamp
    )
    evidence.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_config, evidence / "input_config.json")
    effective = evidence / "effective_config.json"
    _write(effective, payload)
    cfg = load_config(config_json=str(effective))
    rl_dir = resolve_rl_output_dir(cfg)
    report: dict[str, Any] = {
        "stage": "5D-full-weight-native-sync",
        "ok": False,
        "started_at": time.time(),
        "effective_config": str(effective),
        "rl_dir": str(rl_dir),
        "commands": [],
    }
    nvidia = _capture(["nvidia-smi"])
    _write(evidence / "nvidia_smi.json", nvidia)
    _write(evidence / "nvidia_smi_topology.json", _capture(["nvidia-smi", "topo", "-m"]))
    capabilities = probe_vllm_weight_transfer_capabilities().to_dict()
    _write(evidence / "weight_transfer_capabilities.json", capabilities)
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
    except Exception:
        cuda_available = False
        cuda_device_count = 0
    if not cuda_available or cuda_device_count < 3:
        report["error"] = (
            "Stage 5D acceptance requires at least three visible CUDA GPUs; "
            f"available={cuda_available}, count={cuda_device_count}"
        )
        _write(evidence / "stage5d_acceptance_summary.json", report)
        return 2
    if int(nvidia.get("returncode", -1)) != 0:
        report["error"] = "nvidia-smi failed"
        _write(evidence / "stage5d_acceptance_summary.json", report)
        return 2
    if capabilities.get("native_transfer_level") not in {"update_only", "four_phase"}:
        report["error"] = f"vLLM native transfer API is unavailable: {capabilities}"
        _write(evidence / "stage5d_acceptance_summary.json", report)
        return 2

    preflight = [
        sys.executable,
        "scripts/validate_stage5d_native_sync.py",
        "--config",
        str(effective),
        "--output-json",
        str(evidence / "preflight.json"),
    ]
    code = _run(preflight, log_path=evidence / "preflight.log")
    report["commands"].append({"name": "preflight", "returncode": code})
    if code != 0:
        report["error"] = "native-sync preflight failed"
        _write(evidence / "stage5d_acceptance_summary.json", report)
        return code

    if not args.skip_pytest:
        tests = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_vllm_actor.py",
            "tests/test_vllm_weight_transfer.py",
            "tests/test_vllm_rollout_backend.py",
            "tests/test_stage5d_native_sync_validation.py",
        ]
        code = _run(tests, log_path=evidence / "pytest.log")
        report["commands"].append({"name": "pytest", "returncode": code})
        if code != 0:
            report["error"] = "focused tests failed"
            _write(evidence / "stage5d_acceptance_summary.json", report)
            return code

    if int(cfg.rl.max_steps) < int(args.min_updates):
        report["error"] = f"rl.max_steps={cfg.rl.max_steps} is below min_updates={args.min_updates}"
        _write(evidence / "stage5d_acceptance_summary.json", report)
        return 2
    train = [sys.executable, "-m", "fitmotn.cli.train_rl", "--config_json", str(effective)]
    training_start = time.time()
    code = _run(train, log_path=evidence / "training.log", env=os.environ.copy())
    report["training_duration_sec"] = max(0.0, time.time() - training_start)
    report["commands"].append({"name": "train_rl", "returncode": code})
    if code != 0:
        report["error"] = "RL training failed"
        _write(evidence / "stage5d_acceptance_summary.json", report)
        return code

    checks = {
        "native_sync": [
            sys.executable,
            "scripts/validate_stage5d_native_sync.py",
            "--config",
            str(effective),
            "--rl-train-jsonl",
            str(rl_dir / "rl_train.jsonl"),
            "--export-root",
            str(rl_dir / "vllm_sync"),
            "--min-updates",
            str(args.min_updates),
            "--max-sync-p95-sec",
            str(args.max_sync_p95_sec),
            "--output-json",
            str(evidence / "native_sync.json"),
        ],
        "final_checkpoint": [
            sys.executable,
            "-m",
            "fitmotn.cli.validate_checkpoint",
            str(rl_dir / "final_model"),
            "--require-exact",
            "rl",
            "--output-json",
            str(evidence / "final_checkpoint.json"),
        ],
    }
    failures = []
    for name, command in checks.items():
        code = _run(command, log_path=evidence / f"{name}.log")
        report["commands"].append({"name": name, "returncode": code})
        if code != 0:
            failures.append(name)
    report["finished_at"] = time.time()
    report["duration_sec"] = report["finished_at"] - report["started_at"]
    report["ok"] = not failures
    report["failed_checks"] = failures
    _write(evidence / "stage5d_acceptance_summary.json", report)
    print(f"Stage 5D acceptance {'PASSED' if report['ok'] else 'FAILED'}: {evidence}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
