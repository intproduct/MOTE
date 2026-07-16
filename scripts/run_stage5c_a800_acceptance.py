#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitmotn.config import load_config
from fitmotn.diagnostics.config_doctor import inspect_config
from fitmotn.train.rl_controller import resolve_rl_output_dir


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_streaming(command: list[str], *, cwd: Path, log_path: Path, env=None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Stage5C] RUN: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
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


def _capture(command: list[str], *, cwd: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
        return {
            "command": command,
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        return {"command": command, "returncode": -1, "stdout": "", "stderr": str(exc)}


def _environment_report() -> dict[str, Any]:
    versions = {}
    for module_name in ("torch", "transformers", "vllm"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = {"error": str(exc)}
    try:
        import torch

        versions["torch_cuda_version"] = torch.version.cuda
        versions["torch_cuda_available"] = bool(torch.cuda.is_available())
        versions["torch_cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        versions["torch_cuda_probe_error"] = str(exc)
    return {
        "time": time.time(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "cwd": str(ROOT),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": versions,
    }


def _start_gpu_monitor(evidence_dir: Path):
    output = (evidence_dir / "nvidia_smi_compute_apps.csv").open("w", encoding="utf-8")
    command = [
        "nvidia-smi",
        "--query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory",
        "--format=csv",
        "-l",
        "5",
    ]
    try:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=output, stderr=subprocess.STDOUT, text=True)
        return process, output
    except Exception:
        output.close()
        return None, None


def _stop_gpu_monitor(process, output) -> None:
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if output is not None:
        output.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete Stage 5C A800 training smoke test and collect acceptance evidence"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence-dir")
    parser.add_argument("--min-updates", type=int, default=20)
    parser.add_argument("--max-steps", type=int, help="Override rl.max_steps in the generated effective config")
    parser.add_argument("--run-name", help="Override output.run_name so smoke and full runs stay separate")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-training", action="store_true", help="Only run preflight against the configuration")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_json=str(config_path))
    if args.max_steps is not None:
        if int(args.max_steps) < 1:
            parser.error("--max-steps must be >= 1")
        cfg.rl.max_steps = int(args.max_steps)
    if args.run_name:
        cfg.output.run_name = str(args.run_name)
    rl_dir = resolve_rl_output_dir(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence_dir = (
        Path(args.evidence_dir).expanduser().resolve()
        if args.evidence_dir
        else Path(cfg.output.root_dir).expanduser().resolve() / "stage5c_evidence" / timestamp
    )
    evidence_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, evidence_dir / "input_config.json")
    effective_config_path = evidence_dir / "effective_config.json"
    effective_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if args.max_steps is not None:
        effective_payload.setdefault("rl", {})["max_steps"] = int(args.max_steps)
    if args.run_name:
        effective_payload.setdefault("output", {})["run_name"] = str(args.run_name)
    _write_json(effective_config_path, effective_payload)

    report: dict[str, Any] = {
        "stage": "5C",
        "ok": False,
        "config": str(config_path),
        "effective_config": str(effective_config_path),
        "rl_dir": str(rl_dir),
        "evidence_dir": str(evidence_dir),
        "min_updates": int(args.min_updates),
        "commands": [],
        "started_at": time.time(),
    }
    environment_report = _environment_report()
    nvidia_report = _capture(["nvidia-smi"], cwd=ROOT)
    _write_json(evidence_dir / "environment.json", environment_report)
    _write_json(evidence_dir / "nvidia_smi.json", nvidia_report)
    _write_json(evidence_dir / "nvidia_smi_L.json", _capture(["nvidia-smi", "-L"], cwd=ROOT))
    _write_json(evidence_dir / "nvidia_smi_topology.json", _capture(["nvidia-smi", "topo", "-m"], cwd=ROOT))

    doctor = inspect_config(cfg)
    _write_json(evidence_dir / "doctor.json", doctor)
    if not doctor.get("ok"):
        report["error"] = f"configuration doctor failed: {doctor.get('errors')}"
        _write_json(evidence_dir / "stage5c_acceptance_summary.json", report)
        print(report["error"], file=sys.stderr)
        return 2
    if not args.skip_training:
        versions = dict(environment_report.get("versions") or {})
        if not bool(versions.get("torch_cuda_available", False)):
            report["error"] = "torch.cuda.is_available() is false; A800 training acceptance cannot start"
        elif int(nvidia_report.get("returncode", -1)) != 0:
            report["error"] = f"nvidia-smi failed: {nvidia_report.get('stderr')}"
        elif isinstance(versions.get("vllm"), dict):
            report["error"] = f"vLLM import failed: {versions.get('vllm')}"
        else:
            topology = dict((doctor.get("rl") or {}).get("resource_topology") or {})
            required_devices = 1 + int(topology.get("total_rollout_gpus", 0))
            observed_devices = int(versions.get("torch_cuda_device_count", 0))
            if observed_devices < required_devices:
                report["error"] = (
                    f"CUDA process sees {observed_devices} GPUs but Stage 5C topology requires "
                    f"at least {required_devices}"
                )
        if report.get("error"):
            _write_json(evidence_dir / "stage5c_acceptance_summary.json", report)
            print(report["error"], file=sys.stderr)
            return 2
    if int(cfg.rl.max_steps) < int(args.min_updates) and not args.skip_training:
        report["error"] = f"rl.max_steps={cfg.rl.max_steps} is below --min-updates={args.min_updates}"
        _write_json(evidence_dir / "stage5c_acceptance_summary.json", report)
        print(report["error"], file=sys.stderr)
        return 2

    preflight_command = [
        sys.executable,
        "scripts/validate_stage5c_multiactor.py",
        "--config",
        str(effective_config_path),
        "--output-json",
        str(evidence_dir / "stage5c_preflight.json"),
    ]
    code = _run_streaming(preflight_command, cwd=ROOT, log_path=evidence_dir / "stage5c_preflight.log")
    report["commands"].append({"name": "stage5c_preflight", "returncode": code})
    if code != 0:
        report["error"] = "Stage 5C topology preflight failed"
        _write_json(evidence_dir / "stage5c_acceptance_summary.json", report)
        return code

    if not args.skip_pytest:
        test_command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5b_device_topology.py",
            "tests/test_vllm_actor.py",
            "tests/test_vllm_rollout_backend.py",
            "tests/test_stage4_validation_scripts.py",
        ]
        code = _run_streaming(test_command, cwd=ROOT, log_path=evidence_dir / "pytest_stage5c.log")
        report["commands"].append({"name": "pytest_stage5c", "returncode": code})
        if code != 0:
            report["error"] = "Stage 5C focused tests failed"
            _write_json(evidence_dir / "stage5c_acceptance_summary.json", report)
            return code

    if args.skip_training:
        report.update({"ok": True, "preflight_only": True, "finished_at": time.time()})
        _write_json(evidence_dir / "stage5c_acceptance_summary.json", report)
        print(f"Stage 5C preflight passed. Evidence: {evidence_dir}")
        return 0

    monitor, monitor_output = _start_gpu_monitor(evidence_dir)
    try:
        train_command = [sys.executable, "-m", "fitmotn.cli.train_rl", "--config_json", str(effective_config_path)]
        code = _run_streaming(train_command, cwd=ROOT, log_path=evidence_dir / "training.log", env=os.environ.copy())
        report["commands"].append({"name": "train_rl", "returncode": code})
    finally:
        _stop_gpu_monitor(monitor, monitor_output)
    if code != 0:
        report["error"] = "RL training command failed"
        report["finished_at"] = time.time()
        _write_json(evidence_dir / "stage5c_acceptance_summary.json", report)
        return code

    rl_jsonl = rl_dir / "rl_train.jsonl"
    checks = [
        (
            "stage5c_run",
            [
                sys.executable,
                "scripts/validate_stage5c_multiactor.py",
                "--config",
                str(effective_config_path),
                "--rl-train-jsonl",
                str(rl_jsonl),
                "--min-updates",
                str(args.min_updates),
                "--output-json",
                str(evidence_dir / "stage5c_run.json"),
            ],
        ),
        (
            "stage4_run",
            [
                sys.executable,
                "scripts/validate_stage4_rl_run.py",
                str(rl_jsonl),
                "--min-updates",
                str(args.min_updates),
                "--output-json",
                str(evidence_dir / "stage4_run.json"),
            ],
        ),
        (
            "sync_artifacts",
            [
                sys.executable,
                "scripts/validate_stage4_sync_artifacts.py",
                str(rl_dir / "vllm_sync"),
                "--min-policy-version",
                str(args.min_updates),
                "--output-json",
                str(evidence_dir / "sync_artifacts.json"),
            ],
        ),
        (
            "final_checkpoint",
            [
                sys.executable,
                "-m",
                "fitmotn.cli.validate_checkpoint",
                str(rl_dir / "final_model"),
                "--require-exact",
                "rl",
                "--output-json",
                str(evidence_dir / "final_checkpoint.json"),
            ],
        ),
        (
            "timing",
            [
                sys.executable,
                "scripts/analyze_rl_timing.py",
                str(rl_jsonl),
                "--last-n",
                str(args.min_updates),
                "--json",
            ],
        ),
    ]
    failed = []
    for name, command in checks:
        code = _run_streaming(command, cwd=ROOT, log_path=evidence_dir / f"{name}.log")
        report["commands"].append({"name": name, "returncode": code})
        if code != 0:
            failed.append(name)
    report["finished_at"] = time.time()
    report["duration_sec"] = max(0.0, report["finished_at"] - report["started_at"])
    report["ok"] = not failed
    report["failed_checks"] = failed
    if failed:
        report["error"] = f"acceptance checks failed: {failed}"
    _write_json(evidence_dir / "stage5c_acceptance_summary.json", report)
    print(f"\nStage 5C acceptance {'PASSED' if report['ok'] else 'FAILED'}. Evidence: {evidence_dir}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
