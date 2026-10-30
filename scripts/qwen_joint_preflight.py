"""Inspect a Qwen handoff machine; optionally probe GPU numerics, never train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "research/qwen_mote_joint_handoff_20261030"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_plan(machine):
    if machine not in {"pro4000", "v100_pair", "a800", "bw1000"}:
        raise ValueError("Unknown machine profile")
    return json.loads((HANDOFF / "plans" / f"{machine}.json").read_text(encoding="utf-8"))


def gpu_probe():
    import torch
    if not torch.cuda.is_available():
        return {"passed": False, "reason": "No available CUDA/HIP device"}
    device = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device)
    tests = []
    bf16 = bool(torch.cuda.is_bf16_supported())
    for dtype in [torch.float32, torch.float16] + ([torch.bfloat16] if bf16 else []):
        torch.manual_seed(17)
        x = torch.randn(64, 32, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.randn(32, 16, device="cuda", dtype=dtype, requires_grad=True)
        loss = (x @ w).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        tests.append({"dtype": str(dtype), "finite": bool(torch.isfinite(loss)
                      and torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()),
                      "nonzero_grad": bool(x.grad.abs().sum() > 0 and w.grad.abs().sum() > 0)})
    return {"passed": all(t["finite"] and t["nonzero_grad"] for t in tests),
            "scope": "tiny matrix forward/backward only; not Qwen/MoTE acceptance",
            "gpu": prop.name, "visible_device": device, "total_memory_bytes": prop.total_memory,
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None), "bf16_reported_supported": bf16,
            "tests": tests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True, choices=["pro4000", "v100_pair", "a800", "bw1000"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-gpu", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing environment record; use a new output path")
    plan = load_plan(args.machine)
    versions = {}
    for name in ["torch", "transformers", "accelerate", "datasets", "numpy", "pytest"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "machine": args.machine,
              "python": sys.version, "platform": platform.platform(), "versions": versions,
              "commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
              "dirty_files": git("status", "--short").splitlines(),
              "plan_status": plan["status"], "formal_launch_ready": False,
              "formal_launch_pending": ["reviewed four-router Qwen integration release",
                                        "frozen model/tokenizer/data", "common pilot acceptance"],
              "gpu_probe": {"state": "not_requested"}}
    if args.probe_gpu:
        try:
            report["gpu_probe"] = gpu_probe()
        except Exception as exc:
            report["gpu_probe"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents concurrent invocations overwriting a record.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"record": str(args.output), "formal_launch_ready": False,
                      "gpu_probe": report["gpu_probe"]}, ensure_ascii=False))
    return 1 if report["gpu_probe"].get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
