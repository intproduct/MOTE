#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitmotn.config import load_config


def _records(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def validate(
    *,
    config_path: Path,
    run_path: Path | None,
    export_root: Path | None,
    min_updates: int,
    max_payload_ratio: float,
    max_sync_p95_sec: float | None,
) -> dict[str, Any]:
    cfg = load_config(config_json=str(config_path))
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "rollout_backend": "vllm",
        "vllm_execution_mode": "subprocess",
        "vllm_sync_strategy": "weight_transfer_nccl",
        "vllm_native_transfer_required_level": "update_only",
        "vllm_weight_transfer_scope": "trainable_patch",
        "trainable_mode": "patch_only",
    }
    for field, expected in required.items():
        actual = getattr(cfg.rl, field)
        if actual != expected:
            errors.append(f"rl.{field} must be {expected!r}, got {actual!r}")
    if int(cfg.rl.vllm_sync_every_updates) != 1:
        errors.append("acceptance requires rl.vllm_sync_every_updates=1")
    if int(cfg.rl.log_every) != 1:
        errors.append("acceptance requires rl.log_every=1")
    if len(list(cfg.rl.vllm_rollout_actors or [])) < 2:
        errors.append("three-GPU Stage 5E acceptance requires at least two TP1 rollout actors")
    if bool(cfg.rl.allow_stale_vllm_policy):
        errors.append("acceptance requires allow_stale_vllm_policy=false")
    if bool(cfg.rl.vllm_fallback_to_hf) or bool(cfg.rl.vllm_weight_transfer_fallback_to_export_reload):
        errors.append("acceptance forbids HF and export_reload fallbacks")

    run_evidence = None
    if run_path is not None:
        rows = _records(run_path)
        starts = [row for row in rows if row.get("kind") == "run_start"]
        trains = [row for row in rows if row.get("kind") == "train" and int(row.get("update_step", 0)) > 0]
        ends = [row for row in rows if row.get("kind") == "run_end"]
        if not starts:
            errors.append("run has no run_start record")
        else:
            start = starts[-1]
            if not bool(start.get("vllm_weight_transfer_bootstrap")):
                errors.append("run_start does not prove the one-time full-policy bootstrap")
            plan = dict(start.get("patch_transfer_plan") or {})
            if plan.get("transfer_scope") != "trainable_patch":
                errors.append(f"run_start has invalid patch transfer plan: {plan}")
        teardown = None
        if not ends:
            errors.append("run has no run_end record; final checkpoint/teardown did not complete")
        else:
            teardown = dict(ends[-1].get("rollout_teardown") or {})
            if teardown.get("ok") is not True:
                errors.append(f"run did not prove clean rollout teardown: {teardown}")
        max_update = max((int(row.get("update_step", 0)) for row in trains), default=0)
        if max_update < int(min_updates) or len(trains) < int(min_updates):
            errors.append(
                f"run has {len(trains)} native rows and reached update {max_update}; required {min_updates}"
            )

        plan_fingerprints: set[str] = set()
        actor_pids: dict[str, set[int]] = {}
        sync_times: list[float] = []
        payload_ratios: list[float] = []
        payload_bytes: list[int] = []
        full_bytes: list[int] = []
        for row in trains:
            update = int(row.get("update_step", 0))
            if not bool(row.get("vllm_weight_transfer_native_sync")):
                errors.append(f"update {update} did not use native sync")
            if row.get("vllm_weight_transfer_scope") != "trainable_patch":
                errors.append(f"update {update} did not use trainable_patch scope")
            if row.get("weight_transfer_adapter") != "nccl_update_only_subprocess":
                errors.append(f"update {update} used unexpected adapter {row.get('weight_transfer_adapter')!r}")
            if int(row.get("vllm_policy_version", -1)) != update or int(row.get("policy_lag_updates", -1)) != 0:
                errors.append(f"update {update} has stale or mismatched policy metadata")
            if float(row.get("vllm_engine_rebuild_sec", 0.0)) != 0.0:
                errors.append(f"update {update} rebuilt the Engine")
            if bool(row.get("vllm_weight_transfer_fallback_used")) or bool(row.get("vllm_fallback_used")):
                errors.append(f"update {update} used a fallback")
            if not bool(row.get("weight_transfer_commit_barrier")):
                errors.append(f"update {update} missed the all-actor commit barrier")

            tensor_count = int(row.get("weight_transfer_tensor_count", 0))
            selected_bytes = int(row.get("weight_transfer_bytes", 0))
            policy_bytes = int(row.get("full_policy_bytes", 0))
            ratio = float(row.get("weight_transfer_payload_ratio", 1.0))
            fingerprint = str(row.get("weight_transfer_plan_fingerprint") or "")
            if tensor_count <= 0 or selected_bytes <= 0 or policy_bytes <= 0:
                errors.append(f"update {update} has invalid patch payload metadata")
            if not (0.0 < ratio <= float(max_payload_ratio)):
                errors.append(
                    f"update {update} payload ratio={ratio:.6f} exceeds limit={max_payload_ratio:.6f}"
                )
            if not fingerprint:
                errors.append(f"update {update} has no transfer-plan fingerprint")
            else:
                plan_fingerprints.add(fingerprint)
            if bool(row.get("frozen_parameter_drift", True)):
                errors.append(f"update {update} reports frozen-parameter drift")

            results = dict(row.get("weight_transfer_actor_results") or {})
            if len(results) != int(row.get("weight_transfer_actor_count", -1)):
                errors.append(f"update {update} actor-result count mismatch")
            for name, result in results.items():
                if not bool(result.get("committed")):
                    errors.append(f"update {update} actor {name!r} did not commit")
                receive = dict(result.get("receive") or {})
                contract = dict(receive.get("received_weight_update_contract") or {})
                if contract.get("transfer_scope") != "trainable_patch":
                    errors.append(f"update {update} actor {name!r} received wrong scope: {contract}")
                if contract.get("transfer_plan_fingerprint") != fingerprint:
                    errors.append(f"update {update} actor {name!r} plan fingerprint mismatch")
                if int(contract.get("tensor_count", -1)) != tensor_count:
                    errors.append(f"update {update} actor {name!r} tensor-count mismatch")
                validation = dict(receive.get("rollout_facing_validation") or {})
                if validation.get("ok") is not True:
                    errors.append(f"update {update} actor {name!r} failed rollout validation")
                pid = receive.get("engine_process_id")
                if pid is not None:
                    actor_pids.setdefault(str(name), set()).add(int(pid))

            sync_sec = float(row.get("vllm_sync_sec", 0.0))
            if sync_sec > 0.0 and math.isfinite(sync_sec):
                sync_times.append(sync_sec)
            payload_ratios.append(ratio)
            payload_bytes.append(selected_bytes)
            full_bytes.append(policy_bytes)

        if len(plan_fingerprints) > 1:
            errors.append(f"transfer-plan fingerprint changed across updates: {sorted(plan_fingerprints)}")
        for name, pids in actor_pids.items():
            if len(pids) != 1:
                errors.append(f"actor {name!r} PID changed across updates: {sorted(pids)}")
        p95 = _percentile(sync_times, 0.95)
        if max_sync_p95_sec is not None and p95 is not None and p95 > float(max_sync_p95_sec):
            errors.append(f"patch sync p95={p95:.3f}s exceeds {max_sync_p95_sec:.3f}s")
        run_evidence = {
            "native_update_count": len(trains),
            "max_update": max_update,
            "transfer_plan_fingerprints": sorted(plan_fingerprints),
            "actor_pids": {name: sorted(pids) for name, pids in actor_pids.items()},
            "payload_bytes_mean": statistics.mean(payload_bytes) if payload_bytes else None,
            "full_policy_bytes_mean": statistics.mean(full_bytes) if full_bytes else None,
            "payload_ratio_mean": statistics.mean(payload_ratios) if payload_ratios else None,
            "sync_sec_p50": _percentile(sync_times, 0.50),
            "sync_sec_p95": p95,
            "rollout_teardown": teardown,
        }

    artifact_evidence = None
    if export_root is not None:
        hf_exports = sorted(path.name for path in export_root.glob("hf-policy-u*") if path.is_dir())
        raw_exports = sorted(path.name for path in export_root.glob("raw-policy-u*") if path.is_dir())
        if len(hf_exports) > 1 or len(raw_exports) > 1:
            errors.append(f"patch sync produced per-update exports: hf={hf_exports}, raw={raw_exports}")
        artifact_evidence = {"root": str(export_root), "hf_exports": hf_exports, "raw_exports": raw_exports}

    return {
        "ok": not errors,
        "stage": "5E-trainable-patch-native-sync",
        "errors": errors,
        "warnings": warnings,
        "config": str(config_path),
        "run": run_evidence,
        "artifacts": artifact_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 5E trainable-patch native NCCL sync")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rl-train-jsonl")
    parser.add_argument("--export-root")
    parser.add_argument("--min-updates", type=int, default=2)
    parser.add_argument("--max-payload-ratio", type=float, default=0.10)
    parser.add_argument("--max-sync-p95-sec", type=float)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    try:
        result = validate(
            config_path=Path(args.config),
            run_path=None if not args.rl_train_jsonl else Path(args.rl_train_jsonl),
            export_root=None if not args.export_root else Path(args.export_root),
            min_updates=int(args.min_updates),
            max_payload_ratio=float(args.max_payload_ratio),
            max_sync_p95_sec=args.max_sync_p95_sec,
        )
    except Exception as exc:
        result = {"ok": False, "stage": "5E-trainable-patch-native-sync", "errors": [str(exc)]}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
