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
    max_sync_p95_sec: float | None,
) -> dict[str, Any]:
    cfg = load_config(config_json=str(config_path))
    errors: list[str] = []
    warnings: list[str] = []
    if cfg.rl.rollout_backend != "vllm":
        errors.append("rl.rollout_backend must be 'vllm'")
    if cfg.rl.vllm_execution_mode != "subprocess":
        errors.append("rl.vllm_execution_mode must be 'subprocess'")
    if cfg.rl.vllm_sync_strategy != "weight_transfer_nccl":
        errors.append("rl.vllm_sync_strategy must be 'weight_transfer_nccl'")
    if cfg.rl.vllm_native_transfer_required_level != "update_only":
        errors.append("rl.vllm_native_transfer_required_level must be 'update_only'")
    if int(cfg.rl.vllm_sync_every_updates) != 1:
        errors.append("acceptance requires rl.vllm_sync_every_updates=1")
    if int(cfg.rl.log_every) != 1:
        errors.append("acceptance requires rl.log_every=1 so every native update has evidence")
    if len(list(cfg.rl.vllm_rollout_actors or [])) < 2:
        errors.append("three-GPU Stage 5D acceptance requires at least two TP1 rollout actors")
    if not bool(cfg.rl.vllm_weight_transfer_validate_coverage):
        errors.append("acceptance requires vllm_weight_transfer_validate_coverage=true")
    if not bool(cfg.rl.vllm_weight_transfer_validate_after_sync):
        errors.append("acceptance requires vllm_weight_transfer_validate_after_sync=true")
    if bool(cfg.rl.allow_stale_vllm_policy):
        errors.append("rl.allow_stale_vllm_policy must be false during acceptance")
    if bool(cfg.rl.vllm_fallback_to_hf):
        errors.append("rl.vllm_fallback_to_hf must be false during acceptance")
    if bool(cfg.rl.vllm_weight_transfer_fallback_to_export_reload):
        errors.append("native acceptance requires vllm_weight_transfer_fallback_to_export_reload=false")

    run_evidence = None
    if run_path is not None:
        rows = _records(run_path)
        starts = [row for row in rows if row.get("kind") == "run_start"]
        trains = [row for row in rows if row.get("kind") == "train"]
        if not starts:
            errors.append("run has no run_start record")
        else:
            start = starts[-1]
            if not bool(start.get("vllm_weight_transfer_bootstrap")):
                errors.append("run_start does not prove the one-time export bootstrap")
            if bool(start.get("vllm_weight_transfer_native_sync")):
                errors.append("run_start unexpectedly claims native sync instead of export bootstrap")
        max_update = max((int(row.get("update_step", 0)) for row in trains), default=0)
        if max_update < int(min_updates):
            errors.append(f"run reached update {max_update}; required {min_updates}")
        native_rows = [row for row in trains if int(row.get("update_step", 0)) > 0]
        if len(native_rows) < int(min_updates):
            errors.append(
                f"only {len(native_rows)} native update evidence rows were recorded; required {min_updates}"
            )
        sync_times: list[float] = []
        actor_pids: dict[str, set[int]] = {}
        actor_engine_child_pids: dict[str, set[tuple[int, ...]]] = {}
        actor_count = None
        for row in native_rows:
            update = int(row.get("update_step", 0))
            if not bool(row.get("vllm_weight_transfer_native_sync")):
                errors.append(f"update {update} did not use native sync")
            if row.get("weight_transfer_adapter") != "nccl_update_only_subprocess":
                errors.append(f"update {update} used unexpected adapter {row.get('weight_transfer_adapter')!r}")
            if int(row.get("vllm_policy_version", -1)) != update:
                errors.append(f"update {update} policy version mismatch: {row.get('vllm_policy_version')}")
            if int(row.get("policy_lag_updates", -1)) != 0:
                errors.append(f"update {update} has non-zero policy lag")
            if float(row.get("vllm_engine_rebuild_sec", 0.0)) != 0.0:
                errors.append(f"update {update} rebuilt the vLLM engine")
            if bool(row.get("vllm_weight_transfer_fallback_used", False)) or bool(row.get("vllm_fallback_used", False)):
                errors.append(f"update {update} used a fallback")
            if not bool(row.get("weight_transfer_commit_barrier")):
                errors.append(f"update {update} is missing the all-actor commit barrier")
            results = dict(row.get("weight_transfer_actor_results") or {})
            if actor_count is None:
                actor_count = int(row.get("weight_transfer_actor_count", len(results)))
            if len(results) != int(row.get("weight_transfer_actor_count", -1)):
                errors.append(f"update {update} actor-result count mismatch")
            for name, result in results.items():
                if not bool(result.get("committed")):
                    errors.append(f"update {update} actor {name!r} was not committed")
                receive = dict(result.get("receive") or {})
                pid = receive.get("engine_process_id")
                if pid is None:
                    errors.append(f"update {update} actor {name!r} has no engine PID evidence")
                else:
                    actor_pids.setdefault(str(name), set()).add(int(pid))
                resources = dict(receive.get("engine_resources") or {})
                child_ids = tuple(sorted(int(value) for value in resources.get("os_child_process_ids", [])))
                if child_ids:
                    actor_engine_child_pids.setdefault(str(name), set()).add(child_ids)
                validation = dict(receive.get("rollout_facing_validation") or {})
                if validation.get("ok") is not True:
                    errors.append(f"update {update} actor {name!r} failed rollout validation: {validation}")
            sync_sec = float(row.get("vllm_sync_sec", 0.0))
            if sync_sec <= 0.0 or not math.isfinite(sync_sec):
                errors.append(f"update {update} has invalid native sync time {sync_sec}")
            else:
                sync_times.append(sync_sec)
        for name, pids in actor_pids.items():
            if len(pids) != 1:
                errors.append(f"actor {name!r} engine PID changed across native updates: {sorted(pids)}")
        for name, pid_sets in actor_engine_child_pids.items():
            if len(pid_sets) != 1:
                errors.append(
                    f"actor {name!r} EngineCore/worker child PIDs changed across native updates: "
                    f"{sorted(pid_sets)}"
                )
        if native_rows and not actor_engine_child_pids:
            warnings.append(
                "no Linux /proc child-PID evidence was available; actor PID and zero rebuild metadata were checked"
            )
        p95 = _percentile(sync_times, 0.95)
        if max_sync_p95_sec is not None and p95 is not None and p95 > float(max_sync_p95_sec):
            errors.append(f"native sync p95={p95:.3f}s exceeds {max_sync_p95_sec:.3f}s")
        run_evidence = {
            "record_count": len(rows),
            "train_record_count": len(trains),
            "native_update_count": len(native_rows),
            "max_update": max_update,
            "actor_count": actor_count,
            "actor_pids": {name: sorted(pids) for name, pids in actor_pids.items()},
            "actor_engine_child_pids": {
                name: [list(pids) for pids in sorted(pid_sets)]
                for name, pid_sets in actor_engine_child_pids.items()
            },
            "sync_sec_mean": statistics.mean(sync_times) if sync_times else None,
            "sync_sec_p50": _percentile(sync_times, 0.50),
            "sync_sec_p95": p95,
        }

    artifact_evidence = None
    if export_root is not None:
        hf_exports = sorted(path.name for path in export_root.glob("hf-policy-u*") if path.is_dir())
        raw_exports = sorted(path.name for path in export_root.glob("raw-policy-u*") if path.is_dir())
        if len(hf_exports) > 1 or len(raw_exports) > 1:
            errors.append(
                "native run produced per-update export artifacts: "
                f"hf={hf_exports}, raw={raw_exports}"
            )
        artifact_evidence = {"root": str(export_root), "hf_exports": hf_exports, "raw_exports": raw_exports}

    return {
        "ok": not errors,
        "stage": "5D-full-weight-native-sync",
        "errors": errors,
        "warnings": warnings,
        "config": str(config_path),
        "run": run_evidence,
        "artifacts": artifact_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate persistent subprocess vLLM full-weight NCCL sync")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rl-train-jsonl")
    parser.add_argument("--export-root")
    parser.add_argument("--min-updates", type=int, default=20)
    parser.add_argument("--max-sync-p95-sec", type=float)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    try:
        result = validate(
            config_path=Path(args.config),
            run_path=None if not args.rl_train_jsonl else Path(args.rl_train_jsonl),
            export_root=None if not args.export_root else Path(args.export_root),
            min_updates=int(args.min_updates),
            max_sync_p95_sec=args.max_sync_p95_sec,
        )
    except Exception as exc:
        result = {"ok": False, "stage": "5D-full-weight-native-sync", "errors": [str(exc)]}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
