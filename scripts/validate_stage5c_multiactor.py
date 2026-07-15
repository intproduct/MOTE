#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from MOTE.config import load_config
from MOTE.diagnostics.config_doctor import inspect_config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(row, dict):
            records.append(row)
    return records


def _validate_actor_runtime(
    actor: dict[str, Any],
    snapshot: dict[str, Any],
    errors: list[str],
) -> None:
    name = str(actor["name"])
    devices = list(actor["cuda_visible_devices"])
    tp_size = int(actor["tensor_parallel_size"])
    engine_info = dict(snapshot.get("engine") or {})
    resources = dict(engine_info.get("actor_resources") or {})
    observed_mask = [
        part.strip()
        for part in str(resources.get("cuda_visible_devices") or "").split(",")
        if part.strip()
    ]
    if observed_mask != devices:
        errors.append(f"actor {name!r} CUDA mask mismatch: configured={devices}, observed={observed_mask}")
    if int(resources.get("cuda_device_count", -1)) != len(devices):
        errors.append(
            f"actor {name!r} CUDA count mismatch: configured={len(devices)}, "
            f"observed={resources.get('cuda_device_count')}"
        )
    if not bool(resources.get("cuda_available", False)):
        errors.append(f"actor {name!r} did not report CUDA available")
    pid = resources.get("process_id")
    if pid is None or resources.get("process_group_id") != pid:
        errors.append(
            f"actor {name!r} is not an isolated process-group leader: "
            f"pid={pid}, pgid={resources.get('process_group_id')}"
        )
    engine_topology = dict(engine_info.get("engine_topology") or {})
    if not bool(engine_topology.get("tensor_parallel_verified", False)):
        errors.append(f"actor {name!r} runtime TP topology is unverified: {engine_topology}")
    elif int(engine_topology.get("observed_tensor_parallel_size", -1)) != tp_size:
        errors.append(
            f"actor {name!r} TP mismatch: configured={tp_size}, "
            f"observed={engine_topology.get('observed_tensor_parallel_size')}"
        )
    if not bool(snapshot.get("policy_verified", False)):
        errors.append(f"actor {name!r} policy descriptor was not verified")


def validate(config_path: Path, run_jsonl: Path | None, min_updates: int) -> dict[str, Any]:
    cfg = load_config(config_json=str(config_path))
    doctor = inspect_config(cfg)
    errors = list(doctor.get("errors") or [])
    warnings = list(doctor.get("warnings") or [])
    topology = dict((doctor.get("rl") or {}).get("resource_topology") or {})
    actors = list(topology.get("actors") or [])
    actor_names = [str(actor.get("name")) for actor in actors]
    if str(topology.get("execution_mode")) != "subprocess":
        errors.append("Stage 5C requires rl.vllm_execution_mode='subprocess'")
    if len(actors) < 2:
        errors.append(f"Stage 5C requires at least two rollout actors, observed {len(actors)}")
    if len(actor_names) != len(set(actor_names)):
        errors.append(f"rollout actor names are not unique: {actor_names}")
    engine_seeds = [actor.get("engine_seed") for actor in actors]
    if len(engine_seeds) != len(set(engine_seeds)):
        errors.append(f"rollout actor engine seeds are not unique: {engine_seeds}")

    run_evidence = None
    if run_jsonl is not None:
        records = _read_jsonl(run_jsonl)
        starts = [row for row in records if row.get("kind") == "run_start"]
        trains = [row for row in records if row.get("kind") == "train"]
        if not starts:
            errors.append("RL log has no run_start record")
            start = {}
        else:
            start = starts[-1]
        max_update = max((int(row.get("update_step", 0)) for row in trains), default=0)
        if max_update < int(min_updates):
            errors.append(f"RL log reached update {max_update}, required at least {int(min_updates)}")
        runtime_snapshots = dict(start.get("vllm_actor_resources") or {})
        for actor in actors:
            name = str(actor["name"])
            snapshot = dict(runtime_snapshots.get(name) or {})
            if not snapshot:
                errors.append(f"run_start is missing runtime handshake for actor {name!r}")
                continue
            _validate_actor_runtime(actor, snapshot, errors)
        if not bool(start.get("vllm_all_actors_policy_verified", False)):
            errors.append("run_start does not prove the all-actor policy barrier")

        checked_dispatches = 0
        for row in trains:
            update = int(row.get("update_step", 0))
            if int(row.get("policy_lag_updates", 0)) != 0:
                errors.append(f"update {update} has non-zero policy lag")
            if bool(row.get("vllm_fallback_used", False)):
                errors.append(f"update {update} used HF fallback")
            if not bool(row.get("vllm_engine_policy_verified", False)):
                errors.append(f"update {update} rollout policy was not verified across actors")
            if not bool(row.get("vllm_all_actors_policy_verified", False)):
                errors.append(f"update {update} post-update policy barrier was not verified")
            if int(row.get("vllm_rollout_actor_count", -1)) != len(actors):
                errors.append(
                    f"update {update} actor count mismatch: expected={len(actors)}, "
                    f"observed={row.get('vllm_rollout_actor_count')}"
                )
            dispatch = dict(row.get("vllm_actor_dispatch") or {})
            if not dispatch:
                errors.append(f"update {update} is missing multi-actor dispatch evidence")
                continue
            checked_dispatches += 1
            row_count = int(dispatch.get("row_count", -1))
            dispatch_actors = dict(dispatch.get("actors") or {})
            assigned = [
                int(index)
                for actor_dispatch in dispatch_actors.values()
                for index in list(actor_dispatch.get("row_indices") or [])
            ]
            if sorted(assigned) != list(range(row_count)) or len(assigned) != len(set(assigned)):
                errors.append(f"update {update} dispatch lost or duplicated rollout rows")
            expected_descriptor = {
                "policy_version": row.get("vllm_policy_version"),
                "policy_fingerprint": row.get("vllm_policy_fingerprint"),
                "export_dir": row.get("vllm_export_dir"),
            }
            for name, actor_dispatch in dispatch_actors.items():
                descriptor = dict(actor_dispatch.get("policy_descriptor") or {})
                if descriptor != expected_descriptor:
                    errors.append(
                        f"update {update} actor {name!r} used a different policy descriptor: "
                        f"expected={expected_descriptor}, observed={descriptor}"
                    )
                configured_actor = next((actor for actor in actors if actor["name"] == name), None)
                if configured_actor is None:
                    errors.append(f"update {update} contains unknown dispatch actor {name!r}")
                elif actor_dispatch.get("engine_seed") != configured_actor.get("engine_seed"):
                    errors.append(
                        f"update {update} actor {name!r} engine seed mismatch: "
                        f"expected={configured_actor.get('engine_seed')}, "
                        f"observed={actor_dispatch.get('engine_seed')}"
                    )
            expected_active = min(len(actors), max(1, int(dispatch.get("group_size", 1))))
            if int(dispatch.get("active_actor_count", 0)) < expected_active:
                errors.append(
                    f"update {update} used too few active actors: expected_at_least={expected_active}, "
                    f"observed={dispatch.get('active_actor_count')}"
                )
        if trains and checked_dispatches != len(trains):
            errors.append(
                f"only {checked_dispatches}/{len(trains)} train records contain valid dispatch evidence"
            )
        run_evidence = {
            "path": str(run_jsonl),
            "record_count": len(records),
            "train_record_count": len(trains),
            "max_update": max_update,
            "checked_dispatch_count": checked_dispatches,
            "actor_names": actor_names,
        }

    return {
        "ok": not errors,
        "stage": "5C",
        "errors": errors,
        "warnings": warnings,
        "config": str(config_path),
        "topology": topology,
        "doctor": doctor,
        "run_evidence": run_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 5C multi-actor vLLM topology and run evidence")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rl-train-jsonl")
    parser.add_argument("--min-updates", type=int, default=20)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    try:
        result = validate(
            Path(args.config),
            None if not args.rl_train_jsonl else Path(args.rl_train_jsonl),
            int(args.min_updates),
        )
    except Exception as exc:
        result = {"ok": False, "stage": "5C", "errors": [str(exc)], "warnings": []}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
