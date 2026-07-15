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
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def validate(config_path: Path, run_jsonl: Path | None, min_updates: int) -> dict[str, Any]:
    cfg = load_config(config_json=str(config_path))
    doctor = inspect_config(cfg)
    errors = list(doctor.get("errors") or [])
    warnings = list(doctor.get("warnings") or [])
    topology = dict((doctor.get("rl") or {}).get("resource_topology") or {})
    actors = list(topology.get("actors") or [])
    if len(actors) != 1:
        errors.append(f"Stage 5B requires exactly one rollout actor, observed {len(actors)}")
    actor = dict(actors[0]) if actors else {}
    actor_devices = list(actor.get("cuda_visible_devices") or [])
    tp_size = int(actor.get("tensor_parallel_size", 0))
    if str(topology.get("execution_mode")) != "subprocess":
        errors.append("Stage 5B requires rl.vllm_execution_mode='subprocess'")
    if not actor_devices:
        errors.append("Stage 5B requires explicit rl.vllm_actor_cuda_visible_devices")
    if len(actor_devices) != tp_size:
        errors.append(f"actor device count {len(actor_devices)} does not match TP size {tp_size}")

    run_evidence: dict[str, Any] | None = None
    if run_jsonl is not None:
        records = _read_jsonl(run_jsonl)
        starts = [row for row in records if row.get("kind") == "run_start"]
        trains = [row for row in records if row.get("kind") == "train"]
        if not starts:
            errors.append("RL log has no run_start record")
        max_update = max((int(row.get("update_step", 0)) for row in trains), default=0)
        if max_update < int(min_updates):
            errors.append(f"RL log reached update {max_update}, required at least {int(min_updates)}")
        bad_lag = [row.get("update_step") for row in trains if int(row.get("policy_lag_updates", 0)) != 0]
        fallbacks = [row.get("update_step") for row in trains if bool(row.get("vllm_fallback_used", False))]
        unverified_policy = [
            row.get("update_step")
            for row in trains
            if not bool(row.get("vllm_engine_policy_verified", False))
        ]
        if bad_lag:
            errors.append(f"non-zero policy lag at updates {bad_lag[:10]}")
        if fallbacks:
            errors.append(f"HF fallback used at updates {fallbacks[:10]}")
        if unverified_policy:
            errors.append(f"vLLM engine policy was not verified at updates {unverified_policy[:10]}")
        actor_resources = None
        if starts:
            actor_resources = starts[-1].get("vllm_actor_resources")
        if not actor_resources:
            errors.append("run_start does not contain vllm_actor_resources handshake evidence")
        else:
            resources_payload = dict(actor_resources)
            actor_snapshot = (
                resources_payload
                if "engine" in resources_payload
                else dict(resources_payload.get(str(actor.get("name", "actor_0"))) or {})
            )
            engine_info = dict(actor_snapshot.get("engine") or {})
            observed = dict(engine_info.get("actor_resources") or {})
            observed_count = int(observed.get("cuda_device_count", -1))
            if observed_count != len(actor_devices):
                errors.append(
                    f"actor observed {observed_count} CUDA devices but configuration assigned {len(actor_devices)}"
                )
            if not bool(observed.get("cuda_available", False)):
                errors.append(f"actor did not report an available CUDA runtime: {observed}")
            observed_mask = [
                part.strip()
                for part in str(observed.get("cuda_visible_devices") or "").split(",")
                if part.strip()
            ]
            if observed_mask != actor_devices:
                errors.append(
                    f"actor CUDA_VISIBLE_DEVICES {observed_mask} does not match assigned devices {actor_devices}"
                )
            actor_pid = observed.get("process_id")
            actor_pgid = observed.get("process_group_id")
            if actor_pid is None or actor_pgid != actor_pid:
                errors.append(
                    f"actor is not the leader of its isolated cleanup process group: pid={actor_pid}, pgid={actor_pgid}"
                )
            engine_topology = dict(engine_info.get("engine_topology") or {})
            observed_tp = engine_topology.get("observed_tensor_parallel_size")
            if observed_tp is None or not bool(engine_topology.get("tensor_parallel_verified", False)):
                errors.append(f"actor could not verify runtime engine TP topology: {engine_topology}")
            elif int(observed_tp) != tp_size:
                errors.append(f"actor engine TP size {observed_tp} does not match configured TP size {tp_size}")
        run_evidence = {
            "path": str(run_jsonl),
            "record_count": len(records),
            "train_record_count": len(trains),
            "max_update": max_update,
            "actor_resources": actor_resources,
        }

    return {
        "ok": not errors,
        "stage": "5B",
        "errors": errors,
        "warnings": warnings,
        "config": str(config_path),
        "topology": topology,
        "doctor": doctor,
        "run_evidence": run_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 5B isolated multi-GPU vLLM topology and run evidence")
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
        result = {"ok": False, "stage": "5B", "errors": [str(exc)], "warnings": []}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
