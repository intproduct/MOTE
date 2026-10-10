from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..checkpointing import get_restore_state_dict, load_fitmotn_metadata, validate_checkpoint


SELECTOR_FORMAT = "fitmotn_checkpoint_selection_v1"


def _task_score(
    result: Mapping[str, Any],
    task: str,
    expected_contract: Mapping[str, Any],
) -> float:
    tasks = result.get("tasks")
    if not isinstance(tasks, Mapping) or not isinstance(tasks.get(task), Mapping):
        raise ValueError(f"missing task result: {task}")
    task_result = tasks[task]
    for field in ("backend", "protocol", "fewshot", "limit", "primary_metric"):
        expected = expected_contract.get(field)
        if field not in task_result:
            raise ValueError(f"task {task} protocol contract field is missing: {field}")
        actual = task_result.get(field)
        if actual != expected:
            raise ValueError(
                f"task {task} protocol contract mismatch for {field}: "
                f"expected {expected!r}, got {actual!r}"
            )
    for field in (
        "warnings",
        "ignored_runtime_args",
        "unsupported_runtime_args",
        "unsupported_generation_args",
    ):
        if field not in task_result:
            raise ValueError(f"task {task} audit field is missing: {field}")
        value = task_result.get(field)
        if value != []:
            raise ValueError(f"task {task} has non-empty {field}: {value!r}")
    value = task_result.get("primary_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"task {task} primary_score is not numeric: {value!r}")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"task {task} primary_score is not finite: {value!r}")
    return score


def _mid_evals_by_update(eval_summary: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for entry in eval_summary.get("mid_evals", []) or []:
        if not isinstance(entry, Mapping):
            continue
        try:
            update = int(entry["update"])
        except (KeyError, TypeError, ValueError):
            continue
        if update in indexed:
            raise ValueError(f"duplicate mid-evaluation update: {update}")
        indexed[update] = entry
    return indexed


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_checkpoint_selection(path: str | Path, payload: Mapping[str, Any]) -> None:
    _write_json_atomic(Path(path).expanduser().resolve(), payload)


def select_checkpoint(
    run_dir: str | Path,
    eval_summary: Mapping[str, Any] | str | Path,
    *,
    candidate_updates: Iterable[int],
    primary_task: str = "mmlu",
    secondary_task: str = "gsm8k",
    primary_floor: float,
    verify_hashes: bool = False,
    probe_restore: bool = True,
    output_path: str | Path | None = None,
    config_audit: Mapping[str, Any] | None = None,
    task_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select one committed checkpoint using frozen lexicographic rules."""
    root = Path(run_dir).expanduser().resolve()
    if isinstance(eval_summary, Mapping):
        summary = dict(eval_summary)
    else:
        summary = json.loads(Path(eval_summary).expanduser().resolve().read_text(encoding="utf-8"))
    updates = sorted({int(value) for value in candidate_updates})
    if not updates or any(value <= 0 for value in updates):
        raise ValueError("candidate_updates must contain positive integers")
    floor = float(primary_floor)
    if not math.isfinite(floor):
        raise ValueError("primary_floor must be finite")
    indexed = _mid_evals_by_update(summary)
    required_tasks = (str(primary_task), str(secondary_task))
    supplied_contracts = task_contracts or {}
    contracts = {task: dict(supplied_contracts.get(task) or {}) for task in required_tasks}
    required_contract_fields = {"backend", "protocol", "fewshot", "limit", "primary_metric"}
    for task, contract in contracts.items():
        missing = sorted(required_contract_fields - set(contract))
        if missing:
            raise ValueError(f"task {task} selector contract missing fields: {missing}")
    records: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for update in updates:
        checkpoint_dir = root / "checkpoints" / f"checkpoint-{update}"
        record: dict[str, Any] = {
            "update": update,
            "checkpoint_dir": str(checkpoint_dir),
            "accepted": False,
            "primary_score": None,
            "secondary_score": None,
            "rejection_reasons": [],
        }
        validation = validate_checkpoint(checkpoint_dir, verify_hashes=bool(verify_hashes))
        if not validation["ok"]:
            record["rejection_reasons"].extend(
                f"checkpoint_invalid: {message}" for message in validation["errors"]
            )
        entry = indexed.get(update)
        if entry is None:
            record["rejection_reasons"].append("missing_mid_evaluation")
        else:
            result = entry.get("result")
            if not isinstance(result, Mapping):
                record["rejection_reasons"].append("invalid_mid_evaluation_result")
            else:
                expected_backend = contracts[str(primary_task)]["backend"]
                if result.get("primary_backend") != expected_backend:
                    record["rejection_reasons"].append(
                        f"mid_evaluation_backend_mismatch: expected {expected_backend!r}, "
                        f"got {result.get('primary_backend')!r}"
                    )
                if result.get("eval_mode") != "mid":
                    record["rejection_reasons"].append(
                        f"mid_evaluation_mode_mismatch: expected 'mid', got {result.get('eval_mode')!r}"
                    )
                if result.get("warnings") != []:
                    record["rejection_reasons"].append(
                        f"mid_evaluation_has_warnings: {result.get('warnings')!r}"
                    )
                for label, task in (("primary", primary_task), ("secondary", secondary_task)):
                    try:
                        record[f"{label}_score"] = _task_score(result, task, contracts[str(task)])
                    except ValueError as exc:
                        record["rejection_reasons"].append(str(exc))
        if record["primary_score"] is not None and record["primary_score"] < floor:
            record["rejection_reasons"].append(
                f"primary_score_below_floor: {record['primary_score']} < {floor}"
            )
        if not record["rejection_reasons"] and probe_restore:
            try:
                metadata = load_fitmotn_metadata(checkpoint_dir)
                state_dict = get_restore_state_dict(metadata)
                if not isinstance(state_dict, Mapping) or not state_dict:
                    raise ValueError("restore state dict is empty")
                del state_dict, metadata
            except Exception as exc:
                record["rejection_reasons"].append(f"restore_probe_failed: {type(exc).__name__}: {exc}")
        if not record["rejection_reasons"]:
            record["accepted"] = True
            accepted.append(record)
        records.append(record)
    if not accepted:
        raise RuntimeError(
            "checkpoint selector found no valid candidate; audit="
            + json.dumps(records, ensure_ascii=False, sort_keys=True)
        )
    selected = sorted(
        accepted,
        key=lambda item: (-item["primary_score"], -item["secondary_score"], item["update"]),
    )[0]
    payload = {
        "format": SELECTOR_FORMAT,
        "rule": {
            "candidate_updates": updates,
            "primary_task": str(primary_task),
            "secondary_task": str(secondary_task),
            "primary_floor": floor,
            "ordering": ["primary_score_desc", "secondary_score_desc", "update_asc"],
            "verify_hashes": bool(verify_hashes),
            "probe_restore": bool(probe_restore),
            "task_contracts": contracts,
        },
        "config_audit": dict(config_audit or {}),
        "selected_update": int(selected["update"]),
        "selected_checkpoint": selected["checkpoint_dir"],
        "candidates": records,
    }
    target = Path(output_path).expanduser().resolve() if output_path else root / "selected_checkpoint.json"
    _write_json_atomic(target, payload)
    payload["selection_record"] = str(target)
    return payload
