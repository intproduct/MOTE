from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


TIMING_FIELDS = [
    "tokenize_sec",
    "generate_sec",
    "reward_sec",
    "old_logprobs_sec",
    "ref_logprobs_sec",
    "new_logprobs_backward_sec",
    "total_micro_step_sec",
    "vllm_sync_sec",
    "vllm_engine_rebuild_sec",
    "vllm_fingerprint_sec",
    "vllm_checkpoint_save_sec",
    "vllm_export_convert_sec",
    "vllm_export_validation_sec",
    "vllm_export_commit_sec",
    "vllm_export_cleanup_sec",
    "vllm_generate_sec",
    "vllm_generated_tok_per_sec",
]

STAGE_FIELDS = [
    "tokenize_sec",
    "generate_sec",
    "reward_sec",
    "old_logprobs_sec",
    "ref_logprobs_sec",
    "new_logprobs_backward_sec",
    "vllm_sync_sec",
    "vllm_engine_rebuild_sec",
]


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(math.ceil((float(q) / 100.0) * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def summarize_values(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = list(values)
    if not vals:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(len(vals)),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "p90": _percentile(vals, 90),
        "p95": _percentile(vals, 95),
        "max": float(max(vals)),
    }


def load_timing_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(_as_number(record.get(field)) is not None for field in TIMING_FIELDS):
                records.append(record)
    return records


def analyze_timing(path: str | Path, *, last_n: int | None = None) -> dict[str, Any]:
    records = load_timing_records(path)
    if last_n is not None and int(last_n) > 0:
        records = records[-int(last_n) :]

    values_by_field: dict[str, list[float]] = {field: [] for field in TIMING_FIELDS}
    stage_ratios: dict[str, list[float]] = {field: [] for field in STAGE_FIELDS}
    for record in records:
        for field in TIMING_FIELDS:
            value = _as_number(record.get(field))
            if value is not None:
                values_by_field[field].append(value)
        total = _as_number(record.get("total_micro_step_sec"))
        if total is None or total <= 0.0:
            continue
        for field in STAGE_FIELDS:
            value = _as_number(record.get(field))
            if value is not None:
                stage_ratios[field].append(value / total)

    return {
        "path": str(Path(path).expanduser()),
        "record_count": int(len(records)),
        "timing": {field: summarize_values(values) for field, values in values_by_field.items()},
        "stage_total_mean_ratio": {
            field: (float(mean(values)) if values else None) for field, values in stage_ratios.items()
        },
    }


def _print_text(summary: dict[str, Any]) -> None:
    print(f"path: {summary['path']}")
    print(f"records: {summary['record_count']}")
    print("timing:")
    for field, stats in summary["timing"].items():
        if int(stats["count"]) == 0:
            continue
        print(
            f"  {field}: count={stats['count']} mean={stats['mean']:.6f} "
            f"median={stats['median']:.6f} p90={stats['p90']:.6f} "
            f"p95={stats['p95']:.6f} max={stats['max']:.6f}"
        )
    print("stage_total_mean_ratio:")
    for field, value in summary["stage_total_mean_ratio"].items():
        if value is not None:
            print(f"  {field}: {value:.6f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize RL timing fields from rl_train.jsonl")
    parser.add_argument("rl_train_jsonl", type=str)
    parser.add_argument("--last-n", type=int, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    summary = analyze_timing(args.rl_train_jsonl, last_n=args.last_n)
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        _print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
