#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def validate_records(records: list[dict[str, Any]], *, min_updates: int, allow_fallback: bool) -> dict[str, Any]:
    starts = [row for row in records if row.get("kind") == "run_start"]
    train = [row for row in records if row.get("kind") == "train"]
    update_rows = [row for row in train if row.get("did_update") is True or row.get("update_step") is not None]
    errors = []
    if not starts:
        errors.append("missing run_start record")
    if len({int(row.get("update_step", 0)) for row in update_rows}) < int(min_updates):
        errors.append(f"fewer than {int(min_updates)} optimizer updates were recorded")
    for idx, row in enumerate(train):
        for key in ("loss", "policy_loss"):
            value = row.get(key)
            if value is not None and not math.isfinite(float(value)):
                errors.append(f"train[{idx}].{key} is not finite: {value}")
        if int(row.get("policy_lag_updates", 0) or 0) != 0:
            errors.append(f"train[{idx}] has policy_lag_updates={row.get('policy_lag_updates')}")
        if not allow_fallback and bool(row.get("vllm_fallback_used", False)):
            errors.append(f"train[{idx}] used HF fallback: {row.get('vllm_error')}")
    versions = sorted({int(row["vllm_policy_version"]) for row in train if row.get("vllm_policy_version") is not None})
    if versions and versions[-1] < int(min_updates):
        errors.append(f"latest vLLM policy version {versions[-1]} is below required update {int(min_updates)}")
    return {
        "ok": not errors,
        "errors": errors,
        "run_start_count": len(starts),
        "train_record_count": len(train),
        "policy_versions": versions,
        "max_update_step": max((int(row.get("update_step", 0)) for row in train), default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage 4D RL smoke-run JSONL")
    parser.add_argument("jsonl")
    parser.add_argument("--min-updates", type=int, default=2)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = validate_records(records, min_updates=args.min_updates, allow_fallback=args.allow_fallback)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
