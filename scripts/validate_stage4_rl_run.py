#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def validate_records(
    records: list[dict[str, Any]],
    *,
    min_updates: int,
    allow_fallback: bool,
    require_provenance: bool = True,
) -> dict[str, Any]:
    starts = [row for row in records if row.get("kind") == "run_start"]
    train = [row for row in records if row.get("kind") == "train"]
    update_rows = [row for row in train if row.get("did_update") is True or row.get("update_step") is not None]
    errors = []
    if not starts:
        errors.append("missing run_start record")
    initial_fingerprint = starts[-1].get("vllm_policy_fingerprint") if starts else None
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
        if require_provenance:
            for key in (
                "rollout_request_id",
                "rollout_prompt_fingerprint",
                "rollout_sampling_fingerprint",
                "vllm_policy_fingerprint",
            ):
                if not row.get(key):
                    errors.append(f"train[{idx}] is missing {key}")
            if row.get("vllm_engine_policy_verified") is not True:
                errors.append(f"train[{idx}] did not verify the loaded vLLM engine policy")
            if row.get("vllm_export_transaction_committed") is not True:
                errors.append(f"train[{idx}] has no committed export transaction evidence")
            update_step = int(row.get("update_step", 0) or 0)
            rollout_version = row.get("vllm_policy_version")
            post_update_version = row.get("vllm_post_update_policy_version")
            if rollout_version is not None and int(rollout_version) > max(0, update_step - 1):
                errors.append(
                    f"train[{idx}] rollout policy version {rollout_version} is from the future "
                    f"relative to update {update_step}"
                )
            if post_update_version is None or int(post_update_version) < update_step:
                errors.append(
                    f"train[{idx}] post-update vLLM policy version {post_update_version} "
                    f"is below update {update_step}"
                )
            post_fingerprint = row.get("vllm_post_update_policy_fingerprint")
            if not post_fingerprint:
                errors.append(f"train[{idx}] is missing vllm_post_update_policy_fingerprint")
            expected_rollout_fingerprint = (
                initial_fingerprint if idx == 0 else train[idx - 1].get("vllm_post_update_policy_fingerprint")
            )
            if expected_rollout_fingerprint and row.get("vllm_policy_fingerprint") != expected_rollout_fingerprint:
                errors.append(
                    f"train[{idx}] rollout policy fingerprint does not match the preceding committed sync"
                )
    request_ids = [str(row["rollout_request_id"]) for row in train if row.get("rollout_request_id")]
    if len(request_ids) != len(set(request_ids)):
        errors.append("rollout_request_id values are not unique")
    versions = sorted({int(row["vllm_policy_version"]) for row in train if row.get("vllm_policy_version") is not None})
    synced_versions = sorted(
        {
            int(row["vllm_post_update_policy_version"])
            for row in train
            if row.get("vllm_post_update_policy_version") is not None
        }
    )
    if synced_versions and synced_versions[-1] < int(min_updates):
        errors.append(
            f"latest post-update vLLM policy version {synced_versions[-1]} "
            f"is below required update {int(min_updates)}"
        )
    return {
        "ok": not errors,
        "errors": errors,
        "run_start_count": len(starts),
        "train_record_count": len(train),
        "policy_versions": versions,
        "post_update_policy_versions": synced_versions,
        "unique_rollout_request_count": len(set(request_ids)),
        "max_update_step": max((int(row.get("update_step", 0)) for row in train), default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage 4F/4G RL smoke-run JSONL")
    parser.add_argument("jsonl")
    parser.add_argument("--min-updates", type=int, default=2)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--allow-legacy-logs", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = validate_records(
        records,
        min_updates=args.min_updates,
        allow_fallback=args.allow_fallback,
        require_provenance=not args.allow_legacy_logs,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
