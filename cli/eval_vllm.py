from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..eval.vllm_runner import evaluate_with_vllm, inspect_vllm_compatibility


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_or_ckpt", type=str, required=True)
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--limit_per_task", type=int, default=32)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(config_json=args.config_json)
    output_json = Path(args.output_json).resolve() if args.output_json else Path(args.model_or_ckpt).resolve() / "fitmotn_eval_vllm.json"
    try:
        results = evaluate_with_vllm(args.model_or_ckpt, cfg, limit_per_task=args.limit_per_task)
    except NotImplementedError as exc:
        results = {
            "tasks": {},
            "summary": {},
            "capability": inspect_vllm_compatibility(args.model_or_ckpt),
            "error": str(exc),
        }
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        raise
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
