from __future__ import annotations

import argparse
import json

from ..train.checkpoint_selector import select_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one complete FitMoTN checkpoint from a frozen candidate set.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-summary", required=True)
    parser.add_argument("--candidate-updates", required=True, help="Comma-separated update integers")
    parser.add_argument("--primary-task", default="mmlu")
    parser.add_argument("--secondary-task", default="gsm8k")
    parser.add_argument("--primary-floor", required=True, type=float)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--no-probe-restore", action="store_true")
    parser.add_argument("--output")
    parser.add_argument(
        "--task-contracts-json",
        required=True,
        help="JSON object containing frozen backend/protocol/fewshot/limit/primary_metric per selector task",
    )
    args = parser.parse_args()
    result = select_checkpoint(
        args.run_dir,
        args.eval_summary,
        candidate_updates=[int(value.strip()) for value in args.candidate_updates.split(",") if value.strip()],
        primary_task=args.primary_task,
        secondary_task=args.secondary_task,
        primary_floor=args.primary_floor,
        verify_hashes=args.verify_hashes,
        probe_restore=not args.no_probe_restore,
        output_path=args.output,
        task_contracts=json.loads(args.task_contracts_json),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
