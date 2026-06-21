from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as tc

from ..config import load_config
from ..eval.runner import run_eval_tasks
from ..eval.restore import restore_fitmotn_model
from ..runtime import load_causal_lm_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_or_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(
        config_json=args.config_json,
        overrides={
            "eval": {
                "eval_backend": "lm_eval",
                "primary_eval_backend": "lm_eval",
                "lm_eval_device": args.device,
                "backend_defaults": {"lm_eval": {"device": args.device}},
            }
        },
    )
    target = Path(args.model_or_ckpt).resolve()
    if (target / "fitmotn_state.pt").exists():
        model, tokenizer, _ = restore_fitmotn_model(target, device=args.device)
    else:
        model, tokenizer, _ = load_causal_lm_and_tokenizer(
            target,
            device=tc.device(args.device),
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=cfg.model.torch_dtype,
            use_cache=True,
        )
    tasks = args.tasks or cfg.eval.final_tasks
    results = run_eval_tasks(
        cfg,
        tasks=tasks,
        eval_name="eval_hf",
        eval_mode="final",
        model=model,
        tokenizer=tokenizer,
        model_or_path=target,
    )
    output_json = Path(args.output_json).resolve() if args.output_json else target / "fitmotn_eval_hf.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
