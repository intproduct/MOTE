from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as tc

from ..config import load_config
from ..eval.restore import restore_fitmotn_model
from ..eval.runner import run_eval_tasks
from ..runtime import load_causal_lm_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_or_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--eval_backend", type=str, default=None)
    parser.add_argument("--primary_eval_backend", type=str, default=None)
    parser.add_argument("--allow_backend_skip", action="store_true")
    return parser.parse_args()


def _load_model_and_tokenizer(target: Path, cfg, device: str):
    if (target / "fitmotn_state.pt").exists():
        return restore_fitmotn_model(target, device=device)
    return load_causal_lm_and_tokenizer(
        target,
        device=tc.device(device),
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=cfg.model.torch_dtype,
        use_cache=True,
    )


def main():
    args = parse_args()
    eval_overrides = {
        "backend_defaults": {
            "lm_eval": {"device": args.device},
            "evalscope": {"device": args.device},
        }
    }
    if args.eval_backend:
        eval_overrides["eval_backend"] = args.eval_backend
    if args.primary_eval_backend:
        eval_overrides["primary_eval_backend"] = args.primary_eval_backend
    cfg = load_config(config_json=args.config_json, overrides={"eval": eval_overrides})
    target = Path(args.model_or_ckpt).resolve()
    tasks = args.tasks or cfg.eval.final_tasks

    enabled_backends = str(cfg.eval.eval_backend)
    need_model = enabled_backends in {"lm_eval", "both"}
    model = tokenizer = None
    if need_model:
        model, tokenizer, _ = _load_model_and_tokenizer(target, cfg, args.device)

    results = run_eval_tasks(
        cfg,
        tasks=tasks,
        eval_name="eval_auto",
        eval_mode="final",
        model=model,
        tokenizer=tokenizer,
        model_or_path=target,
        allow_backend_skip=bool(args.allow_backend_skip),
    )
    output_json = Path(args.output_json).resolve() if args.output_json else target / "fitmotn_eval_auto.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
