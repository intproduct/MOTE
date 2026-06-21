from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..config.schema import DEFAULT_GSM8K_GRPO_PROMPT_TEMPLATE
from ..rl.runtime import set_trainable_mode_for_rl
from ..train.rl_controller import build_response_mask, run_fitmotn_rl_training
from ..utils.paths import resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Compatibility wrapper for FitMoTN GSM8K GRPO training.")
    parser.add_argument("--model_or_ckpt", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--trainable_mode",
        type=str,
        choices=["all", "patch_only", "motn_only", "gate_only", "router_only", "global_only"],
        default="patch_only",
    )
    parser.add_argument("--no_ref_model", action="store_true", default=False)
    parser.add_argument("--debug_num_prompts", type=int, default=None)
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_GSM8K_GRPO_PROMPT_TEMPLATE)
    parser.add_argument("--sample_log_count", type=int, default=4)
    return parser.parse_args()


def _config_from_legacy_args(args):
    output_dir = Path(resolve_path(args.output_dir, key="cli.output_dir", source="override", allow_none=False))
    cfg = load_config(
        config_json=args.config_json,
        overrides={
            "model": {
                "model_path": args.model_or_ckpt,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
            },
            "output": {
                "root_dir": str(output_dir.parent),
                "run_name": output_dir.name,
                "overwrite_output_dir": True,
            },
            "rl": {
                "enabled": True,
                "mode": "gsm8k_grpo",
                "resume_from": args.model_or_ckpt,
                "train_json": args.train_json,
                "use_config_data": False,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "group_size": args.group_size,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "lr": args.lr,
                "eps_clip": args.eps_clip,
                "beta": args.beta,
                "grad_accum": args.gradient_accumulation_steps,
                "save_every_updates": args.save_steps,
                "log_every": args.logging_steps,
                "seed": args.seed,
                "trainable_mode": args.trainable_mode,
                "no_ref_model": args.no_ref_model,
                "debug_num_prompts": args.debug_num_prompts,
                "prompt_template": args.prompt_template,
                "sample_log_count": args.sample_log_count,
                "output_subdir": ".",
            },
        },
    )
    return cfg


def main():
    args = parse_args()
    cfg = _config_from_legacy_args(args)
    run_fitmotn_rl_training(cfg)


def _build_response_mask(sequences, *, response_start: int, eos_token_id=None, pad_token_id=None):
    return build_response_mask(
        sequences,
        response_start=response_start,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def _set_trainable_mode(model, mode: str):
    info = set_trainable_mode_for_rl(model, mode)
    return info["trainable_names"]


if __name__ == "__main__":
    main()
