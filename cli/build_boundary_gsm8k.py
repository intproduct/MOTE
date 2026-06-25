from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch

from ..audit import to_jsonable
from ..config import load_config
from ..rl.boundary import (
    build_boundary_record,
    build_verified_trace_record,
    is_boundary_prompt,
    summarize_rollout_rewards,
)
from ..rl.data import load_gsm8k_rl_records
from ..rl.generation import (
    is_cache_compat_generation_error,
    rollout_generation_state,
    rollout_grad_context,
)
from ..rl.rewards_gsm8k import gsm8k_reward
from ..rl.runtime import load_policy_for_rl
from ..train.rl_controller import build_response_mask, build_rl_prompt_text


def parse_args():
    parser = argparse.ArgumentParser(description="Build GSM8K boundary prompts and verified traces from rollouts")
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--verified_traces_jsonl", type=str, required=True)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--min_correct_rate", type=float, default=0.25)
    parser.add_argument("--max_correct_rate", type=float, default=0.75)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if int(args.num_rollouts) <= 0:
        raise ValueError(f"--num_rollouts must be > 0, got {args.num_rollouts}")
    if args.max_prompts is not None and int(args.max_prompts) <= 0:
        raise ValueError(f"--max_prompts must be > 0 when set, got {args.max_prompts}")
    if float(args.min_correct_rate) > float(args.max_correct_rate):
        raise ValueError("--min_correct_rate must be <= --max_correct_rate")

    cfg = load_config(config_json=args.config_json)
    if args.resume_from:
        cfg.rl.resume_from = str(Path(args.resume_from).expanduser().resolve())

    seed = args.seed
    if seed is None:
        seed = getattr(cfg.rl, "seed", None)
    if seed is None:
        seed = int(getattr(cfg.train, "seed", 0))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    records = load_gsm8k_rl_records(cfg)
    if args.max_prompts is not None:
        records = records[: int(args.max_prompts)]

    model, tokenizer, _ = load_policy_for_rl(cfg, resume_from=getattr(cfg.rl, "resume_from", None))
    model.eval()

    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else getattr(cfg.rl, "max_new_tokens", 256))
    temperature = float(args.temperature if args.temperature is not None else getattr(cfg.rl, "temperature", 0.7))
    top_p = float(args.top_p if args.top_p is not None else getattr(cfg.rl, "top_p", 0.95))

    boundary_records = []
    verified_trace_records = []
    all_wrong_count = 0
    all_correct_count = 0

    for row in records:
        prompt = build_rl_prompt_text(cfg, tokenizer, row["question"])
        rollout_prompts = [prompt for _ in range(int(args.num_rollouts))]
        enc = tokenizer(rollout_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        response_start = int(input_ids.shape[1])
        requested_use_cache = bool(getattr(cfg.rl, "rollout_use_cache", True))
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        try:
            with rollout_grad_context(bool(getattr(cfg.rl, "rollout_inference_mode", True))):
                with rollout_generation_state(model, use_cache=requested_use_cache):
                    generated = model.generate(use_cache=requested_use_cache, **generate_kwargs)
        except Exception as exc:
            if requested_use_cache and is_cache_compat_generation_error(exc):
                with rollout_grad_context(bool(getattr(cfg.rl, "rollout_inference_mode", True))):
                    with rollout_generation_state(model, use_cache=False):
                        generated = model.generate(use_cache=False, **generate_kwargs)
            else:
                raise
        response_mask = build_response_mask(
            generated,
            response_start=response_start,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ).to(generated.device)

        generated_texts = tokenizer.batch_decode(generated[:, response_start:], skip_special_tokens=True)
        rewards = []
        reward_debugs = []
        for generated_text in generated_texts:
            reward, debug = gsm8k_reward(generated_text, row["answer"])
            rewards.append(float(reward))
            reward_debugs.append(debug)
        response_lens = response_mask.sum(dim=1).detach().cpu().tolist()
        summary = summarize_rollout_rewards(rewards, reward_debugs, response_lens)
        if int(summary["num_correct"]) == 0:
            all_wrong_count += 1
        if int(summary["num_correct"]) == int(summary["num_rollouts"]):
            all_correct_count += 1
        if not is_boundary_prompt(summary["p_correct"], args.min_correct_rate, args.max_correct_rate):
            continue

        boundary_records.append(build_boundary_record(row, summary))
        for generated_text, reward, reward_debug in zip(generated_texts, rewards, reward_debugs):
            if float(reward) == 1.0:
                verified_trace_records.append(build_verified_trace_record(row, generated_text, reward_debug, summary))

    _write_jsonl(args.output_jsonl, boundary_records)
    _write_jsonl(args.verified_traces_jsonl, verified_trace_records)
    print(
        json.dumps(
            {
                "num_prompts": int(len(records)),
                "all_wrong_count": int(all_wrong_count),
                "all_correct_count": int(all_correct_count),
                "boundary_count": int(len(boundary_records)),
                "verified_trace_count": int(len(verified_trace_records)),
                "output_jsonl": str(Path(args.output_jsonl).expanduser().resolve()),
                "verified_traces_jsonl": str(Path(args.verified_traces_jsonl).expanduser().resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
