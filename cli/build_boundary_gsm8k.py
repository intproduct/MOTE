from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch

from ..audit import to_jsonable
from ..config import load_config
from ..rl.boundary import (
    RolloutAuditAccumulator,
    aggregate_spectrum,
    build_prompt_audit_record,
    is_boundary_prompt,
    reward_match_type,
)
from ..rl.data import load_gsm8k_rl_records
from ..rl.generation import tokenize_rollout_prompts
from ..rl.rewards_gsm8k import gsm8k_reward
from ..rl.rollout_backends import HFRolloutBackend, RolloutGenerationConfig
from ..rl.runtime import load_policy_for_rl
from ..rl.vllm_rollout import VLLMRolloutBackend
from ..train.rl_controller import build_rl_prompt_text, build_rollout_attention_and_response_mask


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Build GSM8K boundary data and an evaluation-only multi-rollout spectrum audit"
    )
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--rollout_backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--verified_traces_jsonl", type=str, required=True)
    parser.add_argument("--all_rollouts_jsonl", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--pass_k", type=int, nargs="+", default=None)
    parser.add_argument("--min_correct_rate", type=float, default=0.25)
    parser.add_argument("--max_correct_rate", type=float, default=0.75)
    parser.add_argument("--audit_boundary_min", type=float, default=0.25)
    parser.add_argument("--audit_boundary_max", type=float, default=0.75)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=25)
    return parser.parse_args(argv)


def _validate_args(args) -> list[int]:
    if int(args.num_rollouts) <= 0:
        raise ValueError(f"--num_rollouts must be > 0, got {args.num_rollouts}")
    if int(args.prompt_batch_size) <= 0:
        raise ValueError("--prompt_batch_size must be > 0")
    if args.max_prompts is not None and int(args.max_prompts) <= 0:
        raise ValueError(f"--max_prompts must be > 0 when set, got {args.max_prompts}")
    for low, high, label in (
        (args.min_correct_rate, args.max_correct_rate, "selection"),
        (args.audit_boundary_min, args.audit_boundary_max, "audit boundary"),
    ):
        if not 0.0 <= float(low) <= float(high) <= 1.0:
            raise ValueError(f"{label} rates must satisfy 0 <= min <= max <= 1")
    pass_k = list(args.pass_k or sorted({1, min(4, args.num_rollouts), min(8, args.num_rollouts), args.num_rollouts}))
    if any(int(k) <= 0 or int(k) > int(args.num_rollouts) for k in pass_k):
        raise ValueError("every --pass_k must be in [1, --num_rollouts]")
    return [int(k) for k in pass_k]


def _open_jsonl(path: str | Path | None):
    if path is None:
        return None
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("w", encoding="utf-8")


def _write_line(handle, value: Mapping[str, Any]) -> None:
    if handle is not None:
        handle.write(json.dumps(to_jsonable(dict(value)), ensure_ascii=False) + "\n")


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(to_jsonable(dict(value)), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tokenize_single(tokenizer, prompt: str, max_prompt_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenize_rollout_prompts(tokenizer, [prompt], max_prompt_tokens=max_prompt_tokens)
    return enc["input_ids"], enc["attention_mask"]


def _hf_samples(*, backend, model, tokenizer, prompt, num_rollouts, generation_config, prompt_seed, max_prompt_tokens):
    del prompt_seed
    prompts = [prompt] * int(num_rollouts)
    enc = tokenize_rollout_prompts(tokenizer, prompts, max_prompt_tokens=max_prompt_tokens)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    batch = backend.generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        update_step=0,
    )
    _, response_mask = build_rollout_attention_and_response_mask(
        batch.sequences,
        attention_mask,
        response_start=batch.response_start,
        original_seq_lens=batch.original_seq_lens,
        eos_token_id=tokenizer.eos_token_id,
    )
    texts = tokenizer.batch_decode(batch.sequences[:, batch.response_start:], skip_special_tokens=True)
    result = []
    for index, text in enumerate(texts):
        count = int(response_mask[index].sum().item())
        reached_limit = count >= int(generation_config.max_new_tokens)
        result.append({
            "text": text,
            "token_ids": batch.sequences[index, batch.response_start:batch.original_seq_lens[index]].detach().cpu().tolist(),
            "response_token_count": count,
            "finish_reason": "length" if reached_limit else "stop",
            "truncated": reached_limit,
        })
    return result


def _vllm_samples(*, backend, tokenizer, prompt, num_rollouts, generation_config, prompt_seed, prompt_index, max_prompt_tokens):
    input_ids, attention_mask = _tokenize_single(tokenizer, prompt, max_prompt_tokens)
    effective = input_ids[0][attention_mask[0].to(dtype=torch.bool)].tolist()
    raw = backend.generate_static_samples(
        prompt_token_ids=effective,
        num_samples=num_rollouts,
        max_new_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        seed=prompt_seed,
        prompt_index=prompt_index,
        eos_token_id=tokenizer.eos_token_id,
    )
    result = []
    for item in raw:
        token_ids = list(item["token_ids"])
        finish_reason = item.get("finish_reason")
        truncated = str(finish_reason).lower() in {"length", "max_tokens", "max_token"} or (
            finish_reason is None and len(token_ids) >= int(generation_config.max_new_tokens)
        )
        result.append({
            **item,
            "response_token_count": len(token_ids),
            "truncated": bool(truncated),
        })
    return result


def _copy_checkpoint_callback(source: Path):
    def save(output_dir: Path, update_step: int, checkpoint_name: str, extra: dict[str, Any]) -> Path:
        del update_step, checkpoint_name, extra
        shutil.copytree(source, output_dir, dirs_exist_ok=True)
        return output_dir
    return save


def _prepare_backend(cfg, args, output_root: Path):
    model, tokenizer, load_info = load_policy_for_rl(cfg, resume_from=cfg.rl.resume_from)
    model.eval()
    if args.rollout_backend == "hf":
        return HFRolloutBackend(), model, tokenizer, {"initialization": "hf_checkpoint_restore"}

    # Static evaluation always bootstraps by export/reload exactly once.  It
    # never joins the trainer's native NCCL weight-transfer session.
    cfg.rl.vllm_sync_strategy = "export_reload"
    cfg.rl.vllm_fallback_to_hf = False
    cfg.rl.vllm_export_root = str(output_root / ".vllm_static_exports")
    # The checkpoint metadata wins over a conflicting runtime JSON base path.
    cfg.model.model_path = str(load_info.base_model_path)
    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=output_root / ".vllm_static_runtime",
        save_policy_checkpoint=_copy_checkpoint_callback(Path(cfg.rl.resume_from).resolve()),
    )
    try:
        sync = backend.sync_policy(model=model, tokenizer=tokenizer, update_step=0, force=True)
    except Exception:
        backend.close()
        raise
    metadata = {
        "initialization": "static_export_reload_once",
        "export_dir": sync.export_dir,
        "policy_version": sync.policy_version,
        "checkpoint_metadata_authoritative": bool(load_info.metadata),
        "checkpoint_patch_cfg": load_info.patch_cfg,
    }
    return backend, model, tokenizer, metadata


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    pass_k = _validate_args(args)
    cfg = load_config(config_json=args.config_json)
    if args.resume_from:
        cfg.rl.resume_from = str(Path(args.resume_from).expanduser().resolve())
    if not getattr(cfg.rl, "resume_from", None):
        cfg.rl.resume_from = str(Path(cfg.model.model_path).expanduser().resolve())
    seed = int(args.seed if args.seed is not None else (getattr(cfg.rl, "seed", None) or getattr(cfg.train, "seed", 0)))
    # Preserve the original HF boundary builder's one-time global seeding.
    # vLLM additionally receives the required stable per-prompt request seed.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else cfg.rl.max_new_tokens)
    temperature = float(args.temperature if args.temperature is not None else cfg.rl.temperature)
    top_p = float(args.top_p if args.top_p is not None else cfg.rl.top_p)
    generation_config = RolloutGenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        rollout_micro_batch_size=int(getattr(cfg.rl, "rollout_micro_batch_size", 0)),
        rollout_use_cache=bool(getattr(cfg.rl, "rollout_use_cache", True)),
        rollout_inference_mode=True,
        seed=seed,
    )
    records = load_gsm8k_rl_records(cfg)
    if args.max_prompts is not None:
        records = records[: int(args.max_prompts)]

    output_root = Path(args.summary_json or args.output_jsonl).expanduser().resolve().parent
    backend = None
    model = None
    handles = []
    prompt_records: list[dict[str, Any]] = []
    rollout_accumulator = RolloutAuditAccumulator()
    verified_count = 0
    selected_count = 0
    init_metadata: dict[str, Any] = {}
    completed = False
    try:
        backend, model, tokenizer, init_metadata = _prepare_backend(cfg, args, output_root)
        selected_file = _open_jsonl(args.output_jsonl)
        verified_file = _open_jsonl(args.verified_traces_jsonl)
        all_file = _open_jsonl(args.all_rollouts_jsonl)
        handles = [handle for handle in (selected_file, verified_file, all_file) if handle is not None]
        for prompt_index, row in enumerate(records):
            prompt = build_rl_prompt_text(cfg, tokenizer, row["question"])
            prompt_seed = seed + prompt_index
            common = dict(
                backend=backend, tokenizer=tokenizer, prompt=prompt,
                num_rollouts=args.num_rollouts, generation_config=generation_config,
                prompt_seed=prompt_seed, max_prompt_tokens=int(getattr(cfg.rl, "rollout_max_prompt_tokens", 0)),
            )
            if args.rollout_backend == "hf":
                samples = _hf_samples(model=model, **common)
            else:
                samples = _vllm_samples(prompt_index=prompt_index, **common)
            rollout_rows = []
            for rollout_index, sample in enumerate(samples):
                reward, debug = gsm8k_reward(sample["text"], row["answer"])
                match_type = reward_match_type(debug, reward)
                rollout = {
                    "prompt_index": prompt_index,
                    "idx": row.get("idx"),
                    "rollout_index": rollout_index,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "response": sample["text"],
                    "response_token_count": int(sample["response_token_count"]),
                    "finish_reason": sample.get("finish_reason"),
                    "truncated": bool(sample["truncated"]),
                    "reward": float(reward),
                    "reward_debug": debug,
                    "reward_match_type": match_type,
                    "seed": prompt_seed,
                    "checkpoint": str(cfg.rl.resume_from),
                    "sampling": {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens},
                }
                rollout_rows.append(rollout)
                rollout_accumulator.add(rollout)
                _write_line(all_file, rollout)
            prompt_record = build_prompt_audit_record(
                row, rollout_rows, pass_k=pass_k,
                boundary_min=args.audit_boundary_min, boundary_max=args.audit_boundary_max,
                mgpo_cfg=cfg.rl,
            )
            prompt_records.append(prompt_record)
            if is_boundary_prompt(prompt_record["p_correct"], args.min_correct_rate, args.max_correct_rate):
                _write_line(selected_file, prompt_record)
                selected_count += 1
            for rollout in rollout_rows:
                if rollout["reward"] == 1.0:
                    _write_line(verified_file, {
                        **rollout,
                        "solution": rollout["response"],
                        "final_answer": str(rollout["reward_debug"].get("pred_answer") or ""),
                        "source": "boundary_rollout",
                        "p_correct": prompt_record["p_correct"],
                        "num_rollouts": prompt_record["num_rollouts"],
                        "match_quality_warning": (
                            "fallback-only correctness; inspect before distillation"
                            if rollout["reward_match_type"] == "fallback_last_number_only" else None
                        ),
                    })
                    verified_count += 1
            if args.progress_every > 0 and ((prompt_index + 1) % args.progress_every == 0 or prompt_index + 1 == len(records)):
                partial = aggregate_spectrum(prompt_records, rollout_accumulator, pass_k=pass_k, mgpo_enabled=cfg.rl.mgpo_enabled)
                spectrum = partial["spectrum"]
                print(
                    f"[SpectrumAudit] processed={prompt_index + 1}/{len(records)} "
                    f"sampled_pass1={spectrum['sampled_pass_at_1']:.4f} "
                    f"pass{pass_k[-1]}={spectrum[f'pass_at_{pass_k[-1]}']:.4f} "
                    f"mixed={spectrum['mixed_ratio']:.4f} all_wrong={spectrum['all_wrong_ratio']:.4f} "
                    f"all_correct={spectrum['all_correct_ratio']:.4f} "
                    f"truncated={partial['length_diagnostics']['truncated_rollout_rate']:.4f}",
                    flush=True,
                )
                for handle in handles:
                    handle.flush()
        completed = True
    finally:
        for handle in handles:
            handle.flush()
            handle.close()
        if backend is not None:
            backend.close()
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregates = aggregate_spectrum(
        prompt_records, rollout_accumulator, pass_k=pass_k, mgpo_enabled=bool(cfg.rl.mgpo_enabled)
    )
    summary = {
        "metadata": {
            "model_or_checkpoint": str(cfg.rl.resume_from),
            "dataset": "gsm8k",
            "split": "train",
            "rollout_backend": args.rollout_backend,
            "num_prompts": len(prompt_records),
            "num_rollouts_per_prompt": int(args.num_rollouts),
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "pass_k": pass_k,
            "selection_correct_rate_range": [args.min_correct_rate, args.max_correct_rate],
            "audit_boundary_range": [args.audit_boundary_min, args.audit_boundary_max],
            "prompt_batch_size": int(args.prompt_batch_size),
            **init_metadata,
        },
        **aggregates,
        "outputs": {
            "selected_prompt_count": selected_count,
            "verified_trace_count": verified_count,
            "output_jsonl": str(Path(args.output_jsonl).expanduser().resolve()),
            "verified_traces_jsonl": str(Path(args.verified_traces_jsonl).expanduser().resolve()),
            "all_rollouts_jsonl": None if args.all_rollouts_jsonl is None else str(Path(args.all_rollouts_jsonl).expanduser().resolve()),
        },
        "complete": completed,
    }
    summary_path = args.summary_json or str(Path(args.output_jsonl).with_suffix(".summary.json"))
    _atomic_json(summary_path, summary)
    print(json.dumps({"complete": True, "summary_json": str(Path(summary_path).resolve()), **summary["outputs"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
