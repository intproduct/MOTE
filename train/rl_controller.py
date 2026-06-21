from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..audit import build_environment_snapshot, json_dump, jsonl_append, to_jsonable
from ..chat_formatting import build_chat_prompt_text_for_generation, tokenizer_supports_chat_template
from ..checkpointing import extract_patch_state_dict, save_fitmotn_metadata
from ..eval.runner import run_eval_tasks
from ..rl.data import format_rl_prompt, load_gsm8k_rl_records
from ..rl.grpo import compute_group_advantages, grpo_loss
from ..rl.logprobs import gather_response_logprobs
from ..rl.rewards_gsm8k import gsm8k_reward
from ..rl.runtime import (
    RLPolicyLoadInfo,
    build_trainable_summary,
    collect_train_forward_router_usage,
    disable_runtime_usage_tracking,
    dtype_name_from_load_info,
    enable_runtime_usage_tracking,
    load_policy_for_rl,
    load_reference_for_rl,
    log_trainable_summary,
    reset_runtime_usage_buffers,
    set_trainable_mode_for_rl,
)
from ..utils.paths import assert_no_unsafe_paths


def _cuda_memory_snapshot() -> Dict[str, Optional[float]]:
    if not torch.cuda.is_available():
        return {
            "memory_allocated_gib": None,
            "memory_reserved_gib": None,
            "memory_max_allocated_gib": None,
            "memory_max_reserved_gib": None,
        }
    gib = 1024.0 ** 3
    return {
        "memory_allocated_gib": float(torch.cuda.memory_allocated() / gib),
        "memory_reserved_gib": float(torch.cuda.memory_reserved() / gib),
        "memory_max_allocated_gib": float(torch.cuda.max_memory_allocated() / gib),
        "memory_max_reserved_gib": float(torch.cuda.max_memory_reserved() / gib),
    }


def log_cuda_memory(
    logger: logging.Logger,
    tag: str,
    update_step: Optional[int] = None,
    micro_step: Optional[int] = None,
    reset_peak: bool = False,
) -> None:
    if not torch.cuda.is_available():
        return
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    snap = _cuda_memory_snapshot()
    device = torch.cuda.current_device()
    logger.info(
        "[RLMem] tag=%s update=%s micro=%s device=%s allocated=%.3fGiB reserved=%.3fGiB "
        "max_allocated=%.3fGiB max_reserved=%.3fGiB",
        tag,
        "None" if update_step is None else int(update_step),
        "None" if micro_step is None else int(micro_step),
        torch.cuda.get_device_name(device),
        snap["memory_allocated_gib"],
        snap["memory_reserved_gib"],
        snap["memory_max_allocated_gib"],
        snap["memory_max_reserved_gib"],
    )


def _should_log_memory(fit_cfg, update_step: Optional[int]) -> bool:
    every = int(getattr(fit_cfg.rl, "log_memory_every", 0))
    if every <= 0:
        return False
    if update_step is None:
        return True
    return int(update_step) % every == 0


def _maybe_log_cuda_memory(
    fit_cfg,
    logger: logging.Logger,
    tag: str,
    update_step: Optional[int] = None,
    micro_step: Optional[int] = None,
    reset_peak: bool = False,
) -> None:
    if _should_log_memory(fit_cfg, update_step):
        log_cuda_memory(logger, tag, update_step=update_step, micro_step=micro_step, reset_peak=reset_peak)


def _flatten_response_advantages(advantages: torch.Tensor, total_n: int) -> torch.Tensor:
    adv = advantages.reshape(-1) if advantages.ndim == 2 else advantages
    if adv.ndim != 1 or int(adv.numel()) != int(total_n):
        raise ValueError(f"advantages must flatten to [{total_n}], got shape={tuple(advantages.shape)}")
    return adv


def iter_response_chunks(total_n: int, micro_batch_size: int) -> List[Tuple[int, int]]:
    total_n = int(total_n)
    mb = int(micro_batch_size)
    if total_n <= 0:
        return []
    if mb <= 0 or mb >= total_n:
        return [(0, total_n)]
    return [(start, min(start + mb, total_n)) for start in range(0, total_n, mb)]


def compute_old_logprobs_microbatched(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    micro_batch_size: int,
    response_start: Optional[int] = None,
) -> torch.Tensor:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start, end in iter_response_chunks(int(input_ids.shape[0]), int(micro_batch_size)):
            lp = gather_response_logprobs(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                response_mask[start:end],
                response_start=response_start,
            )
            chunks.append(lp.detach())
            del lp
    return torch.cat(chunks, dim=0)


def resolve_amp_dtype(model) -> Optional[torch.dtype]:
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        return None
    if dtype is torch.float16:
        return torch.float16
    if dtype is torch.bfloat16:
        return torch.bfloat16
    return None


def dtype_name_or_none(dtype: Optional[torch.dtype]) -> Optional[str]:
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def compute_new_logprobs_loss_microbatched(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: Optional[torch.Tensor],
    advantages: torch.Tensor,
    micro_batch_size: int,
    grad_accum: int,
    eps_clip: float,
    beta: float,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype] = None,
    response_start: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    fit_cfg=None,
    update_step: Optional[int] = None,
    micro_step: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    total_n = int(input_ids.shape[0])
    flat_advantages = _flatten_response_advantages(advantages, total_n)
    weighted_loss_total: Optional[torch.Tensor] = None
    metric_sums: Dict[str, torch.Tensor] = {}
    metric_maxes: Dict[str, torch.Tensor] = {}
    autocast_enabled = bool(use_amp and input_ids.device.type == "cuda" and amp_dtype is not None)

    for chunk_idx, (start, end) in enumerate(iter_response_chunks(total_n, int(micro_batch_size))):
        chunk_n = int(end - start)
        weight = float(chunk_n) / float(total_n)
        with torch.autocast(device_type="cuda", dtype=amp_dtype or torch.float32, enabled=autocast_enabled):
            new_logprobs_chunk = gather_response_logprobs(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                response_mask[start:end],
                response_start=response_start,
            )
        ref_chunk = ref_logprobs[start:end] if ref_logprobs is not None else None
        loss_chunk, metrics_chunk = grpo_loss(
            new_logprobs_chunk,
            old_logprobs[start:end],
            ref_chunk,
            flat_advantages[start:end],
            response_mask[start:end],
            eps_clip=float(eps_clip),
            beta=float(beta) if ref_chunk is not None else 0.0,
        )
        weighted_loss = loss_chunk * weight
        scaled_loss = weighted_loss / int(grad_accum)
        if not torch.isfinite(scaled_loss):
            value = float(scaled_loss.detach().cpu())
            raise FloatingPointError(f"Non-finite GRPO loss at micro_step={micro_step}: {value}")
        scaled_loss.backward()

        weighted_detached = weighted_loss.detach()
        weighted_loss_total = weighted_detached if weighted_loss_total is None else weighted_loss_total + weighted_detached
        for key, value in metrics_chunk.items():
            detached = value.detach()
            if key in {"ratio_max", "raw_log_ratio_max"}:
                metric_maxes[key] = detached if key not in metric_maxes else torch.maximum(metric_maxes[key], detached)
            else:
                weighted_metric = detached * weight
                metric_sums[key] = weighted_metric if key not in metric_sums else metric_sums[key] + weighted_metric
        del new_logprobs_chunk, loss_chunk, weighted_loss, scaled_loss, metrics_chunk
        if logger is not None and fit_cfg is not None and _should_log_memory(fit_cfg, update_step):
            logger.debug("[RLMemChunk] chunk=%s start=%s end=%s", chunk_idx, start, end)

    metrics: Dict[str, torch.Tensor] = {**metric_sums, **metric_maxes}
    if weighted_loss_total is None:
        raise RuntimeError("No response chunks were produced for GRPO loss")
    return weighted_loss_total, metrics


def maybe_collect_train_forward_router_usage(model, enabled: bool) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not bool(enabled):
        return None, "usage_tracking_disabled"
    return collect_train_forward_router_usage(model), None


def is_zero_advantage_batch(advantages: torch.Tensor) -> bool:
    return bool((advantages.detach().abs().sum() == 0).cpu().item())


def zero_advantage_retry_exceeded(retry_count: int, max_retries: int) -> bool:
    return int(retry_count) >= int(max_retries)


def build_zero_advantage_skip_record(
    *,
    fit_cfg,
    micro_step: int,
    update_step: int,
    next_update_step: int,
    zero_advantage_retry_count: int,
    rewards: Sequence[float],
    reward_debugs: Sequence[Dict[str, Any]],
    response_lens: Sequence[float],
) -> Dict[str, Any]:
    return {
        "kind": "zero_advantage_skip",
        "micro_step": int(micro_step),
        "update_step": int(update_step),
        "next_update_step": int(next_update_step),
        "zero_advantage_retry_count": int(zero_advantage_retry_count),
        "max_zero_advantage_rollout_retries": int(getattr(fit_cfg.rl, "max_zero_advantage_rollout_retries", 8)),
        "reward_mean": _mean(rewards),
        "reward_std": _std(rewards),
        "strict_acc": _mean(float(debug["strict_match"]) for debug in reward_debugs),
        "fallback_acc": _mean(float(debug["fallback_match"]) for debug in reward_debugs),
        "avg_response_len": _mean(float(value) for value in response_lens),
        "min_response_len": float(min(response_lens)) if response_lens else 0.0,
        "max_response_len": float(max(response_lens)) if response_lens else 0.0,
        "prompt_format": str(getattr(fit_cfg.rl, "prompt_format", "raw") or "raw"),
        "chat_enable_thinking": bool(getattr(fit_cfg.rl, "chat_enable_thinking", False)),
        **_cuda_memory_snapshot(),
        "time": time.time(),
    }


def build_zero_advantage_retry_limit_record(
    *,
    fit_cfg,
    micro_step: int,
    update_step: int,
    zero_advantage_retry_count: int,
) -> Dict[str, Any]:
    return {
        "kind": "zero_advantage_retry_limit",
        "micro_step": int(micro_step),
        "update_step": int(update_step),
        "zero_advantage_retry_count": int(zero_advantage_retry_count),
        "max_zero_advantage_rollout_retries": int(getattr(fit_cfg.rl, "max_zero_advantage_rollout_retries", 8)),
        "zero_advantage_retry_action": str(getattr(fit_cfg.rl, "zero_advantage_retry_action", "warn_continue")),
        **_cuda_memory_snapshot(),
        "time": time.time(),
    }


def build_rl_prompt_text(fit_cfg, tokenizer, question: str) -> str:
    prompt_format = str(getattr(fit_cfg.rl, "prompt_format", "raw") or "raw").strip().lower()
    if prompt_format == "raw":
        return format_rl_prompt(fit_cfg.rl.prompt_template, question)
    if prompt_format != "chat":
        raise ValueError(f"Unsupported rl.prompt_format={prompt_format!r}")
    user_content = format_rl_prompt(fit_cfg.rl.prompt_template, question)
    return build_chat_prompt_text_for_generation(
        tokenizer,
        user_content=user_content,
        system_prompt=getattr(fit_cfg.rl, "chat_system_prompt", None),
        enable_thinking=bool(getattr(fit_cfg.rl, "chat_enable_thinking", False)),
    )


def build_rl_logger(rl_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"fitmotn.rl.{rl_dir.parent.name}.{rl_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(rl_dir / "rl_train.log", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _make_implicit_rl_run_name(fit_cfg) -> str:
    resume_from = getattr(fit_cfg.rl, "resume_from", None)
    if resume_from:
        resume_path = Path(resume_from).expanduser().resolve()
        parent_name = resume_path.parent.name if resume_path.name == "final_model" else resume_path.name
        return f"{parent_name}_grpo"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = Path(fit_cfg.model.model_path).name.replace("/", "_")
    return f"fitmotn_rl_{model_tag}_{ts}"


def resolve_rl_output_dir(fit_cfg) -> Path:
    root_dir = Path(fit_cfg.output.root_dir).expanduser().resolve()
    explicit_run_name = bool(getattr(fit_cfg.output, "run_name", None))
    run_name = fit_cfg.output.run_name or _make_implicit_rl_run_name(fit_cfg)
    rl_dir = root_dir / run_name / str(getattr(fit_cfg.rl, "output_subdir", "rl_grpo"))

    resume_from = getattr(fit_cfg.rl, "resume_from", None)
    if resume_from:
        resume_path = Path(resume_from).expanduser().resolve()
        try:
            if rl_dir.resolve() == resume_path or resume_path in rl_dir.resolve().parents:
                raise ValueError(
                    f"RL output dir {rl_dir} must not be the same as or inside rl.resume_from={resume_path}"
                )
        except FileNotFoundError:
            pass

    if rl_dir.exists() and any(rl_dir.iterdir()) and not bool(fit_cfg.output.overwrite_output_dir):
        if explicit_run_name:
            raise FileExistsError(
                f"RL output dir already exists and output.overwrite_output_dir=false: {rl_dir}"
            )
        base = rl_dir
        suffix = 1
        while rl_dir.exists() and any(rl_dir.iterdir()):
            rl_dir = base.parent / f"{base.name}_{suffix}"
            suffix += 1
    rl_dir.mkdir(parents=True, exist_ok=True)
    return rl_dir


def _cycle_batch(records: Sequence[Dict[str, Any]], start: int, batch_size: int) -> Tuple[List[Dict[str, Any]], int]:
    batch = []
    idx = int(start)
    for _ in range(int(batch_size)):
        batch.append(records[idx % len(records)])
        idx += 1
    return batch, idx


def build_response_mask(
    sequences: torch.Tensor,
    *,
    response_start: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> torch.Tensor:
    mask = torch.zeros_like(sequences, dtype=torch.float32)
    response_start = int(response_start)
    for row in range(sequences.shape[0]):
        end = sequences.shape[1]
        if eos_token_id is not None and response_start < sequences.shape[1]:
            eos_hits = (sequences[row, response_start:] == int(eos_token_id)).nonzero(as_tuple=False)
            if eos_hits.numel() > 0:
                end = response_start + int(eos_hits[0].item()) + 1
        if end > response_start:
            mask[row, response_start:end] = 1.0
    if pad_token_id is not None:
        mask = mask * (sequences != int(pad_token_id)).to(dtype=mask.dtype)
    if mask.shape[1] > 0:
        mask[:, 0] = 0.0
    return mask


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(1, len(values)))


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _save_rl_model_artifacts(
    *,
    model,
    tokenizer,
    output_dir: Path,
    load_info: RLPolicyLoadInfo,
    fit_cfg,
    update_step: int,
    checkpoint_name: str,
    trainable_mode_info: Dict[str, Any],
    extra_metadata: Dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(output_dir)
    except Exception:
        pass
    tokenizer.save_pretrained(output_dir)

    if load_info.layer_idxs:
        metadata = dict(load_info.metadata or {})
        metadata.update(
            {
                "checkpoint_format": metadata.get("checkpoint_format", "patch_state_only_v2"),
                "checkpoint_name": checkpoint_name,
                "save_time": time.time(),
                "base_model_path": load_info.base_model_path,
                "tokenizer_path": load_info.tokenizer_path or load_info.base_model_path,
                "layers_to_patch": list(load_info.layer_idxs),
                "patch_cfg": dict(load_info.patch_cfg),
                "motn_cfg": dict(load_info.patch_cfg),
                "patch_backend": str(load_info.patch_cfg.get("patch_backend", metadata.get("patch_backend", "motn"))),
                "fit_cfg": to_jsonable(asdict(fit_cfg)),
                "patch_state_dict": extract_patch_state_dict(model, load_info.layer_idxs),
                "rl_cfg": to_jsonable(asdict(fit_cfg.rl)),
                "grpo_metadata": {
                    "updates_done": int(update_step),
                    "global_step": int(update_step),
                    "mode": str(fit_cfg.rl.mode),
                    "trainable_mode": str(trainable_mode_info.get("requested_mode")),
                    "effective_trainable_mode": str(trainable_mode_info.get("effective_mode")),
                    "loaded_from": load_info.loaded_from,
                    **dict(extra_metadata or {}),
                },
                "updates_done": int(update_step),
                "global_step": int(update_step),
                "resolved_model_dtype": dtype_name_from_load_info(load_info),
                "trainable_mode": str(trainable_mode_info.get("requested_mode")),
            }
        )
        metadata.pop("state_dict", None)
        save_fitmotn_metadata(output_dir, metadata)
    return output_dir


def _safe_load_info_summary(load_info: RLPolicyLoadInfo) -> Dict[str, Any]:
    metadata = load_info.metadata or {}
    metadata_summary = None
    if isinstance(metadata, dict):
        metadata_summary = {
            "keys": sorted(str(key) for key in metadata.keys() if key not in {"state_dict", "patch_state_dict"}),
            "checkpoint_format": metadata.get("checkpoint_format"),
            "checkpoint_name": metadata.get("checkpoint_name"),
            "updates_done": metadata.get("updates_done"),
            "global_step": metadata.get("global_step"),
        }
    return {
        "is_fitmotn": bool(load_info.is_fitmotn),
        "layer_idxs": list(load_info.layer_idxs),
        "patch_cfg": dict(load_info.patch_cfg),
        "base_model_path": load_info.base_model_path,
        "tokenizer_path": load_info.tokenizer_path,
        "resolved_dtype": dtype_name_from_load_info(load_info),
        "loaded_from": load_info.loaded_from,
        "metadata_summary": metadata_summary,
    }


def _build_group_samples(
    *,
    batch: Sequence[Dict[str, Any]],
    prompts: Sequence[str],
    generated_texts: Sequence[str],
    rewards: Sequence[float],
    reward_debugs: Sequence[Dict[str, Any]],
    advantages: torch.Tensor,
    response_lens: Sequence[float],
    group_size: int,
    max_groups: int,
) -> List[Dict[str, Any]]:
    grouped = []
    adv_cpu = advantages.reshape(len(batch), int(group_size)).detach().cpu()
    for bidx, row in enumerate(batch[: int(max_groups)]):
        samples = []
        for gidx in range(int(group_size)):
            flat = bidx * int(group_size) + gidx
            debug = reward_debugs[flat]
            samples.append(
                {
                    "generated_text": generated_texts[flat],
                    "pred_answer": debug.get("pred_answer"),
                    "reward": float(rewards[flat]),
                    "advantage": float(adv_cpu[bidx, gidx].item()),
                    "match_type": debug.get("match_type"),
                    "strict_match": bool(debug.get("strict_match")),
                    "fallback_match": bool(debug.get("fallback_match")),
                    "response_len": float(response_lens[flat]),
                }
            )
        grouped.append(
            {
                "prompt_id": row.get("idx", bidx),
                "prompt": prompts[bidx],
                "gold_answer": reward_debugs[bidx * int(group_size)].get("gold_answer"),
                "samples": samples,
            }
        )
    return grouped


def _run_best_effort_eval(fit_cfg, model, tokenizer, rl_dir: Path, logger: logging.Logger, update_step: int) -> None:
    try:
        result = run_eval_tasks(
            fit_cfg,
            tasks=list(getattr(fit_cfg.rl, "eval_tasks", ["gsm8k"]) or ["gsm8k"]),
            logger=logger,
            eval_name=f"rl_eval_u{int(update_step)}",
            out_root=rl_dir / "rl_eval_outputs",
            eval_mode="mid",
            model=model,
            tokenizer=tokenizer,
            model_or_path=None,
            allow_backend_skip=True,
        )
        jsonl_append(rl_dir / "rl_eval.jsonl", {"kind": "rl_eval", "update_step": int(update_step), "result": result, "time": time.time()})
    except Exception as exc:
        logger.warning("[RLEval] best-effort eval failed at update=%s: %s", update_step, exc)
        jsonl_append(rl_dir / "rl_eval.jsonl", {"kind": "rl_eval_warning", "update_step": int(update_step), "warning": str(exc), "time": time.time()})


def run_fitmotn_rl_training(fit_cfg):
    if not bool(getattr(fit_cfg.rl, "enabled", False)):
        raise ValueError("run_fitmotn_rl_training requires rl.enabled=true")
    if str(getattr(fit_cfg.rl, "mode", "gsm8k_grpo")) != "gsm8k_grpo":
        raise ValueError("Only rl.mode='gsm8k_grpo' is supported")
    if int(getattr(fit_cfg.rl, "rollout_micro_batch_size", 0)) != 0:
        raise NotImplementedError("rl.rollout_micro_batch_size is reserved for a future rollout microbatch implementation")

    seed = getattr(fit_cfg.rl, "seed", None)
    if seed is None:
        seed = int(getattr(fit_cfg.train, "seed", 0))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    rl_dir = resolve_rl_output_dir(fit_cfg)
    logger = build_rl_logger(rl_dir)
    logger.info("[RLRun] start dir=%s", rl_dir)
    assert_no_unsafe_paths(fit_cfg, context="rl", logger=logger)
    logger.info("[RLCfg] %s", json.dumps(to_jsonable(asdict(fit_cfg)), ensure_ascii=False))
    json_dump(rl_dir / "config.json", asdict(fit_cfg))
    json_dump(rl_dir / "rl_config.json", asdict(fit_cfg.rl))

    train_jsonl_path = rl_dir / "rl_train.jsonl"
    samples_jsonl_path = rl_dir / "rl_samples.jsonl"
    usage_jsonl_path = rl_dir / "rl_usage.jsonl"
    summary_path = rl_dir / "rl_run_summary.json"

    records = load_gsm8k_rl_records(fit_cfg, logger=logger)
    model, tokenizer, load_info = load_policy_for_rl(fit_cfg, resume_from=getattr(fit_cfg.rl, "resume_from", None), logger=logger)
    if str(getattr(fit_cfg.rl, "prompt_format", "raw") or "raw").strip().lower() == "chat" and not tokenizer_supports_chat_template(tokenizer):
        raise ValueError("rl.prompt_format=chat requires tokenizer.apply_chat_template")
    _maybe_log_cuda_memory(fit_cfg, logger, "after_model_load", reset_peak=True)
    first_param = next(model.parameters())
    logger.info(
        "[RLDType] param_dtype=%s resolved_model_dtype=%s cfg_model_torch_dtype=%s cfg_model_use_amp=%s amp_dtype=%s",
        str(first_param.dtype).replace("torch.", ""),
        dtype_name_from_load_info(load_info),
        getattr(fit_cfg.model, "torch_dtype", None),
        bool(getattr(fit_cfg.model, "use_amp", False)),
        dtype_name_or_none(resolve_amp_dtype(model)),
    )
    logger.info(
        "[RLFormat] prompt_format=%s chat_enable_thinking=%s chat_system_prompt_present=%s tokenizer_chat_template_present=%s",
        str(getattr(fit_cfg.rl, "prompt_format", "raw") or "raw"),
        bool(getattr(fit_cfg.rl, "chat_enable_thinking", False)),
        bool(getattr(fit_cfg.rl, "chat_system_prompt", None)),
        bool(getattr(tokenizer, "chat_template", None)),
    )
    if bool(getattr(fit_cfg.rl, "gradient_checkpointing", False)):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        logger.info("[RLCheckpointing] gradient_checkpointing=true use_cache=false")
    elif hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    trainable_mode_info = set_trainable_mode_for_rl(model, fit_cfg.rl.trainable_mode, logger=logger)
    _maybe_log_cuda_memory(fit_cfg, logger, "after_trainable_mode")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected for RL")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(fit_cfg.rl.lr))
    _maybe_log_cuda_memory(fit_cfg, logger, "after_optimizer_build")
    trainable_summary = build_trainable_summary(model, trainable_mode_info["trainable_names"], optimizer=optimizer)
    log_trainable_summary(trainable_summary, rl_dir, logger)
    ref_model = load_reference_for_rl(fit_cfg, logger=logger)

    amp_dtype = resolve_amp_dtype(model)
    amp_enabled = bool(getattr(fit_cfg.model, "use_amp", False) and torch.device(fit_cfg.model.device).type == "cuda" and amp_dtype is not None)
    run_start_time = time.time()
    run_start_record = {
        "kind": "run_start",
        "time": run_start_time,
        "seed": int(seed),
        "num_prompts": int(len(records)),
        "rl_cfg": to_jsonable(asdict(fit_cfg.rl)),
        "trainable_summary": trainable_summary,
        "trainable_mode_info": {key: value for key, value in trainable_mode_info.items() if key != "trainable_names"},
        "load_info": _safe_load_info_summary(load_info),
        "env_snapshot": build_environment_snapshot(),
        "prompt_format": str(getattr(fit_cfg.rl, "prompt_format", "raw") or "raw"),
        "chat_enable_thinking": bool(getattr(fit_cfg.rl, "chat_enable_thinking", False)),
        "amp_dtype": dtype_name_or_none(amp_dtype),
        "chat_system_prompt_present": bool(getattr(fit_cfg.rl, "chat_system_prompt", None)),
        "tokenizer_chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
    }
    jsonl_append(train_jsonl_path, run_start_record)

    data_pos = 0
    micro_step = 0
    optimizer_micro_step = 0
    update_step = 0
    zero_advantage_retry_count = 0
    optimizer.zero_grad(set_to_none=True)

    while update_step < int(fit_cfg.rl.max_steps):
        micro_step += 1
        batch, data_pos = _cycle_batch(records, data_pos, int(fit_cfg.rl.batch_size))
        prompts = [build_rl_prompt_text(fit_cfg, tokenizer, row["question"]) for row in batch]
        gold_answers = [row["answer"] for row in batch]
        rollout_prompts = [prompt for prompt in prompts for _ in range(int(fit_cfg.rl.group_size))]
        rollout_gold = [answer for answer in gold_answers for _ in range(int(fit_cfg.rl.group_size))]

        enc = tokenizer(rollout_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        response_start = int(input_ids.shape[1])
        next_update_step = int(update_step) + 1

        model.eval()
        try:
            disable_runtime_usage_tracking(model)
            reset_runtime_usage_buffers(model)
        except Exception as exc:
            logger.warning("[RLUsage] failed to disable/reset usage buffers before generation: %s", exc)
        _maybe_log_cuda_memory(fit_cfg, logger, "before_generate", update_step=next_update_step, micro_step=micro_step)
        with torch.no_grad():
            prev_use_cache = getattr(model.config, "use_cache", None) if hasattr(model, "config") else None
            generation_use_cache = bool(prev_use_cache) and not bool(getattr(fit_cfg.rl, "gradient_checkpointing", False))
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(fit_cfg.rl.max_new_tokens),
                do_sample=True,
                temperature=float(fit_cfg.rl.temperature),
                top_p=float(fit_cfg.rl.top_p),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=generation_use_cache,
            )
            if prev_use_cache is not None:
                model.config.use_cache = False
            _maybe_log_cuda_memory(fit_cfg, logger, "after_generate", update_step=next_update_step, micro_step=micro_step)
            full_attention = torch.ones_like(generated, dtype=attention_mask.dtype, device=generated.device)
            if response_start > 0:
                full_attention[:, :response_start] = attention_mask
            response_mask = build_response_mask(
                generated,
                response_start=response_start,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ).to(generated.device)

        generated_texts = tokenizer.batch_decode(generated[:, response_start:], skip_special_tokens=True)
        rewards = []
        reward_debugs = []
        for text, gold in zip(generated_texts, rollout_gold):
            reward, debug = gsm8k_reward(text, gold)
            rewards.append(float(reward))
            reward_debugs.append(debug)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=generated.device).reshape(
            int(fit_cfg.rl.batch_size),
            int(fit_cfg.rl.group_size),
        )
        advantages = compute_group_advantages(reward_tensor, int(fit_cfg.rl.group_size)).to(generated.device)
        _maybe_log_cuda_memory(fit_cfg, logger, "after_reward_advantage_before_old_logprobs", update_step=next_update_step, micro_step=micro_step)

        response_lens = response_mask.sum(dim=1).detach().cpu().tolist()
        strict_acc = _mean(float(debug["strict_match"]) for debug in reward_debugs)
        fallback_acc = _mean(float(debug["fallback_match"]) for debug in reward_debugs)
        reward_acc = _mean(rewards)
        group_nonzero_adv_frac = float((advantages.abs().sum(dim=1) > 0).float().mean().detach().cpu().item())
        zero_advantage_batch = is_zero_advantage_batch(advantages)
        if bool(getattr(fit_cfg.rl, "skip_zero_advantage_updates", True)) and zero_advantage_batch:
            zero_advantage_retry_count += 1
            skip_record = build_zero_advantage_skip_record(
                fit_cfg=fit_cfg,
                micro_step=micro_step,
                update_step=update_step,
                next_update_step=next_update_step,
                zero_advantage_retry_count=zero_advantage_retry_count,
                rewards=rewards,
                reward_debugs=reward_debugs,
                response_lens=response_lens,
            )
            jsonl_append(train_jsonl_path, skip_record)
            logger.info("[RLTrain] %s", json.dumps(skip_record, ensure_ascii=False))
            try:
                disable_runtime_usage_tracking(model)
                reset_runtime_usage_buffers(model)
            except Exception as exc:
                logger.warning("[RLUsage] failed to disable/reset usage buffers after zero-advantage skip: %s", exc)
            should_raise_zero_retry = zero_advantage_retry_exceeded(
                zero_advantage_retry_count,
                int(getattr(fit_cfg.rl, "max_zero_advantage_rollout_retries", 8)),
            )
            del enc
            del generated
            del input_ids
            del attention_mask
            del full_attention
            del response_mask
            del rewards
            del reward_debugs
            del reward_tensor
            del advantages
            del generated_texts
            del response_lens
            del prompts
            del rollout_prompts
            del rollout_gold
            del skip_record
            if int(getattr(fit_cfg.rl, "empty_cache_every", 0)) > 0:
                torch.cuda.empty_cache()
            if should_raise_zero_retry:
                retry_limit_record = build_zero_advantage_retry_limit_record(
                    fit_cfg=fit_cfg,
                    micro_step=micro_step,
                    update_step=update_step,
                    zero_advantage_retry_count=zero_advantage_retry_count,
                )
                if str(getattr(fit_cfg.rl, "zero_advantage_retry_action", "warn_continue")) == "raise":
                    jsonl_append(train_jsonl_path, retry_limit_record)
                    raise RuntimeError(
                        "Exceeded rl.max_zero_advantage_rollout_retries while skipping zero-advantage GRPO rollouts; "
                        f"max_zero_advantage_rollout_retries={int(getattr(fit_cfg.rl, 'max_zero_advantage_rollout_retries', 8))}"
                    )
                jsonl_append(train_jsonl_path, retry_limit_record)
                logger.warning("[RLTrain] %s", json.dumps(retry_limit_record, ensure_ascii=False))
                zero_advantage_retry_count = 0
            continue

        try:
            disable_runtime_usage_tracking(model)
            reset_runtime_usage_buffers(model)
        except Exception as exc:
            logger.warning("[RLUsage] failed to disable/reset usage buffers before old logprobs: %s", exc)
        old_logprobs = compute_old_logprobs_microbatched(
            model,
            generated,
            full_attention,
            response_mask,
            int(getattr(fit_cfg.rl, "logprob_micro_batch_size", 1)),
            response_start=response_start,
        ).detach()
        _maybe_log_cuda_memory(fit_cfg, logger, "after_old_logprobs", update_step=next_update_step, micro_step=micro_step)

        model.train()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        try:
            if bool(getattr(fit_cfg.rl, "enable_usage_tracking", False)):
                reset_runtime_usage_buffers(model)
                enable_runtime_usage_tracking(model)
            else:
                disable_runtime_usage_tracking(model)
                reset_runtime_usage_buffers(model)
        except Exception as exc:
            logger.warning("[RLUsage] failed to configure usage tracking before train forward: %s", exc)
        train_forward_router_usage = None
        router_usage_warning = None
        _maybe_log_cuda_memory(fit_cfg, logger, "before_new_logprobs", update_step=next_update_step, micro_step=micro_step)

        if ref_model is not None:
            ref_logprobs = compute_old_logprobs_microbatched(
                ref_model,
                generated,
                full_attention,
                response_mask,
                int(getattr(fit_cfg.rl, "logprob_micro_batch_size", 1)),
                response_start=response_start,
            ).detach()
        else:
            ref_logprobs = None

        loss, loss_metrics = compute_new_logprobs_loss_microbatched(
            model=model,
            input_ids=generated,
            attention_mask=full_attention,
            response_mask=response_mask,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            micro_batch_size=int(getattr(fit_cfg.rl, "logprob_micro_batch_size", 1)),
            grad_accum=int(fit_cfg.rl.grad_accum),
            eps_clip=float(fit_cfg.rl.eps_clip),
            beta=float(fit_cfg.rl.beta) if ref_logprobs is not None else 0.0,
            use_amp=bool(getattr(fit_cfg.model, "use_amp", False)),
            amp_dtype=amp_dtype,
            response_start=response_start,
            logger=logger,
            fit_cfg=fit_cfg,
            update_step=next_update_step,
            micro_step=micro_step,
        )
        _maybe_log_cuda_memory(fit_cfg, logger, "after_new_logprobs_forward_or_chunks", update_step=next_update_step, micro_step=micro_step)
        try:
            train_forward_router_usage, router_usage_warning = maybe_collect_train_forward_router_usage(
                model,
                bool(getattr(fit_cfg.rl, "enable_usage_tracking", False)),
            )
        except Exception as exc:
            router_usage_warning = str(exc)
            logger.warning("[RLUsage] failed to collect train forward usage: %s", exc)
        try:
            disable_runtime_usage_tracking(model)
            reset_runtime_usage_buffers(model)
        except Exception as exc:
            logger.warning("[RLUsage] failed to disable/reset usage buffers after train forward: %s", exc)
        _maybe_log_cuda_memory(fit_cfg, logger, "after_backward", update_step=next_update_step, micro_step=micro_step)

        grad_norm = None
        optimizer_micro_step += 1
        did_update = (optimizer_micro_step % int(fit_cfg.rl.grad_accum)) == 0
        if did_update:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(fit_cfg.rl.max_grad_norm))
            grad_norm = float(grad_norm_tensor.detach().cpu().item()) if isinstance(grad_norm_tensor, torch.Tensor) else float(grad_norm_tensor)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            zero_advantage_retry_count = 0
            _maybe_log_cuda_memory(fit_cfg, logger, "after_optimizer_step", update_step=update_step, micro_step=micro_step)

        if did_update and update_step % int(fit_cfg.rl.log_every) == 0:
            group_samples = None
            record = {
                "kind": "train",
                "micro_step": int(micro_step),
                "update_step": int(update_step),
                "loss": float(loss.detach().cpu().item()),
                "policy_loss": float(loss_metrics["policy_loss"].cpu().item()),
                "reward_mean": _mean(rewards),
                "reward_std": _std(rewards),
                "strict_acc": strict_acc,
                "fallback_acc": fallback_acc,
                "reward_acc": reward_acc,
                "group_nonzero_adv_frac": group_nonzero_adv_frac,
                "adv_abs_mean": float(loss_metrics["adv_abs_mean"].cpu().item()),
                "kl_mean": float(loss_metrics["kl_mean"].cpu().item()),
                "clip_frac": float(loss_metrics["clip_frac"].cpu().item()),
                "ratio_mean": float(loss_metrics["ratio_mean"].cpu().item()),
                "ratio_max": float(loss_metrics["ratio_max"].cpu().item()),
                "raw_log_ratio_max": float(loss_metrics["raw_log_ratio_max"].cpu().item()),
                "avg_response_len": _mean(float(value) for value in response_lens),
                "min_response_len": float(min(response_lens)) if response_lens else 0.0,
                "max_response_len": float(max(response_lens)) if response_lens else 0.0,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": grad_norm,
                "current_trainable_mode": str(trainable_mode_info.get("requested_mode")),
                "effective_trainable_mode": str(trainable_mode_info.get("effective_mode")),
                "resolved_model_dtype": dtype_name_from_load_info(load_info),
                "amp_enabled": amp_enabled,
                "amp_dtype": dtype_name_or_none(amp_dtype),
                "prompt_format": str(getattr(fit_cfg.rl, "prompt_format", "raw") or "raw"),
                "chat_enable_thinking": bool(getattr(fit_cfg.rl, "chat_enable_thinking", False)),
                "response_start": int(response_start),
                "router_usage_warning": router_usage_warning,
                **_cuda_memory_snapshot(),
                "time": time.time(),
            }
            jsonl_append(train_jsonl_path, record)
            logger.info("[RLTrain] %s", json.dumps(record, ensure_ascii=False))
            group_samples = _build_group_samples(
                batch=batch,
                prompts=prompts,
                generated_texts=generated_texts,
                rewards=rewards,
                reward_debugs=reward_debugs,
                advantages=advantages,
                response_lens=response_lens,
                group_size=int(fit_cfg.rl.group_size),
                max_groups=int(fit_cfg.rl.sample_log_count),
            )
            jsonl_append(samples_jsonl_path, {"kind": "samples", "update_step": int(update_step), "time": time.time(), "groups": group_samples})
            if train_forward_router_usage:
                usage_record = {
                    "kind": "usage",
                    "micro_step": int(micro_step),
                    "update_step": int(update_step),
                    "train_forward_router_usage": train_forward_router_usage,
                    "time": time.time(),
                }
                jsonl_append(usage_jsonl_path, usage_record)
            del group_samples

        if did_update and int(fit_cfg.rl.eval_every_updates) > 0 and update_step % int(fit_cfg.rl.eval_every_updates) == 0:
            _run_best_effort_eval(fit_cfg, model, tokenizer, rl_dir, logger, update_step)

        if did_update and int(fit_cfg.rl.save_every_updates) > 0 and update_step % int(fit_cfg.rl.save_every_updates) == 0:
            ckpt_dir = rl_dir / f"checkpoint-{int(update_step)}"
            _save_rl_model_artifacts(
                model=model,
                tokenizer=tokenizer,
                output_dir=ckpt_dir,
                load_info=load_info,
                fit_cfg=fit_cfg,
                update_step=update_step,
                checkpoint_name=ckpt_dir.name,
                trainable_mode_info=trainable_mode_info,
                extra_metadata={"micro_step": int(micro_step)},
            )
            _maybe_log_cuda_memory(fit_cfg, logger, "after_checkpoint_save", update_step=update_step, micro_step=micro_step)

        del enc
        del generated
        del input_ids
        del attention_mask
        del full_attention
        del response_mask
        del old_logprobs
        del rewards
        del reward_debugs
        del reward_tensor
        del advantages
        del generated_texts
        del response_lens
        del ref_logprobs
        del loss
        del loss_metrics
        del train_forward_router_usage
        del router_usage_warning
        del prompts
        del rollout_prompts
        del rollout_gold
        if did_update and int(getattr(fit_cfg.rl, "empty_cache_every", 0)) > 0 and update_step % int(fit_cfg.rl.empty_cache_every) == 0:
            torch.cuda.empty_cache()

    final_model_dir = rl_dir / "final_model"
    _save_rl_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=final_model_dir,
        load_info=load_info,
        fit_cfg=fit_cfg,
        update_step=update_step,
        checkpoint_name=final_model_dir.name,
        trainable_mode_info=trainable_mode_info,
        extra_metadata={"micro_step": int(micro_step), "duration_sec": max(0.0, time.time() - run_start_time)},
    )
    _maybe_log_cuda_memory(fit_cfg, logger, "after_checkpoint_save", update_step=update_step, micro_step=micro_step)
    summary = {
        "kind": "rl_run_summary",
        "rl_dir": str(rl_dir),
        "final_model_dir": str(final_model_dir),
        "updates_done": int(update_step),
        "micro_steps": int(micro_step),
        "optimizer_micro_steps": int(optimizer_micro_step),
        "num_prompts": int(len(records)),
        "duration_sec": max(0.0, time.time() - run_start_time),
        "trainable_summary": trainable_summary,
        "trainable_mode_info": {key: value for key, value in trainable_mode_info.items() if key != "trainable_names"},
        "load_info": _safe_load_info_summary(load_info),
        "rl_cfg": to_jsonable(asdict(fit_cfg.rl)),
        "final_model_restore_compatible": bool(load_info.layer_idxs),
    }
    json_dump(summary_path, summary)
    json_dump(final_model_dir / "rl_run_summary.json", summary)
    jsonl_append(train_jsonl_path, {"kind": "run_end", "update_step": int(update_step), "micro_step": int(micro_step), "time": time.time()})
    logger.info("[RLRun] finished dir=%s final_model=%s", rl_dir, final_model_dir)
    return {
        "rl_dir": str(rl_dir),
        "final_model_dir": str(final_model_dir),
        "run_summary": str(summary_path),
        "train_jsonl": str(train_jsonl_path),
    }


def run_fitmotn_training_and_optional_rl(fit_cfg):
    from .controller import run_fitmotn_training

    sft_result = run_fitmotn_training(fit_cfg)
    rl_cfg = getattr(fit_cfg, "rl", None)
    if rl_cfg is None or not (bool(getattr(rl_cfg, "enabled", False)) and bool(getattr(rl_cfg, "run_after_sft", False))):
        return sft_result

    final_model_dir = None
    if isinstance(sft_result, dict):
        final_model_dir = sft_result.get("final_model_dir")
        if not final_model_dir and sft_result.get("run_dir"):
            inferred = Path(sft_result["run_dir"]).expanduser().resolve() / "final_model"
            if inferred.exists():
                final_model_dir = str(inferred)
    if not final_model_dir:
        raise RuntimeError("SFT completed but final_model_dir could not be determined for rl.run_after_sft")
    final_model_path = Path(final_model_dir).expanduser().resolve()
    if not final_model_path.exists():
        raise FileNotFoundError(f"SFT final_model_dir does not exist for RL resume: {final_model_path}")

    rl_fit_cfg = copy.deepcopy(fit_cfg)
    rl_fit_cfg.rl.resume_from = str(final_model_path)
    rl_result = run_fitmotn_rl_training(rl_fit_cfg)
    if isinstance(sft_result, dict):
        combined = dict(sft_result)
        combined["rl_result"] = rl_result
        return combined
    return {"sft_result": sft_result, "rl_result": rl_result}
