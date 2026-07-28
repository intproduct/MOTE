from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch as tc

from ..audit import (
    cuda_snapshot,
    derive_stage_step,
    host_memory_snapshot,
    tensor_distribution_stats,
    to_jsonable,
)
from .stages import resolved_stage_lr_summary

TRAIN_RECORD_BUFFER_SIZE = 128
USAGE_RECORD_BUFFER_SIZE = 32
USAGE_LIGHT_RECORD_BUFFER_SIZE = 128
PENDING_USAGE_LIGHT_BUFFER_SIZE = 512
RECENT_LOSS_BUFFER_SIZE = 64
REASONING_DATASET_FLAGS = {
    "gsm8k": "use_gsm8k_train",
    "gsm8k_socratic": "use_gsm8k_socratic_train",
    "svamp": "use_svamp_train",
    "synthetic_arithmetic": "use_synthetic_arithmetic_train",
    "metamath": "use_metamath_train",
    "hendrycks_math": "use_math_train",
    "mmlu": "use_mmlu_train",
    "openr1_math": "use_openr1_math",
    "numinamath_cot": "use_numinamath_cot",
    "openthoughts_math": "use_openthoughts_math",
    "bespoke_stratos": "use_bespoke_stratos",
}
REASONING_DATASET_WEIGHTS = {
    "gsm8k": "wt_gsm8k",
    "gsm8k_socratic": "wt_gsm8k_socratic",
    "svamp": "wt_svamp",
    "synthetic_arithmetic": "wt_synthetic_arithmetic",
    "metamath": "wt_metamath",
    "hendrycks_math": "wt_math",
    "mmlu": "wt_mmlu",
    "openr1_math": "wt_openr1_math",
    "numinamath_cot": "wt_numinamath_cot",
    "openthoughts_math": "wt_openthoughts_math",
    "bespoke_stratos": "wt_bespoke_stratos",
}

DEFERRED_LOSS_METRICS_MARKER = "_deferred_loss_observability"
PENDING_LOSS_METRICS_KEY = "_pending_loss_observability"


def _merge_scalar_maps(target: Dict[str, tc.Tensor], source: Dict[str, tc.Tensor]) -> None:
    for key, value in source.items():
        value = value.detach()
        target[key] = value if key not in target else target[key] + value


def accumulate_loss_observability(runtime: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Accumulate detached loss diagnostics on-device until an output boundary."""
    if not metrics or not metrics.get(DEFERRED_LOSS_METRICS_MARKER):
        return
    pending = runtime.get(PENDING_LOSS_METRICS_KEY)
    if pending is None:
        pending = {
            DEFERRED_LOSS_METRICS_MARKER: True,
            "bucket_loss_sums": {},
            "bucket_token_counts": {},
            "task_loss_sums": {},
            "task_token_counts": {},
            "final_answer_loss_sum": metrics["final_answer_loss_sum"].detach(),
            "final_answer_weighted_tokens": metrics["final_answer_weighted_tokens"].detach(),
        }
        runtime[PENDING_LOSS_METRICS_KEY] = pending
    else:
        pending["final_answer_loss_sum"] = (
            pending["final_answer_loss_sum"] + metrics["final_answer_loss_sum"].detach()
        )
        pending["final_answer_weighted_tokens"] = (
            pending["final_answer_weighted_tokens"] + metrics["final_answer_weighted_tokens"].detach()
        )
    for name in ("bucket_loss_sums", "bucket_token_counts", "task_loss_sums", "task_token_counts"):
        _merge_scalar_maps(pending[name], metrics[name])


def materialize_loss_observability(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all deferred scalar diagnostics with a single device-to-host copy."""
    if not metrics or not metrics.get(DEFERRED_LOSS_METRICS_MARKER):
        return metrics or {}

    bucket_names = list(metrics["bucket_loss_sums"])
    task_names = list(metrics["task_loss_sums"])
    scalars = []
    for name in bucket_names:
        scalars.extend((metrics["bucket_loss_sums"][name], metrics["bucket_token_counts"][name]))
    for name in task_names:
        scalars.extend((metrics["task_loss_sums"][name], metrics["task_token_counts"][name]))
    scalars.extend((metrics["final_answer_loss_sum"], metrics["final_answer_weighted_tokens"]))
    host_values = tc.stack([value.detach().to(dtype=tc.float64) for value in scalars]).cpu().tolist()

    result: Dict[str, Any] = {}
    offset = 0
    for name in bucket_names:
        loss_sum, token_count = host_values[offset : offset + 2]
        offset += 2
        count = int(token_count)
        result[f"loss/{name}"] = None if count <= 0 else float(loss_sum) / count
        result[f"tokens/{name}"] = count
    for name in task_names:
        loss_sum, token_count = host_values[offset : offset + 2]
        offset += 2
        count = int(token_count)
        if count > 0:
            result[f"loss_by_task/{name}"] = float(loss_sum) / count
    final_loss_sum, final_token_count = host_values[offset : offset + 2]
    final_count = int(final_token_count)
    result["final_answer_weighted_tokens"] = final_count
    result["final_answer_loss"] = None if final_count <= 0 else float(final_loss_sum) / final_count
    return result


def flush_loss_observability(runtime: Dict[str, Any]) -> Dict[str, Any]:
    """Publish pending GPU diagnostics at a logging or checkpoint boundary."""
    pending = runtime.pop(PENDING_LOSS_METRICS_KEY, None)
    if pending is None:
        return runtime.get("loss_observability") or {}
    materialized = materialize_loss_observability(pending)
    runtime["loss_observability"] = materialized
    runtime["final_answer_weighted_tokens"] = materialized.get("final_answer_weighted_tokens", 0)
    runtime["final_answer_loss"] = materialized.get("final_answer_loss")
    return materialized


@dataclass
class UpdateBatchMeta:
    batch_task_names: Set[str] = field(default_factory=set)
    batch_groups: Set[str] = field(default_factory=set)
    batch_buckets: Set[str] = field(default_factory=set)
    batch_source_families: Set[str] = field(default_factory=set)
    microbatch_count: int = 0

    def clear(self) -> None:
        self.batch_task_names.clear()
        self.batch_groups.clear()
        self.batch_buckets.clear()
        self.batch_source_families.clear()
        self.microbatch_count = 0


def build_reasoning_config_summary(fit_cfg) -> Dict[str, Any]:
    data_cfg = fit_cfg.data
    train_cfg = fit_cfg.train
    enabled = {name: bool(getattr(data_cfg, field, False)) for name, field in REASONING_DATASET_FLAGS.items()}
    weights = {name: float(getattr(data_cfg, field, 0.0)) for name, field in REASONING_DATASET_WEIGHTS.items()}
    return {
        "reasoning_supervision_mode": str(getattr(data_cfg, "reasoning_supervision_mode", "answer_only")),
        "reasoning_datasets_enabled": enabled,
        "reasoning_dataset_weights": weights,
        "task_bucket_mode": str(getattr(train_cfg, "task_bucket_mode", "flat")),
        "stage_b_mode": str(getattr(train_cfg, "stage_b_mode", "mixed")),
        "stage_b_disable_pretrain": bool(getattr(train_cfg, "stage_b_disable_pretrain", False)),
        "stage_b_reasoning_boost": float(getattr(train_cfg, "stage_b_reasoning_boost", 1.0)),
        "answer_format": "Final Answer",
    }


def _current_stage_details(runtime: Dict[str, Any]) -> Dict[str, Any]:
    stage_plan = runtime.get("stage_plan")
    current_stage_name = runtime.get("current_stage")
    if not stage_plan or current_stage_name is None:
        return {
            "stage_pretrain_ratio": None,
            "stage_task_ratio": None,
            "stage_bucket_mode": None,
            "stage_bucket_ratios": None,
            "stage_mode": None,
            "stage_reasoning_focused": None,
            "stage_pretrain_disabled": None,
            "stage_reasoning_boost": None,
        }
    for stage in stage_plan.stages:
        if stage.name == current_stage_name:
            return {
                "stage_pretrain_ratio": stage.pretrain_ratio,
                "stage_task_ratio": stage.task_ratio,
                "stage_bucket_mode": getattr(stage, "task_bucket_mode", None),
                "stage_bucket_ratios": getattr(stage, "bucket_ratios", None),
                "stage_mode": getattr(stage, "mode", None),
                "stage_reasoning_focused": getattr(stage, "reasoning_focused", None),
                "stage_pretrain_disabled": getattr(stage, "pretrain_disabled", None),
                "stage_reasoning_boost": getattr(stage, "reasoning_boost", None),
            }
    return {
        "stage_pretrain_ratio": None,
        "stage_task_ratio": None,
        "stage_bucket_mode": None,
        "stage_bucket_ratios": None,
        "stage_mode": None,
        "stage_reasoning_focused": None,
        "stage_pretrain_disabled": None,
        "stage_reasoning_boost": None,
    }


def make_runtime_state(
    *,
    run_name: str,
    run_dir: Path,
    fit_cfg,
    stage_plan,
    model_dtype: tc.dtype,
    amp_enabled: bool,
    scheduler_metadata: Dict[str, Any],
    env_snapshot: Dict[str, Any],
    param_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    total_updates = int(stage_plan.total_updates)
    batch_size = int(fit_cfg.train.batch_size)
    grad_accum = int(fit_cfg.train.grad_accum)
    seq_len = int(fit_cfg.data.seq_len_run)
    tokens_per_microbatch = batch_size * seq_len
    tokens_per_update = tokens_per_microbatch * grad_accum
    reasoning_summary = build_reasoning_config_summary(fit_cfg)
    stage_lr_summary = resolved_stage_lr_summary(fit_cfg.train)
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "start_time": time.time(),
        "fit_cfg": fit_cfg,
        "stage_plan": stage_plan,
        "resolved_model_dtype": str(model_dtype).replace("torch.", ""),
        "amp_enabled": bool(amp_enabled),
        "warmup_updates": int(scheduler_metadata.get("resolved_lr_warmup_steps", 0)),
        "total_updates_planned": total_updates,
        "actual_total_updates": total_updates,
        "stage_a_updates": int(getattr(stage_plan, "stage_a_updates", 0)),
        "stage_b_updates": int(getattr(stage_plan, "stage_b_updates", 0)),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "tokens_per_microbatch": tokens_per_microbatch,
        "tokens_per_update": tokens_per_update,
        "samples_seen_estimate": 0,
        "tokens_seen_estimate": 0,
        "current_T": None,
        "current_temperature": None,
        "gate_trainable": None,
        "current_stage": None,
        "current_stage_update": None,
        "current_lr": None,
        "current_param_group_lrs": None,
        "current_loss": None,
        "current_epoch": None,
        "optimizer_type": None,
        "optimizer_param_group_count": None,
        "optimizer_param_groups": None,
        "optimizer_fallback_param_names": [],
        "weight_decay": None,
        "scheduler_step": 0,
        "lr_scheduler_enabled": True,
        "lr_scheduler_type": scheduler_metadata.get("lr_scheduler_type"),
        "lr_warmup": scheduler_metadata.get("lr_warmup"),
        "lr_warmup_steps": scheduler_metadata.get("lr_warmup_steps"),
        "resolved_lr_warmup_steps": scheduler_metadata.get("resolved_lr_warmup_steps"),
        "lr_decay_steps": scheduler_metadata.get("lr_decay_steps"),
        "resolved_lr_decay_steps": scheduler_metadata.get("resolved_lr_decay_steps"),
        "actual_training_steps": scheduler_metadata.get("actual_training_steps"),
        "scheduler_total_steps": scheduler_metadata.get("scheduler_total_steps"),
        **stage_lr_summary,
        "temperature_schedule_type": getattr(fit_cfg.train, "temperature_schedule_type", "cosine"),
        "stage_a_begin_t": getattr(fit_cfg.train, "stage_a_begin_t", None),
        "stage_a_end_t": getattr(fit_cfg.train, "stage_a_end_t", None),
        "stage_b_begin_t": getattr(fit_cfg.train, "stage_b_begin_t", None),
        "stage_b_end_t": getattr(fit_cfg.train, "stage_b_end_t", None),
        "final_answer_weight_enabled": bool(getattr(fit_cfg.train, "final_answer_weight_enabled", False)),
        "final_answer_weight": float(getattr(fit_cfg.train, "final_answer_weight", 1.0)),
        "final_answer_marker": getattr(fit_cfg.train, "final_answer_marker", "####"),
        "final_answer_weighted_tokens": 0,
        "final_answer_loss": None,
        "loss_observability": {},
        "loss_scale": None,
        "overflow_or_nan_detected": None,
        "num_nan_grads": None,
        "num_inf_grads": None,
        "optimizer_step_skipped": None,
        "trainable_params": int(param_snapshot["trainable_params"]),
        "total_params": int(param_snapshot["total_params"]),
        "trainable_ratio": param_snapshot["trainable_ratio"],
        "patch_backend": None,
        "patch_cfg": None,
        "gate_arch": None,
        "gate_hidden_dim": None,
        "resolved_gate_hidden_dim": None,
        "gate_activation": None,
        "gate_norm": None,
        "gate_dropout": None,
        "gate_mlp_bias": None,
        "gate_output_init_std": None,
        "gate_residual_delta_scale": None,
        "router_param_count": None,
        "router_param_ratio_vs_blocks": None,
        "gate_router_summary": None,
        "env_snapshot": env_snapshot,
        "param_snapshot": param_snapshot,
        "last_update_end_time": None,
        "last_update_step_time_sec": None,
        "last_update_wall_time": None,
        "update_batch_meta": UpdateBatchMeta(),
        "last_batch_meta": {
            "batch_task_names": None,
            "batch_groups": None,
            "batch_source_families": None,
            "batch_buckets": None,
            "microbatch_count": None,
        },
        "last_train_record": None,
        "last_usage_record": None,
        "last_usage_light_record": None,
        "latest_mid_eval_summary": None,
        "final_full_summary": None,
        "compare_vs_baseline": None,
        "approx_init_summary": None,
        "is_resume_training": bool(getattr(fit_cfg.train, "resume_fitmotn_from", None)),
        "resume_fitmotn_from": getattr(fit_cfg.train, "resume_fitmotn_from", None),
        "resume_stage": getattr(fit_cfg.train, "resume_stage", "auto"),
        "stage2_only_on_resume": bool(getattr(fit_cfg.train, "stage2_only_on_resume", True)),
        "extra_updates": getattr(fit_cfg.train, "extra_updates", None),
        "approx_init_skipped_due_to_resume": False,
        "loaded_fitmotn_state_path": None,
        "loaded_checkpoint_metadata": None,
        "resume_load_summary": None,
        "current_model_structure_summary": None,
        "expert_warmup_scaling_summary": None,
        "baseline_small_summary": None,
        "baseline_final_summary": None,
        "scheduler_state_summary": None,
        "checkpoint_format": "patch_state_only_v2",
        "train_records": deque(maxlen=TRAIN_RECORD_BUFFER_SIZE),
        "usage_records": deque(maxlen=USAGE_RECORD_BUFFER_SIZE),
        "usage_light_records": deque(maxlen=USAGE_LIGHT_RECORD_BUFFER_SIZE),
        "pending_usage_light_snapshots": deque(maxlen=PENDING_USAGE_LIGHT_BUFFER_SIZE),
        "mid_eval_records": [],
        "train_record_count": 0,
        "usage_record_count": 0,
        "usage_light_record_count": 0,
        "tokens_per_sec_sum": 0.0,
        "tokens_per_sec_count": 0,
        "max_cuda_mem_peak_alloc_mb": None,
        "recent_train_losses": deque(maxlen=RECENT_LOSS_BUFFER_SIZE),
        "latest_mid_eval_update": None,
        "reasoning_supervision_mode": reasoning_summary["reasoning_supervision_mode"],
        "reasoning_datasets_enabled": reasoning_summary["reasoning_datasets_enabled"],
        "reasoning_dataset_weights": reasoning_summary["reasoning_dataset_weights"],
        "task_bucket_mode": reasoning_summary["task_bucket_mode"],
        "stage_b_mode": reasoning_summary["stage_b_mode"],
        "stage_b_disable_pretrain": reasoning_summary["stage_b_disable_pretrain"],
        "stage_b_reasoning_boost": reasoning_summary["stage_b_reasoning_boost"],
        "answer_format": reasoning_summary["answer_format"],
        "bucket_sampling_counts": {},
    }


def register_microbatch(runtime: Dict[str, Any], inputs: Dict[str, Any]) -> None:
    meta: UpdateBatchMeta = runtime["update_batch_meta"]
    for key, target in [
        ("task", meta.batch_task_names),
        ("group", meta.batch_groups),
        ("bucket", meta.batch_buckets),
        ("source_family", meta.batch_source_families),
    ]:
        value = inputs.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if item is not None:
                    target.add(str(item))
        else:
            target.add(str(value))
    meta.microbatch_count += 1


def finalize_update_batch_meta(runtime: Dict[str, Any]) -> Dict[str, Any]:
    meta: UpdateBatchMeta = runtime["update_batch_meta"]
    snapshot = {
        "batch_task_names": sorted(meta.batch_task_names) if meta.batch_task_names else None,
        "batch_groups": sorted(meta.batch_groups) if meta.batch_groups else None,
        "batch_buckets": sorted(meta.batch_buckets) if meta.batch_buckets else None,
        "batch_source_families": sorted(meta.batch_source_families) if meta.batch_source_families else None,
        "microbatch_count": int(meta.microbatch_count) if meta.microbatch_count else None,
    }
    runtime["last_batch_meta"] = snapshot
    meta.clear()
    return snapshot


def update_step_clock(runtime: Dict[str, Any], now: Optional[float] = None) -> Optional[float]:
    now = time.time() if now is None else float(now)
    prev = runtime.get("last_update_end_time")
    runtime["last_update_end_time"] = now
    if prev is None:
        runtime["last_update_step_time_sec"] = None
    else:
        runtime["last_update_step_time_sec"] = float(max(0.0, now - float(prev)))
    runtime["last_update_wall_time"] = now
    return runtime["last_update_step_time_sec"]


def update_step_runtime(runtime: Dict[str, Any], *, global_step: int, stage_name: str, stage_start: int, current_t: Optional[float], gate_trainable: Optional[bool], lr: Optional[float], epoch: Optional[float], loss: Optional[float], now: Optional[float] = None) -> None:
    runtime["scheduler_step"] = int(global_step)
    runtime["current_stage"] = stage_name
    runtime["current_stage_update"] = derive_stage_step(global_step, stage_start)
    runtime["gate_trainable"] = None if gate_trainable is None else bool(gate_trainable)
    runtime["current_T"] = None if current_t is None else float(current_t)
    runtime["current_temperature"] = None if current_t is None else float(current_t)
    runtime["current_lr"] = None if lr is None else float(lr)
    runtime["current_epoch"] = None if epoch is None else float(epoch)
    runtime["current_loss"] = None if loss is None else float(loss)
    step_time = update_step_clock(runtime, now=now)
    runtime["samples_seen_estimate"] = int(global_step) * int(runtime["batch_size"]) * int(runtime["grad_accum"])
    runtime["tokens_seen_estimate"] = int(runtime["samples_seen_estimate"]) * int(runtime["seq_len"])
    runtime["step_time_sec"] = step_time
    runtime["tokens_per_sec"] = None if step_time in (None, 0) else float(runtime["tokens_per_update"] / max(step_time, 1e-12))


def snapshot_cuda(runtime: Dict[str, Any], device: tc.device | str | None = None) -> Dict[str, Any]:
    snapshot = cuda_snapshot(device)
    snapshot.update(host_memory_snapshot())
    runtime.update(snapshot)
    return snapshot


def _clone_usage_tensor(value: Any) -> Any:
    if isinstance(value, tc.Tensor):
        return value.detach().clone()
    return value


def _resolve_top1_value(core: Any, fallback_top1: Any) -> tuple[Any, str]:
    last_top1 = getattr(core, "last_top1", None)
    if last_top1 is not None:
        return to_jsonable(last_top1), "core.last_top1"
    if fallback_top1 is not None:
        return fallback_top1, "usage_argmax_fallback"
    return None, "unavailable"


def build_train_record(runtime: Dict[str, Any], logs: Dict[str, Any]) -> Dict[str, Any]:
    batch_meta = runtime.get("last_batch_meta") or {}
    loss = logs.get("loss", logs.get("train_loss"))
    lr = logs.get("learning_rate", logs.get("lr", runtime.get("current_lr")))
    record = {
        "kind": "train",
        "time": time.time(),
        "wall_time": runtime.get("last_update_wall_time", time.time()),
        "run_name": runtime["run_name"],
        "seed": int(runtime.get("fit_cfg").train.seed) if runtime.get("fit_cfg") is not None else None,
        "stage": runtime.get("current_stage"),
        "stage_update": runtime.get("current_stage_update"),
        "update_step": logs.get("step", runtime.get("scheduler_step")),
        "global_step": runtime.get("scheduler_step"),
        "epoch": logs.get("epoch", runtime.get("current_epoch")),
        "train_loss": None if loss is None else float(loss),
        "lr": None if lr is None else float(lr),
        "T": runtime.get("current_T"),
        "current_temperature": runtime.get("current_temperature"),
        "gate_trainable": runtime.get("gate_trainable"),
        "gate_freeze_steps": runtime.get("fit_cfg").train.gate_freeze_steps if runtime.get("fit_cfg") is not None else None,
        "batch_size": runtime.get("batch_size"),
        "seq_len": runtime.get("seq_len"),
        "grad_accum": runtime.get("grad_accum"),
        "tokens_per_microbatch": runtime.get("tokens_per_microbatch"),
        "tokens_per_update": runtime.get("tokens_per_update"),
        "samples_seen_estimate": runtime.get("samples_seen_estimate"),
        "tokens_seen_estimate": runtime.get("tokens_seen_estimate"),
        "optimizer_step_skipped": runtime.get("optimizer_step_skipped"),
        "step_time_sec": runtime.get("step_time_sec"),
        "tokens_per_sec": runtime.get("tokens_per_sec"),
        "batch_task_names": batch_meta.get("batch_task_names"),
        "batch_groups": batch_meta.get("batch_groups"),
        "batch_buckets": batch_meta.get("batch_buckets"),
        "batch_source_families": batch_meta.get("batch_source_families"),
        "stage_pretrain_ratio": None,
        "stage_task_ratio": None,
        "stage_bucket_mode": None,
        "stage_bucket_ratios": None,
        "stage_mode": None,
        "stage_reasoning_focused": None,
        "stage_pretrain_disabled": None,
        "stage_reasoning_boost": None,
        "grad_norm": logs.get("grad_norm", runtime.get("grad_norm")),
        "param_norm": runtime.get("param_norm"),
        "loss_scale": runtime.get("loss_scale"),
        "overflow_or_nan_detected": runtime.get("overflow_or_nan_detected"),
        "num_nan_grads": runtime.get("num_nan_grads"),
        "num_inf_grads": runtime.get("num_inf_grads"),
        "cuda_mem_alloc_mb": runtime.get("cuda_mem_alloc_mb"),
        "cuda_mem_reserved_mb": runtime.get("cuda_mem_reserved_mb"),
        "cuda_mem_peak_alloc_mb": runtime.get("cuda_mem_peak_alloc_mb"),
        "cuda_mem_peak_reserved_mb": runtime.get("cuda_mem_peak_reserved_mb"),
        "cpu_ram_used_mb": runtime.get("cpu_ram_used_mb"),
        "host_ram_used_mb": runtime.get("host_ram_used_mb"),
        "lr_scheduler_enabled": runtime.get("lr_scheduler_enabled"),
        "lr_scheduler_type": runtime.get("lr_scheduler_type"),
        "lr_warmup": runtime.get("lr_warmup"),
        "lr_warmup_steps": runtime.get("lr_warmup_steps"),
        "resolved_lr_warmup_steps": runtime.get("resolved_lr_warmup_steps"),
        "lr_decay_steps": runtime.get("lr_decay_steps"),
        "resolved_lr_decay_steps": runtime.get("resolved_lr_decay_steps"),
        "actual_training_steps": runtime.get("actual_training_steps"),
        "scheduler_total_steps": runtime.get("scheduler_total_steps"),
        "warmup_updates": runtime.get("warmup_updates"),
        "scheduler_step": runtime.get("scheduler_step"),
        "current_param_group_lrs": runtime.get("current_param_group_lrs"),
        "stage_specific_lr_enabled": runtime.get("stage_specific_lr_enabled"),
        "stage_a_block_lr": runtime.get("stage_a_block_lr"),
        "stage_a_router_lr": runtime.get("stage_a_router_lr"),
        "stage_b_block_lr": runtime.get("stage_b_block_lr"),
        "stage_b_router_lr": runtime.get("stage_b_router_lr"),
        "resolved_stage_a_block_lr": runtime.get("resolved_stage_a_block_lr"),
        "resolved_stage_a_router_lr": runtime.get("resolved_stage_a_router_lr"),
        "resolved_stage_b_block_lr": runtime.get("resolved_stage_b_block_lr"),
        "resolved_stage_b_router_lr": runtime.get("resolved_stage_b_router_lr"),
        "temperature_schedule_type": runtime.get("temperature_schedule_type"),
        "stage_a_begin_t": runtime.get("stage_a_begin_t"),
        "stage_a_end_t": runtime.get("stage_a_end_t"),
        "stage_b_begin_t": runtime.get("stage_b_begin_t"),
        "stage_b_end_t": runtime.get("stage_b_end_t"),
        "final_answer_weight_enabled": runtime.get("final_answer_weight_enabled"),
        "final_answer_weight": runtime.get("final_answer_weight"),
        "final_answer_weighted_tokens": runtime.get("final_answer_weighted_tokens"),
        "final_answer_loss": runtime.get("final_answer_loss"),
        "optimizer_type": runtime.get("optimizer_type"),
        "optimizer_param_group_count": runtime.get("optimizer_param_group_count"),
        "optimizer_param_groups": runtime.get("optimizer_param_groups"),
        "optimizer_fallback_param_names": runtime.get("optimizer_fallback_param_names"),
        "weight_decay": runtime.get("weight_decay"),
        "max_grad_norm": runtime.get("fit_cfg").train.max_grad_norm if runtime.get("fit_cfg") is not None else None,
        "resolved_model_dtype": runtime.get("resolved_model_dtype"),
        "amp_enabled": runtime.get("amp_enabled"),
        "reasoning_supervision_mode": runtime.get("reasoning_supervision_mode"),
        "stage_b_mode": runtime.get("stage_b_mode"),
        "stage_b_disable_pretrain": runtime.get("stage_b_disable_pretrain"),
        "configured_stage_b_reasoning_boost": runtime.get("stage_b_reasoning_boost"),
        "reasoning_datasets_enabled": runtime.get("reasoning_datasets_enabled"),
        "reasoning_dataset_weights": runtime.get("reasoning_dataset_weights"),
        "task_bucket_mode": runtime.get("task_bucket_mode"),
        "bucket_sampling_counts": runtime.get("bucket_sampling_counts"),
        "answer_format": runtime.get("answer_format"),
        "is_resume_training": runtime.get("is_resume_training"),
        "resume_fitmotn_from": runtime.get("resume_fitmotn_from"),
        "resume_stage": runtime.get("resume_stage"),
        "stage2_only_on_resume": runtime.get("stage2_only_on_resume"),
        "extra_updates": runtime.get("extra_updates"),
        "actual_total_updates": runtime.get("actual_total_updates"),
        "stage_a_updates": runtime.get("stage_a_updates"),
        "stage_b_updates": runtime.get("stage_b_updates"),
        "approx_init_skipped_due_to_resume": runtime.get("approx_init_skipped_due_to_resume"),
        "loaded_fitmotn_state_path": runtime.get("loaded_fitmotn_state_path"),
        "current_model_structure_summary": runtime.get("current_model_structure_summary"),
    }
    record.update(runtime.get("loss_observability") or {})
    record.update(_current_stage_details(runtime))
    runtime["last_train_record"] = record
    runtime["train_records"].append(record)
    runtime["train_record_count"] = int(runtime.get("train_record_count", 0)) + 1
    tps = record.get("tokens_per_sec")
    if tps is not None:
        runtime["tokens_per_sec_sum"] = float(runtime.get("tokens_per_sec_sum", 0.0)) + float(tps)
        runtime["tokens_per_sec_count"] = int(runtime.get("tokens_per_sec_count", 0)) + 1
    peak_alloc = record.get("cuda_mem_peak_alloc_mb")
    if peak_alloc is not None:
        prev = runtime.get("max_cuda_mem_peak_alloc_mb")
        runtime["max_cuda_mem_peak_alloc_mb"] = float(peak_alloc) if prev is None else max(float(prev), float(peak_alloc))
    if record.get("train_loss") is not None:
        runtime["recent_train_losses"].append(float(record["train_loss"]))
    return record


def capture_usage_light_snapshot(runtime: Dict[str, Any], model: tc.nn.Module) -> Dict[str, Any]:
    snapshot = {
        "kind": "usage_light_snapshot",
        "time": time.time(),
        "run_name": runtime["run_name"],
        "seed": int(runtime.get("fit_cfg").train.seed) if runtime.get("fit_cfg") is not None else None,
        "stage": runtime.get("current_stage"),
        "update_step": runtime.get("scheduler_step"),
        "global_step": runtime.get("scheduler_step"),
        "lr": runtime.get("current_lr"),
        "T": runtime.get("current_T"),
        "gate_trainable": runtime.get("gate_trainable"),
    }
    snapshot.update(_current_stage_details(runtime))
    from ..patching import iter_patched_motn_layers

    for layer_i, module in iter_patched_motn_layers(model):
        layer_key = f"layer_{layer_i:02d}"
        layer_snapshot = {}
        for prefix, core in [
            ("gate", module.gate_proj.core),
            ("up", module.up_proj.core),
            ("down", module.down_proj.core),
        ]:
            stats = core.collect_runtime_usage_tensors() if hasattr(core, "collect_runtime_usage_tensors") else {}
            layer_snapshot[f"usage_{prefix}"] = _clone_usage_tensor(stats.get("expert_counts"))
            layer_snapshot[f"top1_{prefix}"] = _clone_usage_tensor(stats.get("top1_counts"))
        snapshot[layer_key] = layer_snapshot
    runtime["pending_usage_light_snapshots"].append(snapshot)
    return snapshot


def build_usage_light_record(runtime: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "kind": "usage_light",
        "time": snapshot.get("time", time.time()),
        "run_name": snapshot.get("run_name"),
        "seed": snapshot.get("seed"),
        "stage": snapshot.get("stage"),
        "update_step": snapshot.get("update_step"),
        "global_step": snapshot.get("global_step"),
        "lr": snapshot.get("lr"),
        "T": snapshot.get("T"),
        "gate_trainable": snapshot.get("gate_trainable"),
    }
    for key, value in _current_stage_details(runtime).items():
        record[key] = snapshot.get(key, value)
    for key, value in snapshot.items():
        if not str(key).startswith("layer_"):
            continue
        layer_record: Dict[str, Any] = {}
        for metric_key, metric_value in value.items():
            layer_record[metric_key] = to_jsonable(metric_value)
        record[key] = layer_record
    runtime["last_usage_light_record"] = record
    runtime["usage_light_records"].append(record)
    runtime["usage_light_record_count"] = int(runtime.get("usage_light_record_count", 0)) + 1
    return record


def flush_pending_usage_light_records(runtime: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    pending = runtime.get("pending_usage_light_snapshots")
    if pending is None:
        return records
    while pending:
        records.append(build_usage_light_record(runtime, pending.popleft()))
    return records


def build_usage_report_record(runtime: Dict[str, Any], model: tc.nn.Module) -> Dict[str, Any]:
    stage_plan = runtime.get("stage_plan")
    stage = None
    if stage_plan is not None:
        for stage_def in stage_plan.stages:
            if stage_def.name == runtime.get("current_stage"):
                stage = stage_def
                break
    record = {
        "kind": "usage",
        "time": time.time(),
        "run_name": runtime["run_name"],
        "seed": int(runtime.get("fit_cfg").train.seed) if runtime.get("fit_cfg") is not None else None,
        "stage": runtime.get("current_stage"),
        "update_step": runtime.get("scheduler_step"),
        "global_step": runtime.get("scheduler_step"),
        "lr": runtime.get("current_lr"),
        "T": runtime.get("current_T"),
        "gate_trainable": runtime.get("gate_trainable"),
        "reasoning_supervision_mode": runtime.get("reasoning_supervision_mode"),
        "stage_b_mode": runtime.get("stage_b_mode"),
        "reasoning_datasets_enabled": runtime.get("reasoning_datasets_enabled"),
        "reasoning_dataset_weights": runtime.get("reasoning_dataset_weights"),
        "task_bucket_mode": runtime.get("task_bucket_mode"),
        "bucket_sampling_counts": runtime.get("bucket_sampling_counts"),
        "answer_format": runtime.get("answer_format"),
    }
    record.update(_current_stage_details(runtime))
    from ..patching import iter_patched_motn_layers

    for layer_i, module in iter_patched_motn_layers(model):
        layer_key = f"layer_{layer_i:02d}"
        layer_record = {
            "usage_gate": None,
            "usage_up": None,
            "usage_down": None,
            "top1_gate": None,
            "top1_up": None,
            "top1_down": None,
            "top1_source_gate": None,
            "top1_source_up": None,
            "top1_source_down": None,
            "pos_gate": None,
            "pos_up": None,
            "pos_down": None,
            "pos_stats_gate": getattr(module.gate_proj.core, "pos_stats", None),
            "pos_stats_up": getattr(module.up_proj.core, "pos_stats", None),
            "pos_stats_down": getattr(module.down_proj.core, "pos_stats", None),
            "entropy_gate": None,
            "entropy_up": None,
            "entropy_down": None,
            "load_balance_gate": None,
            "load_balance_up": None,
            "load_balance_down": None,
            "importance_gate": None,
            "importance_up": None,
            "importance_down": None,
            "drop_rate_gate": None,
            "drop_rate_up": None,
            "drop_rate_down": None,
            "capacity_gate": None,
            "capacity_up": None,
            "capacity_down": None,
            "active_expert_count_gate": None,
            "active_expert_count_up": None,
            "active_expert_count_down": None,
            "max_expert_share_gate": None,
            "max_expert_share_up": None,
            "max_expert_share_down": None,
            "expert_cv_gate": None,
            "expert_cv_up": None,
            "expert_cv_down": None,
        }
        for prefix, core in [
            ("gate", module.gate_proj.core),
            ("up", module.up_proj.core),
            ("down", module.down_proj.core),
        ]:
            if hasattr(core, "materialize_usage_report"):
                stats = core.materialize_usage_report()
            else:
                stats = tensor_distribution_stats(getattr(core, "last_usage", None))
            layer_record[f"usage_{prefix}"] = stats.get("usage")
            top1_value = stats.get("top1")
            top1_source = "core.materialize_usage_report"
            if top1_value is None:
                top1_value, top1_source = _resolve_top1_value(core, stats.get("top1"))
            layer_record[f"top1_{prefix}"] = top1_value
            layer_record[f"top1_source_{prefix}"] = top1_source
            layer_record[f"pos_{prefix}"] = to_jsonable(getattr(core, "positions", None))
            layer_record[f"entropy_{prefix}"] = stats.get("entropy")
            layer_record[f"load_balance_{prefix}"] = stats.get("load_balance")
            layer_record[f"importance_{prefix}"] = stats.get("importance")
            layer_record[f"drop_rate_{prefix}"] = stats.get("drop_rate")
            layer_record[f"capacity_{prefix}"] = stats.get("capacity")
            layer_record[f"active_expert_count_{prefix}"] = stats.get("active_expert_count")
            layer_record[f"max_expert_share_{prefix}"] = stats.get("max_expert_share")
            layer_record[f"expert_cv_{prefix}"] = stats.get("expert_cv")
        record[layer_key] = layer_record
    runtime["last_usage_record"] = record
    runtime["usage_records"].append(record)
    runtime["usage_record_count"] = int(runtime.get("usage_record_count", 0)) + 1
    return record


def build_usage_record(runtime: Dict[str, Any], model: tc.nn.Module) -> Dict[str, Any]:
    return build_usage_report_record(runtime, model)


def build_mid_eval_record(runtime: Dict[str, Any], result: Dict[str, Any], vs_baseline: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "kind": "mid_eval",
        "time": time.time(),
        "run_name": runtime["run_name"],
        "seed": int(runtime.get("fit_cfg").train.seed) if runtime.get("fit_cfg") is not None else None,
        "stage": runtime.get("current_stage"),
        "update_step": runtime.get("scheduler_step"),
        "global_step": runtime.get("scheduler_step"),
        "lr": runtime.get("current_lr"),
        "T": runtime.get("current_T"),
        "gate_trainable": runtime.get("gate_trainable"),
        "samples_seen_estimate": runtime.get("samples_seen_estimate"),
        "tokens_seen_estimate": runtime.get("tokens_seen_estimate"),
        "resolved_model_dtype": runtime.get("resolved_model_dtype"),
        "amp_enabled": runtime.get("amp_enabled"),
        "reasoning_supervision_mode": runtime.get("reasoning_supervision_mode"),
        "stage_b_mode": runtime.get("stage_b_mode"),
        "reasoning_datasets_enabled": runtime.get("reasoning_datasets_enabled"),
        "reasoning_dataset_weights": runtime.get("reasoning_dataset_weights"),
        "answer_format": runtime.get("answer_format"),
        "result": result,
        "vs_baseline": vs_baseline,
    }
    record.update(_current_stage_details(runtime))
    runtime["latest_mid_eval_summary"] = to_jsonable(result.get("summary"))
    runtime["latest_mid_eval_update"] = record.get("update_step")
    runtime["mid_eval_records"].append(record)
    return record


def build_checkpoint_metadata(
    runtime: Dict[str, Any],
    *,
    layers_to_patch: List[int],
    motn_cfg: Dict[str, Any],
    patch_cfg: Optional[Dict[str, Any]] = None,
    fit_cfg,
    checkpoint_name: Optional[str] = None,
    state_dict: Optional[Dict[str, Any]] = None,
    patch_state_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stage_details = _current_stage_details(runtime)
    metadata = {
        "checkpoint_format": runtime.get("checkpoint_format", "patch_state_only_v2"),
        "run_name": runtime.get("run_name"),
        "checkpoint_name": checkpoint_name or "final_model",
        "save_time": time.time(),
        "base_model_path": fit_cfg.model.model_path,
        "tokenizer_path": fit_cfg.model.model_path,
        "seed": int(fit_cfg.train.seed),
        "layers_to_patch": list(layers_to_patch),
        "patched_layer_count": len(layers_to_patch),
        "patched_layer_indices": list(layers_to_patch),
        "patch_backend": runtime.get("patch_backend", (patch_cfg or motn_cfg).get("patch_backend", "motn")),
        "patch_cfg": patch_cfg or runtime.get("patch_cfg") or motn_cfg,
        "motn_cfg": motn_cfg,
        "gate_arch": runtime.get("gate_arch"),
        "gate_hidden_dim": runtime.get("gate_hidden_dim"),
        "resolved_gate_hidden_dim": runtime.get("resolved_gate_hidden_dim"),
        "gate_activation": runtime.get("gate_activation"),
        "gate_norm": runtime.get("gate_norm"),
        "gate_dropout": runtime.get("gate_dropout"),
        "gate_mlp_bias": runtime.get("gate_mlp_bias"),
        "gate_output_init_std": runtime.get("gate_output_init_std"),
        "gate_residual_delta_scale": runtime.get("gate_residual_delta_scale"),
        "router_param_count": runtime.get("router_param_count"),
        "router_param_ratio_vs_blocks": runtime.get("router_param_ratio_vs_blocks"),
        "gate_router_summary": runtime.get("gate_router_summary"),
        "fit_cfg": fit_cfg,
        "resolved_model_dtype": runtime.get("resolved_model_dtype"),
        "amp_enabled": runtime.get("amp_enabled"),
        "updates_done": runtime.get("scheduler_step"),
        "global_step": runtime.get("scheduler_step"),
        "current_stage": runtime.get("current_stage"),
        "stage_plan": runtime.get("stage_plan"),
        "gate_trainable": runtime.get("gate_trainable"),
        "current_T": runtime.get("current_T"),
        "current_temperature": runtime.get("current_temperature"),
        "warmup_updates": runtime.get("warmup_updates"),
        "lr_scheduler_type": runtime.get("lr_scheduler_type"),
        "lr_warmup": runtime.get("lr_warmup"),
        "lr_warmup_steps": runtime.get("lr_warmup_steps"),
        "resolved_lr_warmup_steps": runtime.get("resolved_lr_warmup_steps"),
        "lr_decay_steps": runtime.get("lr_decay_steps"),
        "resolved_lr_decay_steps": runtime.get("resolved_lr_decay_steps"),
        "actual_training_steps": runtime.get("actual_training_steps"),
        "scheduler_total_steps": runtime.get("scheduler_total_steps"),
        "scheduler_state_summary": runtime.get("scheduler_state_summary"),
        "current_lr": runtime.get("current_lr"),
        "current_param_group_lrs": runtime.get("current_param_group_lrs"),
        "stage_specific_lr_enabled": runtime.get("stage_specific_lr_enabled"),
        "stage_a_block_lr": runtime.get("stage_a_block_lr"),
        "stage_a_router_lr": runtime.get("stage_a_router_lr"),
        "stage_b_block_lr": runtime.get("stage_b_block_lr"),
        "stage_b_router_lr": runtime.get("stage_b_router_lr"),
        "resolved_stage_a_block_lr": runtime.get("resolved_stage_a_block_lr"),
        "resolved_stage_a_router_lr": runtime.get("resolved_stage_a_router_lr"),
        "resolved_stage_b_block_lr": runtime.get("resolved_stage_b_block_lr"),
        "resolved_stage_b_router_lr": runtime.get("resolved_stage_b_router_lr"),
        "temperature_schedule_type": runtime.get("temperature_schedule_type"),
        "stage_a_begin_t": runtime.get("stage_a_begin_t"),
        "stage_a_end_t": runtime.get("stage_a_end_t"),
        "stage_b_begin_t": runtime.get("stage_b_begin_t"),
        "stage_b_end_t": runtime.get("stage_b_end_t"),
        "final_answer_weight_enabled": runtime.get("final_answer_weight_enabled"),
        "final_answer_weight": runtime.get("final_answer_weight"),
        "final_answer_marker": runtime.get("final_answer_marker"),
        "final_answer_weighted_tokens": runtime.get("final_answer_weighted_tokens"),
        "final_answer_loss": runtime.get("final_answer_loss"),
        "loss_observability": runtime.get("loss_observability"),
        "optimizer_param_group_count": runtime.get("optimizer_param_group_count"),
        "optimizer_param_groups": runtime.get("optimizer_param_groups"),
        "optimizer_fallback_param_names": runtime.get("optimizer_fallback_param_names"),
        "trainable_params": runtime.get("trainable_params"),
        "total_params": runtime.get("total_params"),
        "trainable_ratio": runtime.get("trainable_ratio"),
        "baseline_small_summary": runtime.get("baseline_small_summary"),
        "baseline_final_summary": runtime.get("baseline_final_summary"),
        "latest_mid_eval_summary": runtime.get("latest_mid_eval_summary"),
        "final_full_summary": runtime.get("final_full_summary"),
        "compare_vs_baseline": runtime.get("compare_vs_baseline"),
        "approx_init_summary": runtime.get("approx_init_summary"),
        "is_resume_training": runtime.get("is_resume_training"),
        "resume_fitmotn_from": runtime.get("resume_fitmotn_from"),
        "resume_stage": runtime.get("resume_stage"),
        "stage2_only_on_resume": runtime.get("stage2_only_on_resume"),
        "extra_updates": runtime.get("extra_updates"),
        "actual_total_updates": runtime.get("actual_total_updates"),
        "stage_a_updates": runtime.get("stage_a_updates"),
        "stage_b_updates": runtime.get("stage_b_updates"),
        "approx_init_skipped_due_to_resume": runtime.get("approx_init_skipped_due_to_resume"),
        "loaded_fitmotn_state_path": runtime.get("loaded_fitmotn_state_path"),
        "loaded_checkpoint_metadata": runtime.get("loaded_checkpoint_metadata"),
        "resume_load_summary": runtime.get("resume_load_summary"),
        "current_model_structure_summary": runtime.get("current_model_structure_summary"),
        "expert_warmup_scaling_summary": runtime.get("expert_warmup_scaling_summary"),
        "env_snapshot": runtime.get("env_snapshot"),
        "reasoning_supervision_mode": runtime.get("reasoning_supervision_mode"),
        "stage_b_mode": runtime.get("stage_b_mode"),
        "stage_b_disable_pretrain": runtime.get("stage_b_disable_pretrain"),
        "stage_b_reasoning_boost": runtime.get("stage_b_reasoning_boost"),
        "reasoning_datasets_enabled": runtime.get("reasoning_datasets_enabled"),
        "reasoning_dataset_weights": runtime.get("reasoning_dataset_weights"),
        "answer_format": runtime.get("answer_format"),
        "runtime": {
            "tokens_per_microbatch": runtime.get("tokens_per_microbatch"),
            "tokens_per_update": runtime.get("tokens_per_update"),
        },
    }
    metadata.update(stage_details)
    if state_dict is not None:
        metadata["state_dict"] = state_dict
    if patch_state_dict is not None:
        metadata["patch_state_dict"] = patch_state_dict
    return metadata


def build_run_summary(runtime: Dict[str, Any], eval_summary: Dict[str, Any], *, final_model_dir: str, stop_reason: Optional[str] = None) -> Dict[str, Any]:
    stage_details = _current_stage_details(runtime)
    total_wall = max(0.0, float(time.time() - float(runtime.get("start_time", time.time()))))
    recent_losses = list(runtime.get("recent_train_losses", []))
    last_k_losses = recent_losses[-10:]
    final_train_loss = last_k_losses[-1] if last_k_losses else None
    mean_last_k = sum(last_k_losses) / len(last_k_losses) if last_k_losses else None
    tps_count = int(runtime.get("tokens_per_sec_count", 0))
    avg_tokens_per_sec = None
    if tps_count > 0:
        avg_tokens_per_sec = float(runtime.get("tokens_per_sec_sum", 0.0)) / float(tps_count)
    max_cuda_peak = runtime.get("max_cuda_mem_peak_alloc_mb")

    best_mid_update = None
    best_mid_gsm8k = None
    best_mid_mmlu = None
    best_mid_score = None
    for rec in runtime.get("mid_eval_records", []):
        result = rec.get("result", {}) or {}
        tasks = result.get("tasks", {}) or {}
        gsm = tasks.get("gsm8k", {}).get("primary_score")
        mmlu = tasks.get("mmlu", {}).get("primary_score")
        if gsm is None and mmlu is None:
            continue
        score = 0.0
        count = 0
        if gsm is not None:
            score += float(gsm)
            count += 1
        if mmlu is not None:
            score += float(mmlu)
            count += 1
        if count == 0:
            continue
        score = score / count
        if best_mid_score is None or score > best_mid_score:
            best_mid_score = score
            best_mid_update = rec.get("update_step")
            best_mid_gsm8k = gsm
            best_mid_mmlu = mmlu

    summary = {
        "run_name": runtime["run_name"],
        "run_dir": runtime.get("run_dir"),
        "final_model_dir": final_model_dir,
        "total_updates_planned": runtime.get("total_updates_planned"),
        "total_updates_done": runtime.get("scheduler_step"),
        "stopped_early": bool(stop_reason),
        "stop_reason": stop_reason,
        "total_wall_time_sec": total_wall,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "mean_tokens_per_sec": avg_tokens_per_sec,
        "final_stage": runtime.get("current_stage"),
        "resolved_model_dtype": runtime.get("resolved_model_dtype"),
        "amp_enabled": runtime.get("amp_enabled"),
        "seed": int(runtime.get("fit_cfg").train.seed) if runtime.get("fit_cfg") is not None else None,
        "warmup_updates": runtime.get("warmup_updates"),
        "lr_scheduler_type": runtime.get("lr_scheduler_type"),
        "lr_warmup": runtime.get("lr_warmup"),
        "lr_warmup_steps": runtime.get("lr_warmup_steps"),
        "resolved_lr_warmup_steps": runtime.get("resolved_lr_warmup_steps"),
        "lr_decay_steps": runtime.get("lr_decay_steps"),
        "resolved_lr_decay_steps": runtime.get("resolved_lr_decay_steps"),
        "actual_training_steps": runtime.get("actual_training_steps"),
        "scheduler_total_steps": runtime.get("scheduler_total_steps"),
        "checkpoint_format": runtime.get("checkpoint_format"),
        "layers_to_patch": runtime.get("layers_to_patch"),
        "patched_layer_count": runtime.get("patched_layer_count"),
        "patch_backend": runtime.get("patch_backend"),
        "patch_cfg": runtime.get("patch_cfg"),
        "gate_arch": runtime.get("gate_arch"),
        "gate_hidden_dim": runtime.get("gate_hidden_dim"),
        "resolved_gate_hidden_dim": runtime.get("resolved_gate_hidden_dim"),
        "gate_activation": runtime.get("gate_activation"),
        "gate_norm": runtime.get("gate_norm"),
        "gate_dropout": runtime.get("gate_dropout"),
        "gate_mlp_bias": runtime.get("gate_mlp_bias"),
        "gate_output_init_std": runtime.get("gate_output_init_std"),
        "gate_residual_delta_scale": runtime.get("gate_residual_delta_scale"),
        "router_param_count": runtime.get("router_param_count"),
        "router_param_ratio_vs_blocks": runtime.get("router_param_ratio_vs_blocks"),
        "gate_router_summary": runtime.get("gate_router_summary"),
        "trainable_params": runtime.get("trainable_params"),
        "total_params": runtime.get("total_params"),
        "trainable_ratio": runtime.get("trainable_ratio"),
        "baseline_small_summary": None if eval_summary.get("baseline_small") is None else eval_summary.get("baseline_small", {}).get("summary"),
        "baseline_final_summary": None if eval_summary.get("baseline_final") is None else eval_summary.get("baseline_final", {}).get("summary"),
        "latest_mid_eval_summary": runtime.get("latest_mid_eval_summary"),
        "final_full_summary": None if eval_summary.get("final_full") is None else eval_summary.get("final_full", {}).get("summary"),
        "compare_vs_baseline": eval_summary.get("compare_vs_baseline"),
        "approx_init_summary": runtime.get("approx_init_summary"),
        "is_resume_training": runtime.get("is_resume_training"),
        "resume_fitmotn_from": runtime.get("resume_fitmotn_from"),
        "resume_stage": runtime.get("resume_stage"),
        "stage2_only_on_resume": runtime.get("stage2_only_on_resume"),
        "extra_updates": runtime.get("extra_updates"),
        "actual_total_updates": runtime.get("actual_total_updates"),
        "stage_a_updates": runtime.get("stage_a_updates"),
        "stage_b_updates": runtime.get("stage_b_updates"),
        "approx_init_skipped_due_to_resume": runtime.get("approx_init_skipped_due_to_resume"),
        "loaded_fitmotn_state_path": runtime.get("loaded_fitmotn_state_path"),
        "loaded_checkpoint_metadata": runtime.get("loaded_checkpoint_metadata"),
        "resume_load_summary": runtime.get("resume_load_summary"),
        "current_model_structure_summary": runtime.get("current_model_structure_summary"),
        "expert_warmup_scaling_summary": runtime.get("expert_warmup_scaling_summary"),
        "best_mid_eval_update": best_mid_update,
        "best_mid_eval_gsm8k": best_mid_gsm8k,
        "best_mid_eval_mmlu": best_mid_mmlu,
        "max_cuda_mem_peak_mb": max_cuda_peak,
        "mean_train_loss_last_k": mean_last_k,
        "final_train_loss": final_train_loss,
        "scheduler_state_summary": runtime.get("scheduler_state_summary"),
        "current_lr": runtime.get("current_lr"),
        "current_param_group_lrs": runtime.get("current_param_group_lrs"),
        "current_temperature": runtime.get("current_temperature"),
        "stage_specific_lr_enabled": runtime.get("stage_specific_lr_enabled"),
        "stage_a_block_lr": runtime.get("stage_a_block_lr"),
        "stage_a_router_lr": runtime.get("stage_a_router_lr"),
        "stage_b_block_lr": runtime.get("stage_b_block_lr"),
        "stage_b_router_lr": runtime.get("stage_b_router_lr"),
        "resolved_stage_a_block_lr": runtime.get("resolved_stage_a_block_lr"),
        "resolved_stage_a_router_lr": runtime.get("resolved_stage_a_router_lr"),
        "resolved_stage_b_block_lr": runtime.get("resolved_stage_b_block_lr"),
        "resolved_stage_b_router_lr": runtime.get("resolved_stage_b_router_lr"),
        "temperature_schedule_type": runtime.get("temperature_schedule_type"),
        "stage_a_begin_t": runtime.get("stage_a_begin_t"),
        "stage_a_end_t": runtime.get("stage_a_end_t"),
        "stage_b_begin_t": runtime.get("stage_b_begin_t"),
        "stage_b_end_t": runtime.get("stage_b_end_t"),
        "final_answer_weight_enabled": runtime.get("final_answer_weight_enabled"),
        "final_answer_weight": runtime.get("final_answer_weight"),
        "final_answer_marker": runtime.get("final_answer_marker"),
        "final_answer_weighted_tokens": runtime.get("final_answer_weighted_tokens"),
        "final_answer_loss": runtime.get("final_answer_loss"),
        "loss_observability": runtime.get("loss_observability"),
        "optimizer_type": runtime.get("optimizer_type"),
        "optimizer_param_group_count": runtime.get("optimizer_param_group_count"),
        "optimizer_param_groups": runtime.get("optimizer_param_groups"),
        "optimizer_fallback_param_names": runtime.get("optimizer_fallback_param_names"),
        "env_snapshot": runtime.get("env_snapshot"),
        "reasoning_supervision_mode": runtime.get("reasoning_supervision_mode"),
        "stage_b_mode": runtime.get("stage_b_mode"),
        "stage_b_disable_pretrain": runtime.get("stage_b_disable_pretrain"),
        "stage_b_reasoning_boost": runtime.get("stage_b_reasoning_boost"),
        "reasoning_datasets_enabled": runtime.get("reasoning_datasets_enabled"),
        "reasoning_dataset_weights": runtime.get("reasoning_dataset_weights"),
        "answer_format": runtime.get("answer_format"),
    }
    summary.update(stage_details)
    return summary
