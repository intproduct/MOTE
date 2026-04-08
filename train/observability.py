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

TRAIN_RECORD_BUFFER_SIZE = 128
USAGE_RECORD_BUFFER_SIZE = 32
RECENT_LOSS_BUFFER_SIZE = 64
REASONING_DATASET_FLAGS = {
    "gsm8k": "use_gsm8k_train",
    "gsm8k_socratic": "use_gsm8k_socratic_train",
    "svamp": "use_svamp_train",
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
    "metamath": "wt_metamath",
    "hendrycks_math": "wt_math",
    "mmlu": "wt_mmlu",
    "openr1_math": "wt_openr1_math",
    "numinamath_cot": "wt_numinamath_cot",
    "openthoughts_math": "wt_openthoughts_math",
    "bespoke_stratos": "wt_bespoke_stratos",
}


@dataclass
class UpdateBatchMeta:
    batch_task_names: Set[str] = field(default_factory=set)
    batch_groups: Set[str] = field(default_factory=set)
    batch_source_families: Set[str] = field(default_factory=set)
    microbatch_count: int = 0

    def clear(self) -> None:
        self.batch_task_names.clear()
        self.batch_groups.clear()
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
                "stage_mode": getattr(stage, "mode", None),
                "stage_reasoning_focused": getattr(stage, "reasoning_focused", None),
                "stage_pretrain_disabled": getattr(stage, "pretrain_disabled", None),
                "stage_reasoning_boost": getattr(stage, "reasoning_boost", None),
            }
    return {
        "stage_pretrain_ratio": None,
        "stage_task_ratio": None,
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
    warmup_updates: int,
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
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "start_time": time.time(),
        "fit_cfg": fit_cfg,
        "stage_plan": stage_plan,
        "resolved_model_dtype": str(model_dtype).replace("torch.", ""),
        "amp_enabled": bool(amp_enabled),
        "warmup_updates": int(warmup_updates),
        "total_updates_planned": total_updates,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "tokens_per_microbatch": tokens_per_microbatch,
        "tokens_per_update": tokens_per_update,
        "samples_seen_estimate": 0,
        "tokens_seen_estimate": 0,
        "current_T": None,
        "gate_trainable": None,
        "current_stage": None,
        "current_stage_update": None,
        "current_lr": None,
        "current_loss": None,
        "current_epoch": None,
        "optimizer_type": None,
        "weight_decay": None,
        "scheduler_step": 0,
        "lr_scheduler_enabled": bool(fit_cfg.train.lr_warmup),
        "loss_scale": None,
        "overflow_or_nan_detected": None,
        "num_nan_grads": None,
        "num_inf_grads": None,
        "optimizer_step_skipped": None,
        "trainable_params": int(param_snapshot["trainable_params"]),
        "total_params": int(param_snapshot["total_params"]),
        "trainable_ratio": param_snapshot["trainable_ratio"],
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
            "microbatch_count": None,
        },
        "last_train_record": None,
        "last_usage_record": None,
        "latest_mid_eval_summary": None,
        "final_full_summary": None,
        "compare_vs_baseline": None,
        "approx_init_summary": None,
        "expert_warmup_scaling_summary": None,
        "baseline_small_summary": None,
        "baseline_final_summary": None,
        "scheduler_state_summary": None,
        "checkpoint_format": "patch_state_only_v2",
        "train_records": deque(maxlen=TRAIN_RECORD_BUFFER_SIZE),
        "usage_records": deque(maxlen=USAGE_RECORD_BUFFER_SIZE),
        "mid_eval_records": [],
        "train_record_count": 0,
        "usage_record_count": 0,
        "tokens_per_sec_sum": 0.0,
        "tokens_per_sec_count": 0,
        "max_cuda_mem_peak_alloc_mb": None,
        "recent_train_losses": deque(maxlen=RECENT_LOSS_BUFFER_SIZE),
        "latest_mid_eval_update": None,
        "reasoning_supervision_mode": reasoning_summary["reasoning_supervision_mode"],
        "reasoning_datasets_enabled": reasoning_summary["reasoning_datasets_enabled"],
        "reasoning_dataset_weights": reasoning_summary["reasoning_dataset_weights"],
        "stage_b_mode": reasoning_summary["stage_b_mode"],
        "stage_b_disable_pretrain": reasoning_summary["stage_b_disable_pretrain"],
        "stage_b_reasoning_boost": reasoning_summary["stage_b_reasoning_boost"],
        "answer_format": reasoning_summary["answer_format"],
    }


def register_microbatch(runtime: Dict[str, Any], inputs: Dict[str, Any]) -> None:
    meta: UpdateBatchMeta = runtime["update_batch_meta"]
    for key, target in [
        ("task", meta.batch_task_names),
        ("group", meta.batch_groups),
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


def update_step_runtime(runtime: Dict[str, Any], *, global_step: int, stage_name: str, stage_start: int, current_t: float, gate_trainable: bool, lr: Optional[float], epoch: Optional[float], loss: Optional[float], now: Optional[float] = None) -> None:
    runtime["scheduler_step"] = int(global_step)
    runtime["current_stage"] = stage_name
    runtime["current_stage_update"] = derive_stage_step(global_step, stage_start)
    runtime["gate_trainable"] = bool(gate_trainable)
    runtime["current_T"] = float(current_t)
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
        "batch_source_families": batch_meta.get("batch_source_families"),
        "stage_pretrain_ratio": None,
        "stage_task_ratio": None,
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
        "warmup_updates": runtime.get("warmup_updates"),
        "scheduler_step": runtime.get("scheduler_step"),
        "optimizer_type": runtime.get("optimizer_type"),
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
        "answer_format": runtime.get("answer_format"),
    }
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


def build_usage_record(runtime: Dict[str, Any], model: tc.nn.Module) -> Dict[str, Any]:
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
            stats = tensor_distribution_stats(getattr(core, "last_usage", None))
            layer_record[f"usage_{prefix}"] = stats["usage"]
            top1_value, top1_source = _resolve_top1_value(core, stats["top1"])
            layer_record[f"top1_{prefix}"] = top1_value
            layer_record[f"top1_source_{prefix}"] = top1_source
            layer_record[f"pos_{prefix}"] = to_jsonable(getattr(core, "positions", None))
            layer_record[f"entropy_{prefix}"] = stats["entropy"]
            layer_record[f"load_balance_{prefix}"] = stats["load_balance"]
            layer_record[f"active_expert_count_{prefix}"] = stats["active_expert_count"]
            layer_record[f"max_expert_share_{prefix}"] = stats["max_expert_share"]
            layer_record[f"expert_cv_{prefix}"] = stats["expert_cv"]
        record[layer_key] = layer_record
    runtime["last_usage_record"] = record
    runtime["usage_records"].append(record)
    runtime["usage_record_count"] = int(runtime.get("usage_record_count", 0)) + 1
    return record


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
        "motn_cfg": motn_cfg,
        "fit_cfg": fit_cfg,
        "resolved_model_dtype": runtime.get("resolved_model_dtype"),
        "amp_enabled": runtime.get("amp_enabled"),
        "updates_done": runtime.get("scheduler_step"),
        "global_step": runtime.get("scheduler_step"),
        "current_stage": runtime.get("current_stage"),
        "stage_plan": runtime.get("stage_plan"),
        "gate_trainable": runtime.get("gate_trainable"),
        "current_T": runtime.get("current_T"),
        "warmup_updates": runtime.get("warmup_updates"),
        "scheduler_state_summary": runtime.get("scheduler_state_summary"),
        "trainable_params": runtime.get("trainable_params"),
        "total_params": runtime.get("total_params"),
        "trainable_ratio": runtime.get("trainable_ratio"),
        "baseline_small_summary": runtime.get("baseline_small_summary"),
        "baseline_final_summary": runtime.get("baseline_final_summary"),
        "latest_mid_eval_summary": runtime.get("latest_mid_eval_summary"),
        "final_full_summary": runtime.get("final_full_summary"),
        "compare_vs_baseline": runtime.get("compare_vs_baseline"),
        "approx_init_summary": runtime.get("approx_init_summary"),
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
        "checkpoint_format": runtime.get("checkpoint_format"),
        "layers_to_patch": runtime.get("layers_to_patch"),
        "patched_layer_count": runtime.get("patched_layer_count"),
        "trainable_params": runtime.get("trainable_params"),
        "total_params": runtime.get("total_params"),
        "trainable_ratio": runtime.get("trainable_ratio"),
        "baseline_small_summary": None if eval_summary.get("baseline_small") is None else eval_summary.get("baseline_small", {}).get("summary"),
        "baseline_final_summary": None if eval_summary.get("baseline_final") is None else eval_summary.get("baseline_final", {}).get("summary"),
        "latest_mid_eval_summary": runtime.get("latest_mid_eval_summary"),
        "final_full_summary": None if eval_summary.get("final_full") is None else eval_summary.get("final_full", {}).get("summary"),
        "compare_vs_baseline": eval_summary.get("compare_vs_baseline"),
        "approx_init_summary": runtime.get("approx_init_summary"),
        "expert_warmup_scaling_summary": runtime.get("expert_warmup_scaling_summary"),
        "best_mid_eval_update": best_mid_update,
        "best_mid_eval_gsm8k": best_mid_gsm8k,
        "best_mid_eval_mmlu": best_mid_mmlu,
        "max_cuda_mem_peak_mb": max_cuda_peak,
        "mean_train_loss_last_k": mean_last_k,
        "final_train_loss": final_train_loss,
        "scheduler_state_summary": runtime.get("scheduler_state_summary"),
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
