from __future__ import annotations

import math
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List

from ..config.schema import TrainConfig


@dataclass
class StageDefinition:
    name: str
    start_update: int
    end_update: int
    pretrain_ratio: float
    task_ratio: float
    mode: str = "mixed"
    reasoning_focused: bool = False
    pretrain_disabled: bool = False
    reasoning_boost: float = 1.0
    task_bucket_mode: str = "flat"
    bucket_ratios: Dict[str, float] = field(default_factory=dict)

    @property
    def updates(self) -> int:
        return max(0, self.end_update - self.start_update)


@dataclass
class StagePlan:
    total_updates: int
    stage_a_updates: int = 0
    stage_b_updates: int = 0
    stages: List[StageDefinition] = field(default_factory=list)

    def stage_for_step(self, global_step: int) -> StageDefinition:
        step = max(0, int(global_step))
        for stage in self.stages:
            if stage.start_update <= step < stage.end_update:
                return stage
        return self.stages[-1]


class MutableStageState:
    def __init__(self, plan: StagePlan):
        self.plan = plan
        self._lock = Lock()
        self._global_step = 0

    @property
    def global_step(self) -> int:
        with self._lock:
            return self._global_step

    def set_global_step(self, step: int) -> None:
        with self._lock:
            self._global_step = int(step)

    def current_stage(self) -> StageDefinition:
        with self._lock:
            return self.plan.stage_for_step(self._global_step)


def infer_total_updates(cfg: TrainConfig) -> int:
    if (
        getattr(cfg, "resume_fitmotn_from", None)
        and not getattr(cfg, "resume_checkpoint_from", None)
        and getattr(cfg, "extra_updates", None) is not None
    ):
        return max(1, int(cfg.extra_updates))
    if cfg.epochs and cfg.epochs > 0:
        steps_per_epoch = int(cfg.epoch_samples // max(1, cfg.batch_size * cfg.grad_accum))
        return max(1, int(round(cfg.epochs * steps_per_epoch)))
    return max(1, int(cfg.steps))


def build_stage_plan(cfg: TrainConfig) -> StagePlan:
    total_updates = infer_total_updates(cfg)
    task_bucket_mode = str(getattr(cfg, "task_bucket_mode", "flat")).strip().lower()
    resume_stage = str(getattr(cfg, "resume_stage", "auto") or "auto").strip().lower()
    stage2_only_resume = (
        bool(getattr(cfg, "resume_fitmotn_from", None))
        and not bool(getattr(cfg, "resume_checkpoint_from", None))
        and bool(getattr(cfg, "stage2_only_on_resume", True))
        and resume_stage in {"auto", "stage_b"}
    )
    if stage2_only_resume:
        stage_a_updates = 0
        stage_b_updates = total_updates
        stages: List[StageDefinition] = []
    else:
        stage_a_updates = max(1, int(round(total_updates * float(cfg.stage_a_ratio))))
        stage_b_updates = max(0, total_updates - stage_a_updates)
        stages = [
            StageDefinition(
                name="stage_a_recover",
                start_update=0,
                end_update=stage_a_updates,
                pretrain_ratio=float(cfg.stage_a_pretrain_ratio),
                task_ratio=float(cfg.stage_a_task_ratio),
                task_bucket_mode=task_bucket_mode,
                bucket_ratios=_resolve_bucket_ratios(
                    pretrain_ratio=float(cfg.stage_a_pretrain_ratio),
                    task_ratio=float(cfg.stage_a_task_ratio),
                    core_ratio=float(getattr(cfg, "stage_a_core_task_ratio", 0.0)),
                    aux_ratio=float(getattr(cfg, "stage_a_aux_task_ratio", 0.0)),
                    task_bucket_mode=task_bucket_mode,
                ),
            )
        ]
    if stage_b_updates > 0:
        stage_b_mode = str(getattr(cfg, "stage_b_mode", "mixed"))
        pretrain_disabled = bool(getattr(cfg, "stage_b_disable_pretrain", False))
        reasoning_focused = stage_b_mode == "reasoning_recovery"
        pretrain_ratio = 0.0 if pretrain_disabled else float(cfg.stage_b_pretrain_ratio)
        task_ratio = 1.0 if pretrain_disabled else float(cfg.stage_b_task_ratio)
        stages.append(
            StageDefinition(
                name="stage_b_reasoning_recovery" if reasoning_focused else "stage_b_taskaware",
                start_update=stage_a_updates,
                end_update=stage_a_updates + stage_b_updates,
                pretrain_ratio=pretrain_ratio,
                task_ratio=task_ratio,
                mode=stage_b_mode,
                reasoning_focused=reasoning_focused,
                pretrain_disabled=pretrain_disabled,
                reasoning_boost=float(getattr(cfg, "stage_b_reasoning_boost", 1.0)),
                task_bucket_mode=task_bucket_mode,
                bucket_ratios=_resolve_bucket_ratios(
                    pretrain_ratio=pretrain_ratio,
                    task_ratio=task_ratio,
                    core_ratio=float(getattr(cfg, "stage_b_core_task_ratio", 0.0)),
                    aux_ratio=float(getattr(cfg, "stage_b_aux_task_ratio", 0.0)),
                    task_bucket_mode=task_bucket_mode,
                ),
            )
        )
    return StagePlan(total_updates=total_updates, stage_a_updates=stage_a_updates, stage_b_updates=stage_b_updates, stages=stages)


def _resolve_bucket_ratios(
    *,
    pretrain_ratio: float,
    task_ratio: float,
    core_ratio: float,
    aux_ratio: float,
    task_bucket_mode: str,
) -> Dict[str, float]:
    mode = str(task_bucket_mode or "flat").strip().lower()
    if mode == "bucketed":
        return {
            "pretrain_general": float(pretrain_ratio),
            "gsm8k_core": float(core_ratio),
            "aux_reasoning": float(aux_ratio),
        }
    return {
        "pretrain_general": float(pretrain_ratio),
        "task": float(task_ratio),
    }


def _stage_prefix(stage_name: str) -> str:
    name = str(stage_name or "").lower()
    return "stage_b" if name.startswith("stage_b") else "stage_a"


def stage_specific_lr_enabled(cfg: TrainConfig) -> bool:
    return any(
        getattr(cfg, field_name, None) is not None
        for field_name in ("stage_a_block_lr", "stage_a_router_lr", "stage_b_block_lr", "stage_b_router_lr")
    )


def resolve_stage_lrs(cfg: TrainConfig, stage_name: str) -> Dict[str, float]:
    prefix = _stage_prefix(stage_name)
    base_lr = float(getattr(cfg, "lr"))
    block_lr = getattr(cfg, "block_lr", None)
    router_lr = getattr(cfg, "router_lr", None)
    resolved_block_lr = base_lr if block_lr is None else float(block_lr)
    resolved_router_lr = base_lr if router_lr is None else float(router_lr)
    stage_block_lr = getattr(cfg, f"{prefix}_block_lr", None)
    stage_router_lr = getattr(cfg, f"{prefix}_router_lr", None)
    return {
        "blocks": resolved_block_lr if stage_block_lr is None else float(stage_block_lr),
        "router": resolved_router_lr if stage_router_lr is None else float(stage_router_lr),
    }


def resolved_stage_lr_summary(cfg: TrainConfig) -> Dict[str, float | bool | None]:
    stage_a = resolve_stage_lrs(cfg, "stage_a_recover")
    stage_b = resolve_stage_lrs(cfg, "stage_b_taskaware")
    return {
        "stage_specific_lr_enabled": stage_specific_lr_enabled(cfg),
        "stage_a_block_lr": getattr(cfg, "stage_a_block_lr", None),
        "stage_a_router_lr": getattr(cfg, "stage_a_router_lr", None),
        "stage_b_block_lr": getattr(cfg, "stage_b_block_lr", None),
        "stage_b_router_lr": getattr(cfg, "stage_b_router_lr", None),
        "resolved_stage_a_block_lr": stage_a["blocks"],
        "resolved_stage_a_router_lr": stage_a["router"],
        "resolved_stage_b_block_lr": stage_b["blocks"],
        "resolved_stage_b_router_lr": stage_b["router"],
    }


def set_optimizer_stage_lrs(optimizer, cfg: TrainConfig, stage_name: str) -> Dict[str, float]:
    resolved = resolve_stage_lrs(cfg, stage_name)
    if optimizer is None:
        return resolved
    for idx, group in enumerate(getattr(optimizer, "param_groups", [])):
        name = str(group.get("name") or f"group_{idx}")
        if name in resolved:
            group["lr"] = float(resolved[name])
            group["initial_lr"] = float(resolved[name])
    return resolved


def resolve_stage_temperature_bounds(cfg: TrainConfig, stage_name: str) -> tuple[float, float]:
    prefix = _stage_prefix(stage_name)
    stage_begin = getattr(cfg, f"{prefix}_begin_t", None)
    stage_end = getattr(cfg, f"{prefix}_end_t", None)
    t_start = float(getattr(cfg, "begin_t")) if stage_begin is None else float(stage_begin)
    t_end = float(getattr(cfg, "end_t")) if stage_end is None else float(stage_end)
    return t_start, t_end


def resolve_stage_temperature(cfg: TrainConfig, stage, global_step: int) -> float:
    schedule_type = str(getattr(cfg, "temperature_schedule_type", "cosine") or "cosine").strip().lower()
    if schedule_type == "constant":
        prefix = _stage_prefix(stage.name)
        stage_begin = getattr(cfg, f"{prefix}_begin_t", None)
        stage_end = getattr(cfg, f"{prefix}_end_t", None)
        if stage_begin is not None:
            return float(stage_begin)
        if stage_end is not None:
            return float(stage_end)
        begin_t = getattr(cfg, "begin_t", None)
        if begin_t is not None:
            return float(begin_t)
        return float(getattr(cfg, "end_t"))
    t_start, t_end = resolve_stage_temperature_bounds(cfg, stage.name)
    local_step = max(0, int(global_step) - int(stage.start_update))
    total_steps = max(1, int(stage.updates))
    return temperature_schedule(
        local_step,
        t_start=t_start,
        t_end=t_end,
        total_steps=total_steps,
        hold_steps=int(total_steps * 0.10),
        decay_steps=int(total_steps * 0.70),
        mode="cosine",
    )


def temperature_schedule(
    step: int,
    *,
    t_start: float,
    t_end: float,
    total_steps: int,
    hold_steps: int,
    decay_steps: int,
    mode: str = "cosine",
) -> float:
    step = int(step)
    total_steps = max(1, int(total_steps))
    hold_steps = max(0, int(hold_steps))
    decay_steps = max(0, int(decay_steps))
    if hold_steps + decay_steps > total_steps:
        decay_steps = max(0, total_steps - hold_steps)
    if step >= total_steps:
        return float(t_end)
    if step < hold_steps:
        return float(t_start)
    if decay_steps == 0 or step >= hold_steps + decay_steps:
        return float(t_end)
    ratio = (step - hold_steps) / max(1, decay_steps)
    if mode == "linear":
        return float(t_start + ratio * (t_end - t_start))
    if mode == "cosine":
        return float(t_end + 0.5 * (t_start - t_end) * (1 + math.cos(math.pi * ratio)))
    raise ValueError(f"Unknown mode: {mode}")
