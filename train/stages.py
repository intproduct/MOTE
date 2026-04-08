from __future__ import annotations

import math
from dataclasses import dataclass, field
from threading import Lock
from typing import List

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

    @property
    def updates(self) -> int:
        return max(0, self.end_update - self.start_update)


@dataclass
class StagePlan:
    total_updates: int
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
    if cfg.epochs and cfg.epochs > 0:
        steps_per_epoch = int(cfg.epoch_samples // max(1, cfg.batch_size * cfg.grad_accum))
        return max(1, int(round(cfg.epochs * steps_per_epoch)))
    return max(1, int(cfg.steps))


def build_stage_plan(cfg: TrainConfig) -> StagePlan:
    total_updates = infer_total_updates(cfg)
    stage_a_updates = max(1, int(round(total_updates * float(cfg.stage_a_ratio))))
    stage_b_updates = max(0, total_updates - stage_a_updates)
    stages = [
        StageDefinition(
            name="stage_a_recover",
            start_update=0,
            end_update=stage_a_updates,
            pretrain_ratio=float(cfg.stage_a_pretrain_ratio),
            task_ratio=float(cfg.stage_a_task_ratio),
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
            )
        )
    return StagePlan(total_updates=total_updates, stages=stages)


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
