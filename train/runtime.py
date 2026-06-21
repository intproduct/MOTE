from __future__ import annotations

from typing import Any, Dict

from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from ..runtime import dtype_to_name


def build_scheduler_builder(fit_cfg, total_updates: int, logger=None):
    scheduler_type = str(fit_cfg.train.lr_scheduler_type)
    actual_training_steps = int(total_updates)
    configured_warmup_steps = int(fit_cfg.train.lr_warmup_steps)
    warmup_enabled = bool(fit_cfg.train.lr_warmup)
    resolved_warmup_updates = min(configured_warmup_steps, actual_training_steps) if warmup_enabled else 0
    configured_decay_steps = getattr(fit_cfg.train, "lr_decay_steps", None)
    scheduler_total_steps = actual_training_steps if configured_decay_steps is None else int(configured_decay_steps)
    scheduler_warmup_updates = 0 if scheduler_type == "constant" else resolved_warmup_updates
    if scheduler_type in {"linear", "cosine"} and scheduler_total_steps <= scheduler_warmup_updates:
        scheduler_total_steps = scheduler_warmup_updates + 1

    scheduler_metadata = {
        "lr_scheduler_type": scheduler_type,
        "lr_warmup": warmup_enabled,
        "lr_warmup_steps": configured_warmup_steps,
        "resolved_lr_warmup_steps": int(resolved_warmup_updates),
        "lr_decay_steps": None if configured_decay_steps is None else int(configured_decay_steps),
        "resolved_lr_decay_steps": int(scheduler_total_steps),
        "scheduler_total_steps": int(scheduler_total_steps),
        "actual_training_steps": int(actual_training_steps),
    }

    def _builder(optimizer, num_training_steps: int):
        actual_steps = int(num_training_steps)
        warmup_updates = min(configured_warmup_steps, actual_steps) if warmup_enabled else 0
        schedule_warmup_updates = 0 if scheduler_type == "constant" else warmup_updates
        total_decay_steps = actual_steps if configured_decay_steps is None else int(configured_decay_steps)
        if scheduler_type in {"linear", "cosine"} and total_decay_steps <= schedule_warmup_updates:
            total_decay_steps = schedule_warmup_updates + 1
        resolution = {
            "lr_scheduler_type": scheduler_type,
            "lr_warmup": warmup_enabled,
            "lr_warmup_steps": configured_warmup_steps,
            "resolved_lr_warmup_steps": int(warmup_updates),
            "lr_decay_steps": None if configured_decay_steps is None else int(configured_decay_steps),
            "resolved_lr_decay_steps": int(total_decay_steps),
            "scheduler_total_steps": int(total_decay_steps),
            "actual_training_steps": int(actual_steps),
        }
        _builder.last_resolution = resolution
        if logger is not None:
            logger.info(
                "[LR] scheduler_type=%s lr_warmup=%s configured_warmup_steps=%s resolved_warmup_steps=%s "
                "configured_decay_steps=%s resolved_decay_steps=%s actual_training_steps=%s block_lr=%s router_lr=%s",
                scheduler_type,
                warmup_enabled,
                configured_warmup_steps,
                warmup_updates,
                configured_decay_steps,
                total_decay_steps,
                actual_steps,
                fit_cfg.train.block_lr,
                fit_cfg.train.router_lr,
            )
        if scheduler_type == "constant":
            return get_constant_schedule(optimizer)
        if scheduler_type == "constant_with_warmup":
            return get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=schedule_warmup_updates,
            )
        if scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=schedule_warmup_updates,
                num_training_steps=int(total_decay_steps),
            )
        if scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=schedule_warmup_updates,
                num_training_steps=int(total_decay_steps),
            )
        raise ValueError(
            "Unsupported train.lr_scheduler_type "
            f"{scheduler_type!r}; expected one of ['constant', 'constant_with_warmup', 'cosine', 'linear']"
        )

    _builder.last_resolution = dict(scheduler_metadata)
    return _builder, scheduler_metadata


def build_runtime_metadata(*, fit_cfg, model_dtype, warmup_updates: int) -> Dict[str, Any]:
    return {
        "resolved_model_dtype": dtype_to_name(model_dtype),
        "lr_warmup_enabled": bool(fit_cfg.train.lr_warmup),
        "lr_scheduler_type": str(fit_cfg.train.lr_scheduler_type),
        "lr_warmup": bool(fit_cfg.train.lr_warmup),
        "lr_warmup_steps": int(fit_cfg.train.lr_warmup_steps),
        "resolved_lr_warmup_steps": int(warmup_updates),
        "lr_decay_steps": None if getattr(fit_cfg.train, "lr_decay_steps", None) is None else int(fit_cfg.train.lr_decay_steps),
    }
