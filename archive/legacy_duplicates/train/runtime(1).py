from __future__ import annotations

from typing import Any, Dict

from transformers import get_linear_schedule_with_warmup

from ..runtime import dtype_to_name


def build_scheduler_builder(fit_cfg, total_updates: int, logger=None):
    if not bool(fit_cfg.train.lr_warmup):
        return None, 0
    resolved_warmup_updates = min(int(fit_cfg.train.lr_warmup_steps), int(total_updates))

    def _builder(optimizer, num_training_steps: int):
        warmup_updates = min(int(fit_cfg.train.lr_warmup_steps), int(num_training_steps))
        if logger is not None:
            logger.info(
                "[LR] warmup_updates=%s total_updates=%s block_lr=%s router_lr=%s",
                warmup_updates,
                int(num_training_steps),
                fit_cfg.train.block_lr,
                fit_cfg.train.router_lr,
            )
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_updates,
            num_training_steps=int(num_training_steps),
        )

    return _builder, resolved_warmup_updates


def build_runtime_metadata(*, fit_cfg, model_dtype, warmup_updates: int) -> Dict[str, Any]:
    return {
        "resolved_model_dtype": dtype_to_name(model_dtype),
        "lr_warmup_enabled": bool(fit_cfg.train.lr_warmup),
        "lr_warmup_steps": int(fit_cfg.train.lr_warmup_steps),
        "resolved_lr_warmup_steps": int(warmup_updates),
    }
