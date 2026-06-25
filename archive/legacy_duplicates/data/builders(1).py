from __future__ import annotations

from .mixed_iterable import StageAwareMixedTaskIterableDataset


def build_stage_aware_train_dataset(tokenizer, pretrain_tasks, task_tasks, stage_state, data_cfg, train_cfg, logger=None):
    samples_per_epoch = int(train_cfg.batch_size * train_cfg.grad_accum * stage_state.plan.total_updates * 2)
    return StageAwareMixedTaskIterableDataset(
        tokenizer=tokenizer,
        pretrain_tasks=pretrain_tasks,
        task_tasks=task_tasks,
        stage_state=stage_state,
        max_len=int(data_cfg.seq_len_run),
        samples_per_epoch=max(1000, samples_per_epoch),
        seed=int(train_cfg.seed),
        logger=logger,
    )
