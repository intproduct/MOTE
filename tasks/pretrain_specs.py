from __future__ import annotations

from typing import List

from ..config.schema import DataConfig
from ..data.access import check_code_dataset_accessible
from ..data.specs import HFTextTask, LocalTokenShardTask, TaskSpec
from ..runtime import normalize_hf_config


def build_pretrain_tasks(cfg: DataConfig, logger=None) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    use_code = bool(cfg.use_code)
    if use_code and not check_code_dataset_accessible(cfg, logger=logger):
        if logger is not None:
            logger.warning("bigcode/the-stack-v2 is not accessible; disabling code pretrain task (use_code=False)")
        use_code = False
    if cfg.use_wiki_local:
        tasks.append(
            LocalTokenShardTask(
                name="wiki24_tok",
                path=cfg.tok_shard_dir,
                split="train",
                weight=float(cfg.wt_wiki),
                kind="local_token_shards",
                group="pretrain",
                source_family="wiki_shards",
            )
        )
    if cfg.use_fineweb:
        tasks.append(
            HFTextTask(
                name="fineweb",
                path=cfg.fineweb_cache_path,
                split="train",
                weight=float(cfg.wt_fineweb),
                kind="auto",
                hf_name=cfg.fineweb_hf_name,
                hf_config=normalize_hf_config(cfg.fineweb_hf_config),
                text_field=cfg.fineweb_text_field,
                group="pretrain",
                source_family="web_text",
            )
        )
    if use_code:
        tasks.append(
            HFTextTask(
                name="stack_code",
                path=cfg.code_cache_path,
                split="train",
                weight=float(cfg.wt_code),
                kind="auto",
                hf_name=cfg.code_hf_name,
                hf_config=normalize_hf_config(cfg.code_hf_config),
                text_field=cfg.code_text_field,
                group="pretrain",
                source_family="code",
            )
        )
    if not tasks:
        raise ValueError("pretrain task pool is empty")
    return tasks
