from __future__ import annotations

from typing import Any, List

from ..config.schema import DataConfig
from ..data.specs import HFChatTask, HFTextTask, TaskSpec
from ..runtime import normalize_hf_config


def _source_to_kind(source: str, dataset_format: str) -> str:
    if source == "hf":
        return "hf_text" if dataset_format == "text" else "hf_chat"
    if source == "local_jsonl":
        return "jsonl"
    return source


def _max_samples(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def build_extra_dataset_tasks(cfg: DataConfig, group: str | None = None, logger=None) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    for ds in list(getattr(cfg, "extra_datasets", []) or []):
        if not bool(ds.get("enabled", True)):
            continue
        task_group = str(ds.get("group", "task") or "task")
        if group is not None and task_group != group:
            continue
        dataset_format = str(ds.get("format", "chat_messages") or "chat_messages")
        source = str(ds.get("source", "hf") or "hf")
        kind = _source_to_kind(source, dataset_format)
        hf_name = str(ds.get("hf_name", "") or "") or None
        path = str(ds.get("path") or ds.get("cache_path") or hf_name or "")
        common = {
            "name": str(ds["name"]),
            "path": path,
            "split": str(ds.get("split", "train") or "train"),
            "weight": float(ds.get("weight", 1.0)),
            "max_samples": _max_samples(ds.get("max_samples")),
            "kind": kind,
            "hf_name": hf_name,
            "hf_config": normalize_hf_config(ds.get("hf_config")),
            "group": task_group,
            "bucket": str(ds.get("bucket", "extra_task") or "extra_task"),
            "source_family": str(ds.get("source_family", "generic") or "generic"),
            "metadata": {"extra_dataset": True, "dataset_format": dataset_format, "source": source},
        }
        if dataset_format == "text":
            tasks.append(HFTextTask(text_field=str(ds.get("text_field", "text") or "text"), **common))
        else:
            tasks.append(
                HFChatTask(
                    dataset_format=dataset_format,
                    text_field=ds.get("text_field"),
                    messages_field=ds.get("messages_field"),
                    prompt_field=ds.get("prompt_field"),
                    response_field=ds.get("response_field"),
                    system_field=ds.get("system_field"),
                    role_key=str(ds.get("role_key", "role") or "role"),
                    content_key=str(ds.get("content_key", "content") or "content"),
                    role_map=dict(ds.get("role_map") or {}),
                    instruction_field=ds.get("instruction_field"),
                    input_field=ds.get("input_field"),
                    output_field=ds.get("output_field"),
                    question_field=ds.get("question_field"),
                    solution_field=ds.get("solution_field"),
                    answer_field=ds.get("answer_field"),
                    answer_extraction=ds.get("answer_extraction"),
                    skip_if_no_assistant=bool(ds.get("skip_if_no_assistant", True)),
                    skip_empty=bool(ds.get("skip_empty", True)),
                    **common,
                )
            )
        if logger is not None:
            logger.info("[Data] enable extra_dataset=%s kind=%s format=%s group=%s", ds["name"], kind, dataset_format, task_group)
    return tasks
