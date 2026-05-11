from __future__ import annotations

import argparse
import json
from typing import Any

from transformers import AutoTokenizer

from ..config import load_config
from ..data.mixed_iterable import apply_task_sample_limit, load_dataset_any
from ..data.tokenization import make_supervised_example
from ..tasks.reasoning_specs import NormalizedReasoningTask, build_task_mixture_tasks
from ..train.stages import build_stage_plan


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect one normalized reasoning sample and its label spans.")
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--task", type=str, default=None, help="Task name such as gsm8k_train or openr1_math_train.")
    parser.add_argument("--sample_index", type=int, default=0, help="0-based raw sample index inside the selected task.")
    return parser.parse_args()


def _pick_task(tasks, task_name: str | None):
    reasoning_tasks = [task for task in tasks if isinstance(task, NormalizedReasoningTask)]
    if not reasoning_tasks:
        raise ValueError("No reasoning tasks are enabled in this config.")
    if task_name is None:
        return reasoning_tasks[0]
    for task in reasoning_tasks:
        if task.name == task_name:
            return task
    raise ValueError(f"Could not find reasoning task {task_name!r}. Available: {[task.name for task in reasoning_tasks]}")


def _pick_sample(task: NormalizedReasoningTask, sample_index: int) -> Any:
    raw = load_dataset_any(task.kind, task.path, task.split, hf_name=task.hf_name, hf_config=task.hf_config, metadata=task.metadata)
    raw = apply_task_sample_limit(raw, task)
    for idx, ex in enumerate(raw):
        if idx == sample_index:
            return ex
    raise IndexError(f"Task {task.name} does not have sample_index={sample_index}")


def _label_span_summary(labels) -> dict[str, Any]:
    label_list = labels.tolist()
    first_trainable = next((idx for idx, value in enumerate(label_list) if value != -100), None)
    last_trainable = next((len(label_list) - 1 - idx for idx, value in enumerate(reversed(label_list)) if value != -100), None)
    return {
        "total_tokens": len(label_list),
        "masked_prompt_tokens": sum(1 for value in label_list if value == -100),
        "trainable_tokens": sum(1 for value in label_list if value != -100),
        "first_trainable_index": first_trainable,
        "last_trainable_index": last_trainable,
        "prompt_all_masked": all(value == -100 for value in label_list[: first_trainable or 0]),
        "has_trainable_region": first_trainable is not None,
    }


def main():
    args = parse_args()
    cfg = load_config(config_json=args.config_json)
    tasks = build_task_mixture_tasks(cfg.data)
    task = _pick_task(tasks, args.task)
    stage_plan = build_stage_plan(cfg.train)
    ex = _pick_sample(task, int(args.sample_index))
    normalized = task.normalize_example(ex)
    print(f"task={task.name}")
    print(f"bucket={task.bucket}")
    print(f"group={task.group}")
    print(f"source_family={task.source_family}")
    print(f"dataset_name={task.dataset_name}")
    print(f"reasoning_supervision_mode={task.supervision_mode}")
    print(f"task_bucket_mode={cfg.train.task_bucket_mode}")
    for stage in stage_plan.stages:
        print(f"{stage.name}_bucket_ratios=" + json.dumps(dict(getattr(stage, "bucket_ratios", {}) or {}), ensure_ascii=False))
    print(f"trace_strategy={normalized.get('trace_strategy')}")
    print(f"ok={normalized.get('ok')}")
    if not normalized.get("ok"):
        print(json.dumps(normalized, ensure_ascii=False, indent=2))
        return

    prompt = normalized["prompt"]
    target = normalized["target"]
    record = normalized.get("reasoning_record") or {}
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_path,
        use_fast=True,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = make_supervised_example(tokenizer, prompt, target, int(cfg.data.seq_len_run), add_eos=True)
    label_summary = _label_span_summary(tokenized["labels"])
    print(f"final_answer={record.get('final_answer')}")
    print(f"approx_tokens={normalized.get('approx_tokens')}")
    print("label_span=" + json.dumps(label_summary, ensure_ascii=False))
    print("prompt_text:")
    print(prompt)
    print("\ntarget_text:")
    print(target)
    print("\nreasoning_record:")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
