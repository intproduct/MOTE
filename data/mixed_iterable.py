from __future__ import annotations

import random
import time
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List

from torch.utils.data import IterableDataset, get_worker_info

from .caching import download_and_cache_dataset, load_dataset_auto_cached, resolve_split
from .shard_loader import iter_jsonl, iter_jsonl_gz, iter_local_token_shards
from .specs import HFChatTask, HFTextTask, TaskSpec
from .tokenization import (
    build_example_from_token_ids,
    make_causal_lm_example_from_text,
    make_chat_supervised_example,
    make_supervised_example,
)

try:
    from datasets import load_dataset, load_from_disk
except Exception:
    load_dataset = None
    load_from_disk = None


def load_dataset_any(kind: str, path: str, split: str, hf_name: str | None = None, hf_config: str | None = None):
    if kind == "auto":
        res = load_dataset_auto_cached(path, split, hf_name=hf_name, hf_config=hf_config)
        if isinstance(res, tuple) and len(res) == 6:
            _, ready, cache_path, ds_name, ds_cfg, ds_split = res
            return download_and_cache_dataset(cache_path, ready, ds_name, ds_cfg, ds_split)
        return res
    if kind == "load_from_disk":
        if load_from_disk is None:
            raise RuntimeError("datasets not installed")
        return resolve_split(load_from_disk(path), split)
    if kind == "jsonl":
        return iter_jsonl(Path(path))
    if kind == "jsonl_gz":
        return iter_jsonl_gz(Path(path))
    if kind in ["hf", "hf_text", "hf_chat"]:
        if load_dataset is None:
            raise RuntimeError("datasets not installed")
        return load_dataset(hf_name or path, hf_config, split=split)
    if kind == "local_token_shards":
        return iter_local_token_shards(path)
    raise ValueError(f"Unknown kind={kind}")


def apply_task_sample_limit(raw, task: TaskSpec):
    max_samples = getattr(task, "max_samples", None)
    if max_samples is None:
        return raw
    try:
        max_samples = int(max_samples)
    except Exception:
        return raw
    if max_samples <= 0:
        return raw
    if hasattr(raw, "select"):
        try:
            size = len(raw)
            return raw.select(range(min(size, max_samples)))
        except Exception:
            pass
    return islice(iter(raw), max_samples)


class StageAwareMixedTaskIterableDataset(IterableDataset):
    REASONING_SOURCE_FAMILIES = {"reasoning", "math_reasoning", "mcq_reasoning"}

    def __init__(
        self,
        tokenizer,
        pretrain_tasks: List[TaskSpec],
        task_tasks: List[TaskSpec],
        stage_state,
        max_len: int,
        samples_per_epoch: int,
        seed: int = 0,
        logger=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrain_tasks = list(pretrain_tasks)
        self.task_tasks = list(task_tasks)
        self.stage_state = stage_state
        self.max_len = int(max_len)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.logger = logger
        self._pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
        self._task_stats: Dict[str, Dict[str, Any]] = {}
        self._logged_task_sources: set[str] = set()

    @property
    def pad_id(self) -> int:
        return int(self._pad_id)

    def _load_raw_for_task(self, task: TaskSpec, wid: int):
        if getattr(task, "kind", "load_from_disk") == "auto":
            res = load_dataset_auto_cached(task.path, task.split, hf_name=getattr(task, "hf_name", None), hf_config=getattr(task, "hf_config", None))
            if isinstance(res, tuple) and len(res) == 6:
                _, ready, path, hf_name, hf_config, split = res
                self._log_task_load(task, source="remote", raw_total=None)
                if wid == 0:
                    return apply_task_sample_limit(download_and_cache_dataset(path, ready, hf_name, hf_config, split), task)
                t0 = time.time()
                while not ready.exists():
                    if time.time() - t0 > 3600:
                        raise TimeoutError(f"Timeout waiting for dataset cache: {path}")
                    time.sleep(1.0)
                ds = load_from_disk(str(path))
                loaded = resolve_split(ds, split=task.split) if hasattr(ds, "__len__") else ds
                self._log_task_load(task, source="cached_after_download", raw_total=self._safe_len(loaded))
                return apply_task_sample_limit(loaded, task)
            self._log_task_load(task, source="cached", raw_total=self._safe_len(res))
            return apply_task_sample_limit(res, task)
        loaded = load_dataset_any(task.kind, task.path, task.split, hf_name=task.hf_name, hf_config=task.hf_config)
        self._log_task_load(task, source=str(task.kind), raw_total=self._safe_len(loaded))
        return apply_task_sample_limit(loaded, task)

    def _make_iter(self, raw_or_factory):
        return iter(raw_or_factory() if callable(raw_or_factory) else raw_or_factory)

    def _task_probs(self) -> tuple[List[TaskSpec], List[float]]:
        stage = self.stage_state.current_stage()
        tasks: List[TaskSpec] = []
        weights: List[float] = []
        for task in self.pretrain_tasks:
            tasks.append(task)
            weights.append(max(0.0, float(task.weight) * float(stage.pretrain_ratio)))
        for task in self.task_tasks:
            task_weight = max(0.0, float(task.weight) * float(stage.task_ratio))
            if bool(getattr(stage, "reasoning_focused", False)) and str(task.source_family) in self.REASONING_SOURCE_FAMILIES:
                task_weight *= max(0.0, float(getattr(stage, "reasoning_boost", 1.0)))
            tasks.append(task)
            weights.append(task_weight)
        if sum(weights) <= 0:
            raise ValueError("All task weights are zero for current stage")
        return tasks, weights

    def _safe_len(self, raw) -> int | None:
        try:
            return len(raw)
        except Exception:
            return None

    def _get_task_stats(self, task: TaskSpec) -> Dict[str, Any]:
        return self._task_stats.setdefault(
            task.name,
            {
                "loaded_samples": 0,
                "skipped_missing_fields": 0,
                "skipped_overlong": 0,
                "selected_trace_strategy": {},
                "skip_logs": 0,
            },
        )

    def _log_task_load(self, task: TaskSpec, *, source: str, raw_total: int | None) -> None:
        if self.logger is None or task.name in self._logged_task_sources:
            return
        self._logged_task_sources.add(task.name)
        msg = [f"[Data] task={task.name}", f"source={source}"]
        if raw_total is not None:
            msg.append(f"raw_samples={raw_total}")
        if getattr(task, "supports_skip", False):
            policy = task.describe_data_policy()
            if policy.get("length_policy"):
                msg.append(f"length_policy={policy['length_policy']}")
            if policy.get("trace_selection_policy"):
                msg.append(f"trace_policy={policy['trace_selection_policy']}")
        self.logger.info(" ".join(msg))

    def _record_skip(self, task: TaskSpec, reason: str, trace_strategy: str | None) -> None:
        stats = self._get_task_stats(task)
        if reason == "overlong":
            stats["skipped_overlong"] += 1
        else:
            stats["skipped_missing_fields"] += 1
        if trace_strategy:
            trace_stats = stats["selected_trace_strategy"]
            trace_stats[trace_strategy] = int(trace_stats.get(trace_strategy, 0)) + 1
        if self.logger is not None and (stats["skip_logs"] == 0 or (stats["skipped_missing_fields"] + stats["skipped_overlong"]) % 100 == 0):
            stats["skip_logs"] += 1
            self.logger.info(
                "[Data] task=%s skipped_missing_fields=%s skipped_overlong=%s trace_selection=%s",
                task.name,
                stats["skipped_missing_fields"],
                stats["skipped_overlong"],
                stats["selected_trace_strategy"],
            )

    def _record_success(self, task: TaskSpec, trace_strategy: str | None) -> None:
        stats = self._get_task_stats(task)
        stats["loaded_samples"] += 1
        if trace_strategy:
            trace_stats = stats["selected_trace_strategy"]
            trace_stats[trace_strategy] = int(trace_stats.get(trace_strategy, 0)) + 1

    def _to_supervised(self, task: TaskSpec, ex: Dict[str, Any]):
        if task.kind == "local_token_shards":
            if "input_ids" in ex:
                sup = build_example_from_token_ids(ex["input_ids"], self.max_len, eos_id=self.tokenizer.eos_token_id)
            elif "text" in ex:
                sup = make_causal_lm_example_from_text(self.tokenizer, ex["text"], self.max_len, add_eos=True)
            else:
                raise KeyError(f"[{task.name}] local shard example missing input_ids/text")
            sup["task"] = task.name
            sup["group"] = task.group
            sup["source_family"] = task.source_family
            sup["eval_type"] = "causal_lm"
            return sup
        if task.kind in ["hf_text"]:
            txt = ex.get(task.text_field) if isinstance(task, HFTextTask) else None
            if txt is None:
                for cand in ["text", "content", "body", "document", "code", "completion"]:
                    if cand in ex and ex[cand] is not None:
                        txt = ex[cand]
                        break
            sup = make_causal_lm_example_from_text(self.tokenizer, str(txt), self.max_len, add_eos=True)
            sup["task"] = task.name
            sup["group"] = task.group
            sup["eval_type"] = "causal_lm"
            return sup
        if task.kind == "hf_chat":
            if not isinstance(task, HFChatTask):
                raise TypeError(f"[{task.name}] hf_chat task must be HFChatTask")
            messages = ex.get(task.messages_field) if task.messages_field else None
            if not messages:
                messages = []
                sys_text = ex.get(task.system_field) if task.system_field else None
                prompt = ex.get(task.prompt_field) if task.prompt_field else None
                response = ex.get(task.response_field) if task.response_field else None
                if sys_text:
                    messages.append({"role": "system", "content": str(sys_text)})
                if prompt:
                    messages.append({"role": "user", "content": str(prompt)})
                if response:
                    messages.append({"role": "assistant", "content": str(response)})
            sup = make_chat_supervised_example(self.tokenizer, messages=messages, max_len=self.max_len, add_eos=True)
            sup["task"] = task.name
            sup["group"] = task.group
            sup["source_family"] = task.source_family
            sup["eval_type"] = "chat_sft"
            return sup
        mapped = task.map_example(ex)
        if mapped is None:
            self._record_skip(task, str(task.metadata.get("last_reason", "missing_fields")), str(task.metadata.get("last_trace_strategy") or "unknown"))
            return None
        prompt, answer, eval_type = mapped
        sup = make_supervised_example(self.tokenizer, prompt, answer, self.max_len, add_eos=True)
        sup["task"] = task.name
        sup["group"] = task.group
        sup["source_family"] = task.source_family
        sup["eval_type"] = eval_type
        self._record_success(task, str(task.metadata.get("last_trace_strategy") or "direct_fields"))
        return sup

    def __iter__(self):
        worker = get_worker_info()
        wid = worker.id if worker else 0
        wnum = worker.num_workers if worker else 1
        rng = random.Random(self.seed + 997 * wid)

        task_objs, _ = self._task_probs()
        factories = {task.name: (lambda t=task: self._load_raw_for_task(t, wid)) for task in task_objs}
        iters = {name: self._make_iter(factory) for name, factory in factories.items()}

        total = self.samples_per_epoch
        my_total = total // wnum + (1 if wid < (total % wnum) else 0)

        for _ in range(my_total):
            tasks, weights = self._task_probs()
            tidx = rng.choices(range(len(tasks)), weights=weights, k=1)[0]
            task = tasks[tidx]
            attempts = 0
            while True:
                try:
                    ex = next(iters[task.name])
                    sup = self._to_supervised(task, ex)
                    if sup is not None:
                        yield sup
                        break
                except StopIteration:
                    iters[task.name] = self._make_iter(factories[task.name])
                attempts += 1
                if attempts >= 1000:
                    stats = self._get_task_stats(task)
                    raise RuntimeError(
                        f"Task {task.name} exceeded skip budget while building supervised samples; "
                        f"stats={stats}"
                    )
