from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from .config_utils import resolve_task_eval_settings


PRIMARY_FILTER = "strict-match"
PRIMARY_METRIC = "exact_match"


def resolve_lm_eval_num_fewshot(fit_cfg, cli_value: Optional[int]) -> int:
    """Match the formal final-eval precedence while allowing a CLI override."""
    if cli_value is not None:
        return int(cli_value)
    settings = resolve_task_eval_settings(fit_cfg, "gsm8k", "final", backend="lm_eval")
    return int(settings["fewshot"])


def _supported_kwargs(callable_obj, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _task_config(task, key: str, default: Any = None) -> Any:
    getter = getattr(task, "get_config", None)
    if callable(getter):
        value = getter(key)
        return default if value is None else value
    config = getattr(task, "config", None)
    return getattr(config, key, default)


def _set_task_config(task, key: str, value: Any, *, update: bool = False) -> None:
    setter = getattr(task, "set_config", None)
    if not callable(setter):
        raise RuntimeError("installed lm_eval task does not expose set_config")
    kwargs = {"key": key, "value": value, "update": update}
    setter(**_supported_kwargs(setter, kwargs))


def _instance_arguments(instance) -> tuple[Any, ...]:
    value = getattr(instance, "arguments", None)
    if value is None:
        value = getattr(instance, "args", None)
    if value is None:
        raise RuntimeError("lm_eval request instance exposes neither arguments nor args")
    return tuple(value)


def _first_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_text(item)
            if found is not None:
                return found
    return None


def _load_task(task_name: str):
    try:
        tasks_module = importlib.import_module("lm_eval.tasks")
        manager = tasks_module.TaskManager()
    except Exception as exc:
        raise ImportError(
            'The lm_eval protocol requires the same lm-evaluation-harness installation used by '
            'formal eval. Install it with: pip install -U "lm_eval[hf]"'
        ) from exc

    if hasattr(manager, "load"):
        loaded = manager.load([task_name])
    else:
        get_task_dict = getattr(tasks_module, "get_task_dict", None)
        if not callable(get_task_dict):
            raise RuntimeError("installed lm_eval exposes neither TaskManager.load nor get_task_dict")
        loaded = get_task_dict([task_name], task_manager=manager)
    task_map = loaded.get("tasks", loaded) if isinstance(loaded, Mapping) else {}
    task = task_map.get(task_name) if isinstance(task_map, Mapping) else None
    if task is None:
        raise RuntimeError(f"lm_eval TaskManager did not load task {task_name!r}")
    return task


def _lm_eval_version() -> str:
    try:
        return importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        try:
            return importlib.metadata.version("lm-eval")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"


@dataclass
class LMEvalGSM8KRecord:
    dataset_index: int
    doc: Mapping[str, Any]
    prompt: str
    generation_kwargs: Dict[str, Any]
    instances: list[Any]

    def as_boundary_row(self) -> Dict[str, Any]:
        question = self.doc.get("question", "")
        answer = self.doc.get("answer", "")
        return {
            "idx": self.dataset_index,
            "dataset_index": self.dataset_index,
            "question": str(question),
            "answer": str(answer),
            "source": "gsm8k_test",
            "_lm_eval_prompt": self.prompt,
            "_lm_eval_record": self,
        }


class LMEvalGSM8KProtocol:
    """Adapter over lm_eval's own GSM8K requests, filters, and metrics."""

    task_name = "gsm8k"
    split = "test"
    filter_name = PRIMARY_FILTER
    metric_name = PRIMARY_METRIC

    def __init__(
        self,
        *,
        task,
        num_fewshot: int,
        runtime: Mapping[str, Any],
        generation_overrides: Optional[Mapping[str, Any]],
        seed: int,
        max_prompts: Optional[int],
        chat_template=None,
        tokenizer_name: str = "",
        lm_eval_version: str = "unknown",
    ):
        self.task = task
        self.num_fewshot = int(num_fewshot)
        self.runtime = dict(runtime)
        self.generation_overrides = dict(generation_overrides or {})
        self.seed = int(seed)
        self.lm_eval_version = str(lm_eval_version)
        self.records: list[LMEvalGSM8KRecord] = []

        if _task_config(task, "output_type") != "generate_until":
            raise RuntimeError("lm_eval GSM8K task must use output_type='generate_until'")
        if self.generation_overrides:
            _set_task_config(task, "generation_kwargs", self.generation_overrides, update=True)
        _set_task_config(task, "num_fewshot", self.num_fewshot)
        set_seed = getattr(task, "set_fewshot_seed", None)
        if callable(set_seed):
            set_seed(seed=self.seed)

        apply_chat = bool(self.runtime.get("apply_chat_template", False))
        if apply_chat and chat_template is None:
            raise ValueError("lm_eval apply_chat_template=true requires an exact chat template callback")
        build_kwargs = {
            "limit": max_prompts,
            "samples": None,
            "rank": 0,
            "world_size": 1,
            "cache_requests": False,
            "rewrite_requests_cache": False,
            "system_instruction": self.runtime.get("system_instruction"),
            "apply_chat_template": apply_chat,
            "fewshot_as_multiturn": bool(self.runtime.get("fewshot_as_multiturn", True)),
            "chat_template": chat_template if apply_chat else None,
            "tokenizer_name": tokenizer_name if apply_chat else "",
        }
        task.build_all_requests(**_supported_kwargs(task.build_all_requests, build_kwargs))
        self._collect_records(max_prompts=max_prompts)

    @classmethod
    def from_fit_config(
        cls,
        fit_cfg,
        *,
        num_fewshot: Optional[int],
        seed: int,
        max_prompts: Optional[int],
        chat_template=None,
        tokenizer_name: str = "",
    ) -> "LMEvalGSM8KProtocol":
        settings = resolve_task_eval_settings(fit_cfg, "gsm8k", "final", backend="lm_eval")
        resolved_fewshot = resolve_lm_eval_num_fewshot(fit_cfg, num_fewshot)
        return cls(
            task=_load_task("gsm8k"),
            num_fewshot=resolved_fewshot,
            runtime=settings.get("runtime") or {},
            generation_overrides=settings.get("gen_kwargs"),
            seed=seed,
            max_prompts=max_prompts,
            chat_template=chat_template,
            tokenizer_name=tokenizer_name,
            lm_eval_version=_lm_eval_version(),
        )

    def _collect_records(self, *, max_prompts: Optional[int]) -> None:
        instances_by_doc: Dict[int, list[Any]] = {}
        for instance in list(getattr(self.task, "instances", []) or []):
            instances_by_doc.setdefault(int(instance.doc_id), []).append(instance)
        for values in instances_by_doc.values():
            values.sort(key=lambda value: int(getattr(value, "idx", 0)))

        iterator_kwargs = {
            "rank": 0,
            "limit": max_prompts,
            "world_size": 1,
            "samples": None,
        }
        iterator: Iterable[tuple[int, Mapping[str, Any]]] = self.task.doc_iterator(
            **_supported_kwargs(self.task.doc_iterator, iterator_kwargs)
        )
        for doc_id, doc in iterator:
            instances = instances_by_doc.get(int(doc_id), [])
            if not instances:
                raise RuntimeError(f"lm_eval produced no request for GSM8K doc_id={doc_id}")
            arguments = _instance_arguments(instances[0])
            if not arguments or not isinstance(arguments[0], str):
                raise RuntimeError(f"lm_eval GSM8K doc_id={doc_id} request has no text prompt")
            generation_kwargs = dict(arguments[1] or {}) if len(arguments) > 1 else {}
            self.records.append(
                LMEvalGSM8KRecord(
                    dataset_index=int(doc_id),
                    doc=doc,
                    prompt=arguments[0],
                    generation_kwargs=generation_kwargs,
                    instances=instances,
                )
            )

    def boundary_rows(self) -> list[Dict[str, Any]]:
        return [record.as_boundary_row() for record in self.records]

    def score(self, record: LMEvalGSM8KRecord, completion: str) -> Dict[str, Any]:
        for instance in record.instances:
            instance.resps = [completion]
            instance.filtered_resps = {}
        # lm_eval normally filters every task instance after all generations
        # finish. Pass@K scores one completion at a time, so temporarily expose
        # only this document's requests while invoking the exact same pipelines.
        all_instances = self.task.instances
        try:
            self.task.instances = record.instances
            self.task.apply_filters()
        finally:
            self.task.instances = all_instances
        filtered = []
        for instance in record.instances:
            if self.filter_name not in instance.filtered_resps:
                available = sorted(instance.filtered_resps)
                raise RuntimeError(
                    f"lm_eval GSM8K filter {self.filter_name!r} is unavailable; available={available}"
                )
            filtered.append(instance.filtered_resps[self.filter_name])
        metrics = self.task.process_results(record.doc, filtered)
        if self.metric_name not in metrics:
            raise RuntimeError(
                f"lm_eval GSM8K process_results omitted {self.metric_name!r}: {sorted(metrics)}"
            )
        metric_value = metrics[self.metric_name]
        if not isinstance(metric_value, (bool, int, float)):
            raise RuntimeError(
                f"lm_eval GSM8K exact_match returned unsupported value {metric_value!r}"
            )
        return {
            "correct": bool(float(metric_value) == 1.0),
            "metric_value": float(metric_value),
            "filter": self.filter_name,
            "metric": self.metric_name,
            "extracted_answer": _first_text(filtered),
            "target": self.task.doc_to_target(record.doc),
            "filtered_responses": filtered,
        }

    def metadata(self) -> Dict[str, Any]:
        task_config = getattr(self.task, "config", None)
        return {
            "task": self.task_name,
            "split": self.split,
            "num_fewshot": self.num_fewshot,
            "prompt_source": "lm_eval task request instance.arguments[0]",
            "scoring_source": "lm_eval task apply_filters/process_results",
            "scorer": f"{self.metric_name},{self.filter_name}",
            "apply_chat_template": bool(self.runtime.get("apply_chat_template", False)),
            "enable_thinking": bool(self.runtime.get("enable_thinking", False)),
            "fewshot_as_multiturn": bool(self.runtime.get("fewshot_as_multiturn", True)),
            "system_instruction": self.runtime.get("system_instruction"),
            "lm_eval_version": self.lm_eval_version,
            "lm_eval_task_source": (
                getattr(self.task, "CONFIG_FILE", None)
                or f"{self.task.__class__.__module__}.{self.task.__class__.__qualname__}"
            ),
            "lm_eval_task_config": {
                "dataset_path": getattr(task_config, "dataset_path", None),
                "dataset_name": getattr(task_config, "dataset_name", None),
                "test_split": getattr(task_config, "test_split", None),
                "fewshot_split": getattr(task_config, "fewshot_split", None),
                "output_type": _task_config(self.task, "output_type"),
                "generation_kwargs": _task_config(self.task, "generation_kwargs", {}),
                "repeats": _task_config(self.task, "repeats", 1),
                "metadata": getattr(task_config, "metadata", None),
            },
        }
