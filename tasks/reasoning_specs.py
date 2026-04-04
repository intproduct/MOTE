from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from ..config.schema import DataConfig
from ..data.access import inspect_task_dataset
from ..data.specs import TaskSpec
from ..runtime import normalize_hf_config
from .reasoning_normalization import normalize_reasoning_sample


@dataclass
class NormalizedReasoningTask(TaskSpec):
    dataset_name: str = "generic_reasoning"
    length_policy: Dict[str, Any] = field(default_factory=dict)
    trace_selection_policy: str = "prefer_short_if_available"
    supports_skip: bool = True

    def describe_data_policy(self) -> Mapping[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "length_policy": dict(self.length_policy),
            "trace_selection_policy": self.trace_selection_policy,
        }

    def map_example(self, ex: Dict[str, Any]):
        normalized = normalize_reasoning_sample(ex, self.dataset_name, self.length_policy)
        self.metadata["last_reason"] = normalized.get("reason")
        self.metadata["last_trace_strategy"] = normalized.get("trace_strategy")
        if not normalized.get("ok"):
            return None
        return normalized["prompt"], normalized["target"], normalized.get("eval_type", "numeric")


class GSM8KTask(NormalizedReasoningTask):
    def map_example(self, ex: Dict[str, Any]):
        return super().map_example(ex)


class MATHTask(NormalizedReasoningTask):
    def map_example(self, ex: Dict[str, Any]):
        return super().map_example(ex)


class MMLUTask(TaskSpec):
    def map_example(self, ex: Dict[str, Any]):
        q = ex.get("question") or ex.get("prompt")
        choices = ex.get("choices") or ex.get("options")
        ans = ex.get("answer")
        if q is None or choices is None or ans is None:
            raise KeyError(f"[{self.name}] bad mmlu example keys={list(ex.keys())}")
        labels = ["A", "B", "C", "D", "E", "F"]
        if isinstance(choices, dict):
            items = [(k, choices[k]) for k in sorted(choices.keys())]
        else:
            items = [(labels[i], choices[i]) for i in range(min(len(choices), len(labels)))]
        opt_text = "\n".join([f"{k}. {v}" for k, v in items])
        if isinstance(ans, int):
            ref = items[ans][0]
        else:
            ref = str(ans).strip()
            if ref.isdigit():
                ref = items[int(ref)][0]
        return f"Question:\n{q}\n\nChoices:\n{opt_text}\n\nAnswer (just the letter):", ref, "mcq"


MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _resolve_effective_max_samples(value) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except Exception:
        return None
    return value if value > 0 else None


def _register_task(tasks: List[TaskSpec], task: TaskSpec, logger=None) -> bool:
    info = inspect_task_dataset(task, logger=logger)
    if not info.get("ok"):
        if logger is not None:
            logger.warning("[Data] disable task=%s reason=%s", task.name, info.get("reason"))
        return False
    sample = info.get("sample")
    if sample is not None:
        try:
            mapped = task.map_example(sample)
            if mapped is None and not getattr(task, "supports_skip", False):
                raise KeyError(f"[{task.name}] sample normalization returned skip")
        except Exception as exc:
            if logger is not None:
                logger.warning("[Data] disable task=%s reason=field validation failed: %r", task.name, exc)
            return False

    details = [f"source={info.get('source')}"]
    if task.max_samples is not None:
        resolved_total = info.get("resolved_samples")
        details.append(f"max_samples={task.max_samples}")
        details.append(f"resolved_samples={min(resolved_total, task.max_samples) if resolved_total is not None else 'unknown'}")
    if task.name == "metamath_train":
        details.append("fields=query,response,original_question")
    if getattr(task, "supports_skip", False):
        policy = task.describe_data_policy()
        length_policy = dict(policy.get("length_policy") or {})
        if length_policy:
            details.append(f"length_policy={length_policy}")
        trace_policy = policy.get("trace_selection_policy")
        if trace_policy:
            details.append(f"trace_policy={trace_policy}")
    if logger is not None:
        logger.info("[Data] enable task=%s %s", task.name, " ".join(details))
    tasks.append(task)
    return True


def _build_reasoning_limits(
    cfg: DataConfig,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    return {
        "max_chars": int(cfg.reasoning_max_chars if max_chars is None else max_chars),
        "max_approx_tokens": int(cfg.reasoning_max_approx_tokens if max_tokens is None else max_tokens),
        "prefer_short_reasoning": bool(cfg.prefer_short_reasoning),
        "skip_overlong_reasoning_samples": bool(cfg.skip_overlong_reasoning_samples),
    }


def build_task_mixture_tasks(cfg: DataConfig, logger=None) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    requested_task_flags = []
    if cfg.use_gsm8k_train:
        requested_task_flags.append("gsm8k_train")
        _register_task(
            tasks,
            GSM8KTask(
                name="gsm8k_train",
                path=cfg.gsm8k_cache_path,
                split="train",
                weight=float(cfg.wt_gsm8k),
                dataset_name="gsm8k",
                length_policy=_build_reasoning_limits(cfg),
                kind="auto",
                hf_name=cfg.gsm8k_hf_name,
                hf_config=normalize_hf_config(cfg.gsm8k_hf_config),
                group="task",
                source_family="reasoning",
            ),
            logger=logger,
        )
    if bool(getattr(cfg, "use_gsm8k_socratic_train", False)):
        requested_task_flags.append("gsm8k_socratic_train")
        _register_task(
            tasks,
            GSM8KTask(
                name="gsm8k_socratic_train",
                path=cfg.gsm8k_socratic_cache_path,
                split="train",
                weight=float(cfg.wt_gsm8k_socratic),
                dataset_name="gsm8k",
                length_policy=_build_reasoning_limits(cfg),
                kind="auto",
                hf_name=cfg.gsm8k_socratic_hf_name,
                hf_config=normalize_hf_config(cfg.gsm8k_socratic_hf_config),
                group="task",
                source_family="reasoning",
            ),
            logger=logger,
        )
    if bool(getattr(cfg, "use_svamp_train", False)):
        requested_task_flags.append("svamp_train")
        _register_task(
            tasks,
            GSM8KTask(
                name="svamp_train",
                path=cfg.svamp_cache_path,
                split="train",
                weight=float(cfg.wt_svamp),
                dataset_name="gsm8k",
                length_policy=_build_reasoning_limits(cfg),
                kind="auto",
                hf_name=cfg.svamp_hf_name,
                hf_config=normalize_hf_config(cfg.svamp_hf_config),
                group="task",
                source_family="reasoning",
            ),
            logger=logger,
        )
    if bool(getattr(cfg, "use_metamath_train", False)):
        requested_task_flags.append("metamath_train")
        _register_task(
            tasks,
            GSM8KTask(
                name="metamath_train",
                path=cfg.metamath_cache_path,
                split="train",
                weight=float(cfg.wt_metamath),
                max_samples=_resolve_effective_max_samples(getattr(cfg, "metamath_max_samples", None)),
                dataset_name="gsm8k",
                length_policy=_build_reasoning_limits(cfg),
                kind="auto",
                hf_name=cfg.metamath_hf_name,
                hf_config=normalize_hf_config(cfg.metamath_hf_config),
                group="task",
                source_family="reasoning",
            ),
            logger=logger,
        )
    if cfg.use_math_train:
        per_subset_weight = float(cfg.wt_math) / max(1, len(MATH_SUBSETS))
        for subset in MATH_SUBSETS:
            requested_task_flags.append(f"math_{subset}")
            _register_task(
                tasks,
                MATHTask(
                    name=f"math_{subset}",
                    path=str(cfg.math_cache_root) + f"/{subset}",
                    split=cfg.math_split,
                    weight=per_subset_weight,
                    dataset_name="hendrycks_math",
                    length_policy=_build_reasoning_limits(cfg),
                    kind="auto",
                    hf_name=cfg.math_hf_name,
                    hf_config=normalize_hf_config(subset),
                    group="task",
                    source_family="math_reasoning",
                ),
                logger=logger,
            )
    if bool(getattr(cfg, "use_openr1_math", False)):
        requested_task_flags.append("openr1_math_train")
        _register_task(
            tasks,
            NormalizedReasoningTask(
                name="openr1_math_train",
                path=cfg.openr1_math_cache_path,
                split="train",
                weight=float(cfg.wt_openr1_math),
                dataset_name="openr1_math",
                length_policy=_build_reasoning_limits(cfg),
                trace_selection_policy="prefer_verified_then_shorter_trace",
                kind="auto",
                hf_name=cfg.openr1_math_hf_name,
                hf_config=normalize_hf_config(cfg.openr1_math_hf_config),
                group="task",
                source_family="math_reasoning",
            ),
            logger=logger,
        )
    if bool(getattr(cfg, "use_numinamath_cot", False)):
        requested_task_flags.append("numinamath_cot_train")
        _register_task(
            tasks,
            NormalizedReasoningTask(
                name="numinamath_cot_train",
                path=cfg.numinamath_cot_cache_path,
                split="train",
                weight=float(cfg.wt_numinamath_cot),
                dataset_name="numinamath_cot",
                length_policy=_build_reasoning_limits(cfg),
                trace_selection_policy="problem_solution_with_answer_fallback",
                kind="auto",
                hf_name=cfg.numinamath_cot_hf_name,
                hf_config=normalize_hf_config(cfg.numinamath_cot_hf_config),
                group="task",
                source_family="math_reasoning",
            ),
            logger=logger,
        )
    if bool(getattr(cfg, "use_openthoughts_math", False)):
        requested_task_flags.append("openthoughts_math_train")
        _register_task(
            tasks,
            NormalizedReasoningTask(
                name="openthoughts_math_train",
                path=cfg.openthoughts_math_cache_path,
                split="train",
                weight=float(cfg.wt_openthoughts_math),
                dataset_name="openthoughts_math",
                length_policy=_build_reasoning_limits(
                    cfg,
                    max_chars=int(cfg.openthoughts_max_chars),
                    max_tokens=int(cfg.openthoughts_max_approx_tokens),
                ),
                trace_selection_policy="prefer_short_correct_conversation_trace",
                kind="auto",
                hf_name=cfg.openthoughts_math_hf_name,
                hf_config=normalize_hf_config(cfg.openthoughts_math_hf_config),
                group="task",
                source_family="math_reasoning",
            ),
            logger=logger,
        )
    if bool(getattr(cfg, "use_bespoke_stratos", False)):
        requested_task_flags.append("bespoke_stratos_train")
        _register_task(
            tasks,
            NormalizedReasoningTask(
                name="bespoke_stratos_train",
                path=cfg.bespoke_stratos_cache_path,
                split="train",
                weight=float(cfg.wt_bespoke_stratos),
                dataset_name="bespoke_stratos",
                length_policy=_build_reasoning_limits(cfg),
                trace_selection_policy="conversation_trace_with_answer_split",
                kind="auto",
                hf_name=cfg.bespoke_stratos_hf_name,
                hf_config=normalize_hf_config(cfg.bespoke_stratos_hf_config),
                group="task",
                source_family="reasoning",
            ),
            logger=logger,
        )
    if cfg.use_mmlu_train:
        requested_task_flags.append("mmlu_auxiliary_train")
        train_split = str(getattr(cfg, "mmlu_train_split", "auxiliary_train"))
        if train_split == "auxiliary_train":
            _register_task(
                tasks,
                MMLUTask(
                    name="mmlu_auxiliary_train",
                    path=cfg.mmlu_cache_path,
                    split=train_split,
                    weight=float(cfg.wt_mmlu),
                    kind="auto",
                    hf_name=cfg.mmlu_hf_name,
                    hf_config=normalize_hf_config(cfg.mmlu_hf_config),
                    group="task",
                    source_family="mcq_reasoning",
                ),
                logger=logger,
            )
        elif logger is not None:
            logger.warning(
                "[Data] disable task=mmlu_auxiliary_train reason=unsupported training split %s (only auxiliary_train is allowed)",
                train_split,
            )
    if requested_task_flags and not tasks:
        raise ValueError(f"reasoning task pool is empty after dataset checks; requested={requested_task_flags}")
    return tasks
