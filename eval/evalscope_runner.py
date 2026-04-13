from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_utils import resolve_task_eval_settings
from .metrics import pick_primary_score


TASK_NAME_MAP = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "hendrycks_math": "hendrycks_math",
}
SUPPORTED_TASKS = set(TASK_NAME_MAP.keys())


def import_evalscope():
    try:
        module = importlib.import_module("evalscope")
        run_task = getattr(module, "run_task")
        task_config = getattr(module, "TaskConfig")
        return run_task, task_config
    except Exception as exc:
        raise ImportError("EvalScope backend requires 'evalscope'. Install it with: pip install evalscope") from exc


def _make_skipped_backend_result(eval_name: str, eval_mode: str, tasks: List[str], reason: str, warnings: List[str]) -> Dict[str, Any]:
    task_results = {}
    summary = {}
    for task_name in tasks:
        task_results[task_name] = {
            "backend": "evalscope",
            "primary_metric": None,
            "primary_score": None,
            "metrics": {},
            "fewshot": None,
            "limit": None,
            "gen_kwargs": None,
            "runtime": {},
            "runtime_effects": {},
            "generation_effects": {},
            "ignored_runtime_args": [],
            "unsupported_runtime_args": [],
            "unsupported_generation_args": [],
            "warnings": warnings,
            "status": "skipped",
            "skip_reason": reason,
            "raw": None,
        }
        summary[task_name] = {
            "backend": "evalscope",
            "primary_metric": None,
            "primary_score": None,
            "warnings": warnings,
            "status": "skipped",
            "skip_reason": reason,
        }
    return {
        "backend": "evalscope",
        "eval_name": eval_name,
        "tasks": task_results,
        "summary": summary,
        "time": time.time(),
        "eval_mode": eval_mode,
        "warnings": warnings,
        "status": "skipped",
        "skip_reason": reason,
    }


def _collect_numeric_metrics(obj: Any, prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (int, float)):
                metrics[next_prefix] = float(value)
            else:
                metrics.update(_collect_numeric_metrics(value, next_prefix))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            metrics.update(_collect_numeric_metrics(item, f"{prefix}[{idx}]" if prefix else f"[{idx}]"))
    return metrics


def _extract_evalscope_metric_block(raw_res: Any, task_name: str) -> Dict[str, Any]:
    if isinstance(raw_res, dict):
        for key in [task_name, TASK_NAME_MAP.get(task_name, task_name), "results", "report", "summary", "metrics"]:
            value = raw_res.get(key) if key in raw_res else None
            if isinstance(value, dict):
                numeric = {k: v for k, v in _collect_numeric_metrics(value).items() if isinstance(v, (int, float, float))}
                if numeric:
                    return numeric
            if isinstance(value, list):
                collected = _collect_numeric_metrics(value)
                if collected:
                    return collected
        collected = _collect_numeric_metrics(raw_res)
        if collected:
            return collected
    elif isinstance(raw_res, list):
        collected = _collect_numeric_metrics(raw_res)
        if collected:
            return collected
    return {}


def _build_evalscope_task_cfg(model_or_path: str, task_name: str, task_settings: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    dataset_name = TASK_NAME_MAP[task_name]
    backend_cfg = dict(task_settings.get("backend_config", {}) or {})
    runtime_cfg = dict(task_settings.get("runtime", {}) or {})
    raw_generation_cfg = dict(task_settings.get("gen_kwargs", {}) or {})
    generation_cfg: Dict[str, Any] = {}
    generation_effects: Dict[str, Dict[str, Any]] = {}
    runtime_effects: Dict[str, Dict[str, Any]] = {}

    for key, value in raw_generation_cfg.items():
        if value is None:
            continue
        mapped_key = "max_tokens" if key == "max_gen_toks" else key
        generation_cfg[mapped_key] = value
        generation_effects[key] = {
            "status": "effective",
            "target": f"generation_config.{mapped_key}",
            "value": value,
        }

    chat_template_kwargs = dict(runtime_cfg.get("chat_template_args", {}) or {})
    if runtime_cfg.get("apply_chat_template") is not None:
        runtime_effects["apply_chat_template"] = {
            "status": "best_effort",
            "target": "chat_template",
            "value": bool(runtime_cfg.get("apply_chat_template")),
        }
    if runtime_cfg.get("enable_thinking") is not None:
        chat_template_kwargs["enable_thinking"] = bool(runtime_cfg.get("enable_thinking"))
        runtime_effects["enable_thinking"] = {
            "status": "best_effort",
            "target": "generation_config.chat_template_kwargs.enable_thinking",
            "value": bool(runtime_cfg.get("enable_thinking")),
        }
    if runtime_cfg.get("think_end_token"):
        chat_template_kwargs["think_end_token"] = runtime_cfg.get("think_end_token")
        runtime_effects["think_end_token"] = {
            "status": "best_effort",
            "target": "generation_config.chat_template_kwargs.think_end_token",
            "value": runtime_cfg.get("think_end_token"),
        }
    if runtime_cfg.get("chat_template_args"):
        runtime_effects["chat_template_args"] = {
            "status": "best_effort",
            "target": "generation_config.chat_template_kwargs",
            "value": dict(runtime_cfg.get("chat_template_args", {}) or {}),
        }
    if runtime_cfg.get("system_instruction"):
        runtime_effects["system_instruction"] = {
            "status": "effective",
            "target": f"dataset_args.{task_name}.system_prompt",
            "value": runtime_cfg.get("system_instruction"),
        }
    if runtime_cfg.get("fewshot_as_multiturn") is not None:
        runtime_effects["fewshot_as_multiturn"] = {
            "status": "record_only",
            "target": None,
            "value": bool(runtime_cfg.get("fewshot_as_multiturn")),
            "reason": "EvalScope does not expose a documented few-shot multi-turn switch in TaskConfig.",
        }

    if chat_template_kwargs:
        generation_cfg["chat_template_kwargs"] = chat_template_kwargs

    task_cfg: Dict[str, Any] = {
        "model": str(model_or_path),
        "datasets": [dataset_name],
        "work_dir": str(work_dir),
        "limit": task_settings.get("limit"),
        "generation_config": generation_cfg,
        "eval_batch_size": int(backend_cfg.get("batch_size", 1)),
        "dataset_args": {
            task_name: {
                "few_shot_num": int(task_settings["fewshot"]) if task_settings.get("fewshot") is not None else None,
                "system_prompt": runtime_cfg.get("system_instruction"),
            }
        },
        "model_args": dict(backend_cfg.get("model_args", {}) or {}),
    }
    if task_settings.get("fewshot") is not None:
        task_cfg["few_shot_num"] = int(task_settings["fewshot"])
        task_cfg["num_fewshot"] = int(task_settings["fewshot"])
        runtime_effects["fewshot"] = {
            "status": "effective",
            "target": f"dataset_args.{task_name}.few_shot_num",
            "value": int(task_settings["fewshot"]),
        }
    if backend_cfg.get("device") is not None:
        task_cfg["device"] = backend_cfg.get("device")
        task_cfg["model_args"]["device"] = backend_cfg.get("device")
    for key, value in dict(backend_cfg.get("task_config", {}) or {}).items():
        task_cfg[key] = value
    if runtime_cfg.get("apply_chat_template") is True:
        task_cfg["chat_template"] = True
    if not task_cfg["dataset_args"][task_name]["few_shot_num"] and task_cfg["dataset_args"][task_name]["few_shot_num"] != 0:
        task_cfg["dataset_args"][task_name].pop("few_shot_num", None)
    if not task_cfg["dataset_args"][task_name]["system_prompt"]:
        task_cfg["dataset_args"][task_name].pop("system_prompt", None)
    if not task_cfg["dataset_args"][task_name]:
        task_cfg.pop("dataset_args", None)
    if not task_cfg["model_args"]:
        task_cfg.pop("model_args", None)
    task_cfg = {k: v for k, v in task_cfg.items() if v is not None}
    return task_cfg, runtime_effects, generation_effects


def run_evalscope_tasks(
    fit_cfg,
    tasks: List[str],
    logger=None,
    eval_name: str = "eval",
    out_root: str | Path | None = None,
    eval_mode: str = "final",
    model_or_path: str | Path | None = None,
    allow_skip: bool = False,
) -> Dict[str, Any]:
    warnings: List[str] = []
    unresolved_tasks = [task_name for task_name in tasks if task_name not in SUPPORTED_TASKS]
    if unresolved_tasks:
        warnings.extend([f"EvalScope backend does not support task '{task_name}' yet; skipped" for task_name in unresolved_tasks])

    if model_or_path is None:
        reason = "evalscope requires model_or_path/checkpoint path and cannot evaluate in-memory patched nn.Module"
        if allow_skip:
            warnings.append(reason)
            return _make_skipped_backend_result(eval_name, eval_mode, tasks, reason, warnings)
        raise ValueError(reason)

    try:
        run_task, TaskConfig = import_evalscope()
    except ImportError as exc:
        if allow_skip:
            warnings.append(str(exc))
            return _make_skipped_backend_result(eval_name, eval_mode, tasks, str(exc), warnings)
        raise

    output_dir = Path(out_root or "./evalscope_outputs").resolve() / eval_name
    output_dir.mkdir(parents=True, exist_ok=True)
    task_to_result: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}

    for task_name in tasks:
        task_settings = resolve_task_eval_settings(fit_cfg, task_name, eval_mode, backend="evalscope")
        task_warnings = list(warnings)
        if task_name not in SUPPORTED_TASKS:
            task_to_result[task_name] = {
                "backend": "evalscope",
                "protocol": task_settings.get("protocol"),
                "primary_metric": None,
                "primary_score": None,
                "metrics": {},
                "fewshot": int(task_settings["fewshot"]),
                "limit": task_settings.get("limit"),
                "runtime": dict(task_settings.get("runtime", {}) or {}),
                "gen_kwargs": task_settings.get("gen_kwargs"),
                "runtime_effects": {},
                "generation_effects": {},
                "ignored_runtime_args": [],
                "unsupported_runtime_args": sorted(set(task_settings.get("unsupported_runtime_keys", []))),
                "unsupported_generation_args": sorted(set(task_settings.get("unsupported_generation_keys", []))),
                "warnings": task_warnings,
                "status": "skipped",
                "skip_reason": f"unsupported task '{task_name}'",
                "raw": None,
            }
            summary[task_name] = {
                "backend": "evalscope",
                "primary_metric": None,
                "primary_score": None,
                "warnings": task_warnings,
                "status": "skipped",
                "skip_reason": f"unsupported task '{task_name}'",
            }
            continue

        for key in task_settings.get("unsupported_generation_keys", []):
            task_warnings.append(f"EvalScope backend does not recognize generation key '{key}', ignored")
        for key in task_settings.get("unsupported_runtime_keys", []):
            task_warnings.append(f"EvalScope backend does not recognize runtime key '{key}', ignored")

        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        task_cfg_dict, runtime_effects, generation_effects = _build_evalscope_task_cfg(str(model_or_path), task_name, task_settings, task_dir)
        ignored_runtime_args = [
            key
            for key, effect in runtime_effects.items()
            if str(effect.get("status")) == "record_only"
        ]
        task_warnings.extend(
            [
                f"EvalScope runtime arg '{key}' recorded only: {runtime_effects[key].get('reason')}"
                for key in ignored_runtime_args
            ]
        )
        raw_res = None
        error_text = None
        try:
            task_cfg = TaskConfig(**task_cfg_dict)
            raw_res = run_task(task_cfg=task_cfg)
        except TypeError:
            try:
                raw_res = run_task(task_cfg=task_cfg_dict)
            except Exception as exc:
                error_text = repr(exc)
        except Exception as exc:
            error_text = repr(exc)

        if raw_res is None and error_text is not None:
            task_to_result[task_name] = {
                "backend": "evalscope",
                "protocol": task_settings.get("protocol"),
                "primary_metric": None,
                "primary_score": None,
                "metrics": {},
                "fewshot": int(task_settings["fewshot"]),
                "limit": task_settings.get("limit"),
                "runtime": dict(task_settings.get("runtime", {}) or {}),
                "gen_kwargs": task_settings.get("gen_kwargs"),
                "runtime_effects": runtime_effects,
                "generation_effects": generation_effects,
                "ignored_runtime_args": ignored_runtime_args,
                "unsupported_runtime_args": sorted(set(task_settings.get("unsupported_runtime_keys", []))),
                "unsupported_generation_args": sorted(set(task_settings.get("unsupported_generation_keys", []))),
                "warnings": task_warnings,
                "status": "error",
                "error": error_text,
                "task_config": task_cfg_dict,
                "raw": None,
            }
            summary[task_name] = {
                "backend": "evalscope",
                "primary_metric": None,
                "primary_score": None,
                "runtime": dict(task_settings.get("runtime", {}) or {}),
                "warnings": task_warnings,
                "status": "error",
                "error": error_text,
            }
            continue

        metric_block = _extract_evalscope_metric_block(raw_res, task_name)
        primary_metric, primary_score = pick_primary_score(task_name, metric_block)
        task_to_result[task_name] = {
            "backend": "evalscope",
            "protocol": task_settings.get("protocol"),
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "metrics": metric_block,
            "fewshot": int(task_settings["fewshot"]),
            "limit": task_settings.get("limit"),
            "runtime": dict(task_settings.get("runtime", {}) or {}),
            "gen_kwargs": task_settings.get("gen_kwargs"),
            "runtime_effects": runtime_effects,
            "generation_effects": generation_effects,
            "ignored_runtime_args": ignored_runtime_args,
            "unsupported_runtime_args": sorted(set(task_settings.get("unsupported_runtime_keys", []))),
            "unsupported_generation_args": sorted(set(task_settings.get("unsupported_generation_keys", []))),
            "warnings": task_warnings,
            "raw": raw_res,
            "status": "ok",
            "output_dir": str(task_dir),
            "task_config": task_cfg_dict,
        }
        summary[task_name] = {
            "backend": "evalscope",
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "fewshot": int(task_settings["fewshot"]),
            "limit": task_settings.get("limit"),
            "runtime": dict(task_settings.get("runtime", {}) or {}),
            "gen_kwargs": task_settings.get("gen_kwargs"),
            "runtime_effects": runtime_effects,
            "generation_effects": generation_effects,
            "warnings": task_warnings,
        }
        if logger is not None:
            logger.info(
                f"[EvalScope:{eval_name}] task={task_name} primary_metric={primary_metric} primary_score={primary_score} protocol={task_settings.get('protocol')}"
            )

    return {
        "backend": "evalscope",
        "eval_name": eval_name,
        "tasks": task_to_result,
        "summary": summary,
        "time": time.time(),
        "output_dir": str(output_dir),
        "eval_mode": eval_mode,
        "warnings": sorted(set(warnings)),
    }
