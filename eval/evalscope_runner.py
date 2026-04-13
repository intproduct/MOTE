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
    generation_cfg = dict(task_settings.get("gen_kwargs", {}) or {})
    task_cfg: Dict[str, Any] = {
        "model": str(model_or_path),
        "datasets": [dataset_name],
        "work_dir": str(work_dir),
        "limit": task_settings.get("limit"),
        "generation_config": generation_cfg,
        "eval_batch_size": int(backend_cfg.get("batch_size", 1)),
    }
    if task_settings.get("fewshot") is not None:
        task_cfg["few_shot_num"] = int(task_settings["fewshot"])
        task_cfg["num_fewshot"] = int(task_settings["fewshot"])
    if backend_cfg.get("device") is not None:
        task_cfg["device"] = backend_cfg.get("device")
    for key, value in dict(backend_cfg.get("task_config", {}) or {}).items():
        task_cfg[key] = value
    return {k: v for k, v in task_cfg.items() if v is not None}


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
                "gen_kwargs": task_settings.get("gen_kwargs"),
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
        task_cfg_dict = _build_evalscope_task_cfg(str(model_or_path), task_name, task_settings, task_dir)
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
                "gen_kwargs": task_settings.get("gen_kwargs"),
                "warnings": task_warnings,
                "status": "error",
                "error": error_text,
                "raw": None,
            }
            summary[task_name] = {
                "backend": "evalscope",
                "primary_metric": None,
                "primary_score": None,
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
            "gen_kwargs": task_settings.get("gen_kwargs"),
            "warnings": task_warnings,
            "raw": raw_res,
            "status": "ok",
            "output_dir": str(task_dir),
        }
        summary[task_name] = {
            "backend": "evalscope",
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "fewshot": int(task_settings["fewshot"]),
            "limit": task_settings.get("limit"),
            "gen_kwargs": task_settings.get("gen_kwargs"),
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
