from __future__ import annotations

import importlib
import inspect
import time
from pathlib import Path
from typing import Any, Dict, List

import torch as tc

from ..chat_formatting import tokenizer_supports_chat_template
from ..utils.paths import resolve_path
from .config_utils import resolve_task_eval_settings
from .gsm8k_custom_metrics import build_gsm8k_metric_samples, compute_gsm8k_fallback_metrics, write_jsonl
from .metrics import extract_metric_block, pick_primary_score


def import_lm_eval():
    try:
        evaluator = importlib.import_module("lm_eval.evaluator")
        from lm_eval.models.huggingface import HFLM  # type: ignore

        return evaluator, HFLM
    except Exception as exc:
        raise ImportError('Please install lm-eval with the HF backend: pip install -U "lm_eval[hf]"') from exc


def _supported_param_names(callable_obj) -> set[str]:
    return set(inspect.signature(callable_obj).parameters.keys())


def _is_gsm8k_task(task_name: str) -> bool:
    return "gsm8k" in str(task_name).lower()


def _samples_for_task(raw_res: Dict[str, Any], task_name: str) -> List[Dict[str, Any]]:
    samples = raw_res.get("samples", {}) if isinstance(raw_res, dict) else {}
    if not isinstance(samples, dict):
        return []
    direct = samples.get(task_name)
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    collected: List[Dict[str, Any]] = []
    for key, value in samples.items():
        if isinstance(key, str) and (key == task_name or key.startswith(task_name + "_")) and isinstance(value, list):
            collected.extend(item for item in value if isinstance(item, dict))
    return collected


def _build_hflm_kwargs(model, tokenizer, fit_cfg, task_settings: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], List[str]]:
    _, HFLM = import_lm_eval()
    supported = _supported_param_names(HFLM.__init__)
    backend_cfg = task_settings["backend_config"]
    runtime_cfg = dict(task_settings.get("runtime", {}) or {})
    warnings: List[str] = []
    ignored_runtime_args: List[str] = []

    kwargs: Dict[str, Any] = {
        "pretrained": model,
        "tokenizer": tokenizer,
        "batch_size": int(backend_cfg.get("batch_size", fit_cfg.eval.lm_eval_batch_size)),
        "max_length": int(fit_cfg.data.seq_len_run),
        "trust_remote_code": True,
        "device": str(backend_cfg.get("device", fit_cfg.eval.lm_eval_device)),
    }
    for key in ["enable_thinking", "think_end_token", "chat_template_args"]:
        value = runtime_cfg.get(key)
        if value in (None, {}, []):
            continue
        if key in supported:
            kwargs[key] = value
        else:
            warnings.append(f"lm_eval HFLM does not support runtime arg '{key}', ignored")
            ignored_runtime_args.append(key)

    for key, value in dict(backend_cfg.get("hflm_init", {}) or {}).items():
        if key in supported:
            kwargs[key] = value
        else:
            warnings.append(f"lm_eval HFLM init arg '{key}' is unsupported in current version and was ignored")
            ignored_runtime_args.append(key)
    return kwargs, warnings, ignored_runtime_args


def make_hflm(model, tokenizer, fit_cfg, task_settings: Dict[str, Any]):
    _, HFLM = import_lm_eval()
    ctor_kwargs, warnings, ignored_runtime_args = _build_hflm_kwargs(model, tokenizer, fit_cfg, task_settings)
    ctor_errors = []
    candidate_kwargs = [
        dict(ctor_kwargs),
        {k: v for k, v in ctor_kwargs.items() if k != "max_length"},
        {k: v for k, v in ctor_kwargs.items() if k not in {"max_length", "device", "trust_remote_code"}},
    ]
    for kwargs in candidate_kwargs:
        try:
            return HFLM(**kwargs), warnings, ignored_runtime_args
        except TypeError as exc:
            ctor_errors.append(f"{kwargs} -> {repr(exc)}")
    raise RuntimeError("Unable to construct HFLM.\n" + "\n".join(ctor_errors))


@tc.no_grad()
def run_lm_eval_tasks(model, tokenizer, fit_cfg, tasks: List[str], logger=None, eval_name: str = "eval", out_root: str | Path | None = None, eval_mode: str = "final") -> Dict[str, Any]:
    evaluator, _ = import_lm_eval()
    simple_supported = _supported_param_names(evaluator.simple_evaluate)
    model.eval()
    output_root = out_root if out_root is not None else "${OUTPUT_ROOT}/lm_eval_outputs"
    output_dir = Path(resolve_path(str(output_root), key="eval.lm_eval.output_root", source="config", cfg=fit_cfg, allow_none=False)) / eval_name
    output_dir.mkdir(parents=True, exist_ok=True)
    task_to_result: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    backend_warnings: List[str] = []

    for task_name in tasks:
        task_settings = resolve_task_eval_settings(fit_cfg, task_name, eval_mode, backend="lm_eval")
        hflm, hflm_warnings, ignored_runtime_args = make_hflm(model, tokenizer, fit_cfg, task_settings)
        warnings = list(hflm_warnings)
        runtime_cfg = dict(task_settings.get("runtime", {}) or {})
        backend_cfg = dict(task_settings.get("backend_config", {}) or {})
        gen_kwargs = task_settings.get("gen_kwargs")
        ignored_simple_args: List[str] = []
        if runtime_cfg.get("apply_chat_template") is True and not tokenizer_supports_chat_template(tokenizer):
            raise ValueError("eval.runtime.apply_chat_template=true requires tokenizer.apply_chat_template")
        if logger is not None:
            logger.info(
                "[EvalRuntime:%s] backend=lm_eval task=%s apply_chat_template=%s enable_thinking=%s",
                eval_name,
                task_name,
                runtime_cfg.get("apply_chat_template"),
                runtime_cfg.get("enable_thinking"),
            )

        for key in task_settings.get("unsupported_generation_keys", []):
            warnings.append(f"lm_eval backend does not recognize generation key '{key}', ignored")
        for key in task_settings.get("unsupported_runtime_keys", []):
            warnings.append(f"lm_eval backend does not recognize runtime key '{key}', ignored")
            ignored_runtime_args.append(key)

        call_kwargs: Dict[str, Any] = {
            "model": hflm,
            "tasks": [task_name],
            "num_fewshot": int(task_settings["fewshot"]),
            "batch_size": int(backend_cfg.get("batch_size", fit_cfg.eval.lm_eval_batch_size)),
            "log_samples": bool(getattr(fit_cfg.eval, "save_samples", False)),
        }
        if task_settings.get("limit") is not None:
            call_kwargs["limit"] = task_settings["limit"]
        if gen_kwargs is not None:
            call_kwargs["gen_kwargs"] = gen_kwargs

        optional_simple_args = {
            "apply_chat_template": runtime_cfg.get("apply_chat_template"),
            "fewshot_as_multiturn": runtime_cfg.get("fewshot_as_multiturn"),
            "system_instruction": runtime_cfg.get("system_instruction"),
            "metadata": {
                "backend": "lm_eval",
                "protocol": task_settings.get("protocol"),
                "runtime": runtime_cfg,
                "generation": task_settings.get("generation"),
            },
            "confirm_run_unsafe_code": True,
        }
        for key, value in optional_simple_args.items():
            if value is None:
                continue
            if key in simple_supported:
                call_kwargs[key] = value
            else:
                warnings.append(f"lm_eval simple_evaluate does not support arg '{key}', ignored")
                ignored_simple_args.append(key)

        for key, value in dict(backend_cfg.get("simple_evaluate", {}) or {}).items():
            if key in simple_supported:
                call_kwargs[key] = value
            else:
                warnings.append(f"lm_eval simple_evaluate arg '{key}' is unsupported in current version and was ignored")
                ignored_simple_args.append(key)

        custom_gsm8k_enabled = bool(getattr(fit_cfg.eval, "gsm8k_custom_metrics", False)) and _is_gsm8k_task(task_name)
        if custom_gsm8k_enabled:
            if "log_samples" in simple_supported:
                call_kwargs["log_samples"] = True
            else:
                warnings.append("GSM8K custom metrics require lm_eval sample logging, but simple_evaluate does not support log_samples")

        raw_res = None
        errors = []
        candidate_call_kwargs = [
            dict(call_kwargs),
            {k: v for k, v in call_kwargs.items() if k != "confirm_run_unsafe_code"},
            {k: v for k, v in call_kwargs.items() if k not in {"confirm_run_unsafe_code", "metadata"}},
        ]
        for attempt_kwargs in candidate_call_kwargs:
            try:
                raw_res = evaluator.simple_evaluate(**attempt_kwargs)
                break
            except Exception as exc:
                errors.append(f"{attempt_kwargs} -> {repr(exc)}")

        if raw_res is None:
            task_to_result[task_name] = {
                "backend": "lm_eval",
                "protocol": task_settings.get("protocol"),
                "primary_metric": None,
                "primary_score": None,
                "metrics": {},
                "fewshot": int(task_settings["fewshot"]),
                "limit": task_settings.get("limit"),
                "runtime": runtime_cfg,
                "gen_kwargs": gen_kwargs,
                "warnings": warnings,
                "ignored_runtime_args": sorted(set(ignored_runtime_args + ignored_simple_args)),
                "unsupported_runtime_args": sorted(set(task_settings.get("unsupported_runtime_keys", []))),
                "unsupported_generation_args": sorted(set(task_settings.get("unsupported_generation_keys", []))),
                "error": " | ".join(errors),
            }
            summary[task_name] = {
                "primary_metric": None,
                "primary_score": None,
                "fewshot": int(task_settings["fewshot"]),
                "limit": task_settings.get("limit"),
                "backend": "lm_eval",
                "protocol": task_settings.get("protocol"),
                "warnings": warnings,
                "error": " | ".join(errors),
            }
            backend_warnings.extend(warnings)
            continue

        metric_block = extract_metric_block(raw_res, task_name)
        primary_metric, primary_score = pick_primary_score(task_name, metric_block)
        task_result = {
            "backend": "lm_eval",
            "protocol": task_settings.get("protocol"),
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "metrics": metric_block,
            "fewshot": int(task_settings["fewshot"]),
            "limit": task_settings.get("limit"),
            "runtime": runtime_cfg,
            "gen_kwargs": gen_kwargs,
            "output_dir": str(output_dir),
            "warnings": warnings,
            "ignored_runtime_args": sorted(set(ignored_runtime_args + ignored_simple_args)),
            "unsupported_runtime_args": sorted(set(task_settings.get("unsupported_runtime_keys", []))),
            "unsupported_generation_args": sorted(set(task_settings.get("unsupported_generation_keys", []))),
            "raw": raw_res,
        }
        task_summary = {
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "fewshot": int(task_settings["fewshot"]),
            "limit": task_settings.get("limit"),
                "backend": "lm_eval",
                "protocol": task_settings.get("protocol"),
                "runtime": runtime_cfg,
                "apply_chat_template": runtime_cfg.get("apply_chat_template"),
                "enable_thinking": runtime_cfg.get("enable_thinking"),
                "gen_kwargs": gen_kwargs,
                "warnings": warnings,
            }

        if custom_gsm8k_enabled:
            lm_eval_samples = _samples_for_task(raw_res, task_name)
            if not lm_eval_samples:
                custom_warning = "GSM8K custom metrics requested, but lm_eval did not return sample logs for this task"
                warnings.append(custom_warning)
                empty_custom_metrics = {
                    "strict_hash_acc": None,
                    "fallback_last_number_acc": None,
                    "last_number_acc": None,
                    "has_hash_answer_rate": None,
                    "correct_but_no_hash_rate": None,
                    "fallback_minus_strict": None,
                    "strict_correct_fallback_wrong_count": 0,
                    "fallback_correct_strict_wrong_count": 0,
                    "num_samples": 0,
                    "num_total_samples": 0,
                    "warnings": [custom_warning],
                }
                task_result.update(
                    {
                        "custom_metrics": empty_custom_metrics,
                        "lm_eval_strict": metric_block.get("exact_match,strict-match"),
                        "lm_eval_flexible_extract": metric_block.get("exact_match,flexible-extract"),
                        "fallback_last_number": None,
                    }
                )
                task_summary.update(
                    {
                        "lm_eval_strict": metric_block.get("exact_match,strict-match"),
                        "lm_eval_flexible_extract": metric_block.get("exact_match,flexible-extract"),
                        "fallback_last_number": None,
                    }
                )
                task_summary["custom_metric_warnings"] = [custom_warning]
            else:
                metric_samples = build_gsm8k_metric_samples(task_name, lm_eval_samples)
                custom_result = compute_gsm8k_fallback_metrics(metric_samples)
                custom_summary = dict(custom_result["summary"])
                records = list(custom_result["records"])
                sample_path = output_dir / f"{task_name}_custom_samples.jsonl"
                write_jsonl(sample_path, records)
                custom_warnings = list(custom_result.get("warnings", []) or [])
                if custom_warnings:
                    warnings.extend(custom_warnings)
                task_result.update(
                    {
                        "custom_metrics": custom_summary,
                        "custom_samples_path": str(sample_path),
                        "lm_eval_strict": metric_block.get("exact_match,strict-match"),
                        "lm_eval_flexible_extract": metric_block.get("exact_match,flexible-extract"),
                        "fallback_last_number": custom_summary.get("fallback_last_number_acc"),
                    }
                )
                task_summary.update(
                    {
                        "lm_eval_strict": metric_block.get("exact_match,strict-match"),
                        "lm_eval_flexible_extract": metric_block.get("exact_match,flexible-extract"),
                        "fallback_last_number": custom_summary.get("fallback_last_number_acc"),
                        "custom_metric_warnings": custom_warnings,
                    }
                )

        task_result["warnings"] = warnings
        task_summary["warnings"] = warnings
        task_to_result[task_name] = task_result
        summary[task_name] = task_summary
        backend_warnings.extend(warnings)
        if logger is not None:
            logger.info(
                f"[LM-Eval:{eval_name}] task={task_name} primary_metric={primary_metric} primary_score={primary_score} protocol={task_settings.get('protocol')}"
            )
    return {
        "backend": "lm_eval",
        "eval_name": eval_name,
        "tasks": task_to_result,
        "summary": summary,
        "time": time.time(),
        "output_dir": str(output_dir),
        "eval_mode": eval_mode,
        "warnings": sorted(set(backend_warnings)),
    }
