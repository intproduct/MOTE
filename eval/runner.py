from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from .config_utils import get_enabled_backends, get_primary_backend
from .evalscope_runner import run_evalscope_tasks
from .lm_eval_hf import run_lm_eval_tasks
from ..utils.paths import assert_no_unsafe_paths


def get_primary_backend_result(eval_res: Dict[str, Any]) -> Dict[str, Any]:
    if "results" not in eval_res:
        return eval_res
    primary_backend = eval_res.get("primary_backend")
    if primary_backend is None:
        raise KeyError("eval result is missing primary_backend")
    return dict(eval_res.get("results", {}).get(primary_backend, {}) or {})


def run_eval_tasks(
    fit_cfg,
    tasks: List[str],
    logger=None,
    eval_name: str = "eval",
    out_root: str | Path | None = None,
    eval_mode: str = "final",
    model=None,
    tokenizer=None,
    model_or_path: str | Path | None = None,
    allow_backend_skip: bool = False,
) -> Dict[str, Any]:
    enabled_backends = get_enabled_backends(fit_cfg.eval)
    primary_backend = get_primary_backend(fit_cfg.eval)
    started_at = time.time()
    backend_results: Dict[str, Any] = {}
    warnings: List[str] = []
    assert_no_unsafe_paths(fit_cfg, context="eval", logger=logger)

    if "lm_eval" in enabled_backends:
        if model is None or tokenizer is None:
            raise ValueError("lm_eval backend requires model and tokenizer")
        backend_results["lm_eval"] = run_lm_eval_tasks(
            model,
            tokenizer,
            fit_cfg,
            tasks=tasks,
            logger=logger,
            eval_name=eval_name,
            out_root=out_root,
            eval_mode=eval_mode,
        )
        warnings.extend(list(backend_results["lm_eval"].get("warnings", []) or []))

    if "evalscope" in enabled_backends:
        evalscope_out_root = None
        if out_root is not None:
            evalscope_out_root = Path(out_root).parent / "evalscope_outputs"
        backend_results["evalscope"] = run_evalscope_tasks(
            fit_cfg,
            tasks=tasks,
            logger=logger,
            eval_name=eval_name,
            out_root=evalscope_out_root,
            eval_mode=eval_mode,
            model_or_path=model_or_path,
            allow_skip=allow_backend_skip,
        )
        warnings.extend(list(backend_results["evalscope"].get("warnings", []) or []))

    primary_result = dict(backend_results.get(primary_backend, {}) or {})
    if not primary_result:
        raise RuntimeError(f"Primary eval backend {primary_backend!r} produced no result")

    return {
        "eval_name": eval_name,
        "eval_mode": eval_mode,
        "primary_backend": primary_backend,
        "results": backend_results,
        "tasks": dict(primary_result.get("tasks", {}) or {}),
        "summary": dict(primary_result.get("summary", {}) or {}),
        "time": time.time(),
        "duration_sec": max(0.0, time.time() - started_at),
        "warnings": sorted(set(warnings)),
    }
