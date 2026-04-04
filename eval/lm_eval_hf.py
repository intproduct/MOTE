from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch as tc

from .metrics import extract_metric_block, pick_primary_score


def import_lm_eval():
    try:
        evaluator = importlib.import_module("lm_eval.evaluator")
        from lm_eval.models.huggingface import HFLM  # type: ignore

        return evaluator, HFLM
    except Exception as exc:
        raise ImportError('Please install lm-eval with the HF backend: pip install -U "lm_eval[hf]"') from exc


def fewshot_for_task(task_name: str, eval_cfg) -> int:
    if task_name == "gsm8k":
        return int(eval_cfg.lm_eval_num_fewshot_gsm8k)
    if task_name == "mmlu":
        return int(eval_cfg.lm_eval_num_fewshot_mmlu)
    if task_name == "hendrycks_math":
        return int(eval_cfg.lm_eval_num_fewshot_math)
    return 0


def limit_for_task(task_name: str, eval_cfg, eval_mode: str) -> Optional[int]:
    if eval_mode == "baseline_small":
        mapping = {"gsm8k": eval_cfg.baseline_small_limit_gsm8k, "mmlu": eval_cfg.baseline_small_limit_mmlu}
    elif eval_mode == "mid":
        mapping = {"gsm8k": eval_cfg.early_limit_gsm8k, "mmlu": eval_cfg.early_limit_mmlu}
    elif eval_mode == "final":
        mapping = {"gsm8k": eval_cfg.final_limit_gsm8k, "mmlu": eval_cfg.final_limit_mmlu, "hendrycks_math": eval_cfg.final_limit_math}
    else:
        raise ValueError(f"unknown eval_mode={eval_mode}")
    value = int(mapping.get(task_name, 0))
    return None if value <= 0 else value


def gen_kwargs_for_task(task_name: str, eval_cfg, eval_mode: str) -> Optional[Dict[str, Any]]:
    if task_name == "gsm8k":
        max_gen_toks = eval_cfg.early_max_gen_toks_gsm8k if eval_mode in ("baseline_small", "mid") else eval_cfg.final_max_gen_toks_gsm8k
        return {"max_gen_toks": int(max_gen_toks)}
    if task_name == "hendrycks_math" and eval_mode == "final":
        return {"max_gen_toks": int(eval_cfg.final_max_gen_toks_math)}
    return None


def make_hflm(model, tokenizer, fit_cfg):
    _, HFLM = import_lm_eval()
    ctor_errors = []
    for kwargs in [
        dict(pretrained=model, tokenizer=tokenizer, batch_size=int(fit_cfg.eval.lm_eval_batch_size), max_length=int(fit_cfg.data.seq_len_run), trust_remote_code=True, device=str(fit_cfg.eval.lm_eval_device)),
        dict(pretrained=model, tokenizer=tokenizer, batch_size=int(fit_cfg.eval.lm_eval_batch_size), trust_remote_code=True, device=str(fit_cfg.eval.lm_eval_device)),
        dict(pretrained=model, tokenizer=tokenizer, batch_size=int(fit_cfg.eval.lm_eval_batch_size)),
    ]:
        try:
            return HFLM(**kwargs)
        except TypeError as exc:
            ctor_errors.append(f"{kwargs} -> {repr(exc)}")
    raise RuntimeError("Unable to construct HFLM.\n" + "\n".join(ctor_errors))


@tc.no_grad()
def run_lm_eval_tasks(model, tokenizer, fit_cfg, tasks: List[str], logger=None, eval_name: str = "eval", out_root: str | Path | None = None, eval_mode: str = "final") -> Dict[str, Any]:
    evaluator, _ = import_lm_eval()
    model.eval()
    hflm = make_hflm(model, tokenizer, fit_cfg)
    output_dir = Path(out_root or "./lm_eval_outputs").resolve() / eval_name
    output_dir.mkdir(parents=True, exist_ok=True)
    task_to_result: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for task_name in tasks:
        num_fewshot = fewshot_for_task(task_name, fit_cfg.eval)
        limit = limit_for_task(task_name, fit_cfg.eval, eval_mode)
        gen_kwargs = gen_kwargs_for_task(task_name, fit_cfg.eval, eval_mode)
        raw_res = None
        errors = []
        for call_kwargs in [
            dict(model=hflm, tasks=[task_name], num_fewshot=num_fewshot, batch_size=int(fit_cfg.eval.lm_eval_batch_size), log_samples=False, confirm_run_unsafe_code=True),
            dict(model=hflm, tasks=[task_name], num_fewshot=num_fewshot, batch_size=int(fit_cfg.eval.lm_eval_batch_size), log_samples=False),
            dict(model=hflm, tasks=[task_name], num_fewshot=num_fewshot, batch_size=int(fit_cfg.eval.lm_eval_batch_size)),
        ]:
            if limit is not None:
                call_kwargs["limit"] = limit
            if gen_kwargs is not None:
                call_kwargs["gen_kwargs"] = gen_kwargs
            try:
                raw_res = evaluator.simple_evaluate(**call_kwargs)
                break
            except Exception as exc:
                errors.append(f"{call_kwargs} -> {repr(exc)}")
        if raw_res is None:
            task_to_result[task_name] = {"primary_metric": None, "primary_score": None, "metrics": {}, "error": " | ".join(errors)}
            summary[task_name] = {"primary_metric": None, "primary_score": None, "error": " | ".join(errors)}
            continue
        metric_block = extract_metric_block(raw_res, task_name)
        primary_metric, primary_score = pick_primary_score(task_name, metric_block)
        task_to_result[task_name] = {
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "metrics": metric_block,
            "fewshot": num_fewshot,
            "limit": limit,
            "gen_kwargs": gen_kwargs,
            "output_dir": str(output_dir),
            "raw": raw_res,
        }
        summary[task_name] = {"primary_metric": primary_metric, "primary_score": primary_score, "fewshot": num_fewshot, "limit": limit}
        if logger is not None:
            logger.info(f"[LM-Eval:{eval_name}] task={task_name} primary_metric={primary_metric} primary_score={primary_score}")
    return {"eval_name": eval_name, "tasks": task_to_result, "summary": summary, "time": time.time(), "output_dir": str(output_dir), "eval_mode": eval_mode}
