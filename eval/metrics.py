from __future__ import annotations

from typing import Any, Dict, Optional


def extract_metric_block(raw_res: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    results_block = raw_res.get("results", {}) or {}
    groups_block = raw_res.get("groups", {}) or {}
    if task_name in groups_block and isinstance(groups_block[task_name], dict):
        return groups_block[task_name]
    if task_name in results_block and isinstance(results_block[task_name], dict):
        return results_block[task_name]
    merged: Dict[str, list[float]] = {}
    found = False
    for src in [groups_block, results_block]:
        for key, value in src.items():
            if isinstance(value, dict) and (key == task_name or key.startswith(task_name + "_")):
                found = True
                for metric_key, metric_value in value.items():
                    if isinstance(metric_value, (int, float)):
                        merged.setdefault(metric_key, []).append(float(metric_value))
    if found:
        return {k: sum(vs) / len(vs) for k, vs in merged.items()}
    return {}


def pick_primary_score(task_name: str, metric_block: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    priority_map = {
        "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none", "acc,none"],
        "mmlu": ["acc,none", "acc_norm,none"],
        "hendrycks_math": ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none", "acc,none"],
    }
    for key in priority_map.get(task_name, []):
        value = metric_block.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key, value in metric_block.items():
        if isinstance(value, (int, float)) and "_stderr" not in key:
            return key, float(value)
    return None, None


def get_primary_score(eval_res: Dict[str, Any], task_name: str, default: float = float("-inf")) -> float:
    try:
        value = eval_res["tasks"][task_name]["primary_score"]
        return default if value is None else float(value)
    except Exception:
        return default


def build_early_stop_record(cur_eval: Dict[str, Any], baseline_eval: Dict[str, Any], fit_cfg) -> Dict[str, Any]:
    gsm_cur = get_primary_score(cur_eval, "gsm8k")
    mmlu_cur = get_primary_score(cur_eval, "mmlu")
    gsm_base = get_primary_score(baseline_eval, "gsm8k")
    mmlu_base = get_primary_score(baseline_eval, "mmlu")
    passed_abs_gate = (
        gsm_cur >= float(fit_cfg.train.early_stop_abs_gsm8k)
        and mmlu_cur >= float(fit_cfg.train.early_stop_abs_mmlu)
    )
    return {
        "gsm8k": {"current": gsm_cur, "baseline": gsm_base, "delta_vs_baseline": gsm_cur - gsm_base},
        "mmlu": {"current": mmlu_cur, "baseline": mmlu_base, "delta_vs_baseline": mmlu_cur - mmlu_base},
        "passed_abs_gate": bool(passed_abs_gate),
    }
