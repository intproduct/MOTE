from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from ..checkpointing import load_fitmotn_metadata
from ..tasks.reasoning_specs import GSM8KTask, MMLUTask
from ..data.mixed_iterable import load_dataset_any
from ..runtime import normalize_hf_config


def _normalize_text(s: str) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").strip().replace("\r\n", "\n"))


def _extract_final_answer(s: str) -> str:
    s = _normalize_text(s)
    matches = re.findall(r"####\s*([^\n]+)", s)
    if matches:
        return _normalize_text(matches[-1])
    lines = [line.strip() for line in s.split("\n") if line.strip()]
    return lines[-1] if lines else s


def _numeric_match(pred: str, ref: str) -> float:
    def pick_num(x: str):
        found = re.findall(r"[-+]?\d+(?:\.\d+)?", _normalize_text(x).replace(",", ""))
        return found[-1] if found else None
    pred_num, ref_num = pick_num(pred), pick_num(ref)
    if pred_num is None or ref_num is None:
        return 1.0 if _normalize_text(pred) == _normalize_text(ref) else 0.0
    return 1.0 if pred_num == ref_num else 0.0


def _mcq_choice_match(pred: str, ref: str) -> float:
    def pick_choice(x: str) -> str:
        upper = _normalize_text(x).upper()
        found = re.findall(r"\b([A-E])\b", upper)
        if found:
            return found[0]
        found = re.findall(r"([A-E])", upper)
        return found[0] if found else ""
    return 1.0 if pick_choice(pred) == pick_choice(ref) else 0.0


def _build_eval_tasks(fit_cfg) -> List[Any]:
    tasks: List[Any] = []
    for name in fit_cfg.eval.final_tasks:
        if name == "gsm8k":
            tasks.append(GSM8KTask(name="gsm8k_eval", path=fit_cfg.data.gsm8k_cache_path, split="test", kind="auto", hf_name=fit_cfg.data.gsm8k_hf_name, hf_config=normalize_hf_config(fit_cfg.data.gsm8k_hf_config)))
        elif name == "mmlu":
            tasks.append(MMLUTask(name="mmlu_eval", path=fit_cfg.data.mmlu_cache_path, split=fit_cfg.data.mmlu_split, kind="auto", hf_name=fit_cfg.data.mmlu_hf_name, hf_config=normalize_hf_config(fit_cfg.data.mmlu_hf_config)))
    return tasks


def inspect_vllm_compatibility(model_path: str) -> Dict[str, Any]:
    ckpt_dir = Path(model_path).resolve()
    report: Dict[str, Any] = {
        "model_path": str(ckpt_dir),
        "has_fitmotn_state": bool((ckpt_dir / "fitmotn_state.pt").exists()),
        "supports_vllm_eval": True,
        "reason": None,
        "checkpoint_format": None,
    }
    if report["has_fitmotn_state"]:
        metadata = load_fitmotn_metadata(ckpt_dir)
        report["checkpoint_format"] = metadata.get("checkpoint_format")
        if metadata.get("layers_to_patch"):
            report["supports_vllm_eval"] = False
            report["reason"] = "patched_fitmotn_checkpoint_not_supported_by_vllm_runner"
    return report


def evaluate_with_vllm(model_path: str, fit_cfg, limit_per_task: int = 32) -> Dict[str, Any]:
    compatibility = inspect_vllm_compatibility(model_path)
    if not compatibility["supports_vllm_eval"]:
        raise NotImplementedError(
            "vLLM evaluation for patched FitMoTN checkpoints is intentionally not implemented in MVP. "
            f"reason={compatibility['reason']}"
        )
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as exc:
        raise ImportError("vLLM is required for eval_vllm.py") from exc

    llm = LLM(model=str(model_path), trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)
    results: Dict[str, Any] = {"tasks": {}, "summary": {}, "capability": compatibility}
    for task in _build_eval_tasks(fit_cfg):
        raw = load_dataset_any(task.kind, task.path, task.split, hf_name=task.hf_name, hf_config=task.hf_config)
        n = 0
        correct = 0.0
        prompts = []
        refs = []
        eval_types = []
        for ex in raw:
            prompt, answer, eval_type = task.map_example(ex)
            prompts.append(prompt)
            refs.append(answer)
            eval_types.append(eval_type)
            if len(prompts) >= limit_per_task:
                break
        outputs = llm.generate(prompts, sampling)
        for output, ref, eval_type in zip(outputs, refs, eval_types):
            text = output.outputs[0].text if output.outputs else ""
            if eval_type == "mcq":
                correct += _mcq_choice_match(text, ref)
            else:
                correct += _numeric_match(_extract_final_answer(text), _extract_final_answer(ref))
            n += 1
        acc = correct / max(1, n)
        results["tasks"][task.name] = {"acc": acc, "n": n}
        results["summary"][task.name] = {"primary_metric": "acc", "primary_score": acc}
    return results
