from __future__ import annotations

import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ..export.format import EXPORT_MANIFEST_FILENAME, classify_model_path
from ..export.manifest import read_raw_checkpoint_json
from ..tasks.reasoning_specs import GSM8KTask, MMLUTask
from ..tasks.answer_extraction import extract_final_answer
from ..data.mixed_iterable import load_dataset_any
from ..runtime import normalize_hf_config


def _normalize_text(s: str) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").strip().replace("\r\n", "\n"))


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
    path_kind = classify_model_path(ckpt_dir)
    report: Dict[str, Any] = {
        "model_path": str(ckpt_dir),
        "path_kind": path_kind,
        "has_fitmotn_state": path_kind == "raw_fitmotn_checkpoint",
        "is_exported_fitmotn_dir": path_kind == "exported_fitmotn_dir",
        "supports_vllm_eval": True,
        "reason": None,
        "checkpoint_format": None,
        "export_stage": None,
        "vllm_ready": None,
    }
    if path_kind == "raw_fitmotn_checkpoint":
        metadata, _ = read_raw_checkpoint_json(ckpt_dir)
        report["checkpoint_format"] = metadata.get("checkpoint_format")
        report["supports_vllm_eval"] = False
        report["reason"] = "raw_fitmotn_checkpoint_requires_export_hf"
    elif path_kind == "exported_fitmotn_dir":
        try:
            with (ckpt_dir / EXPORT_MANIFEST_FILENAME).open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
        report["export_stage"] = manifest.get("export_stage")
        report["vllm_ready"] = manifest.get("vllm_ready")
        report["vllm_backend"] = manifest.get("vllm_backend")
        report["vllm_model_impl"] = manifest.get("vllm_model_impl")
        if manifest.get("vllm_ready") is not True:
            report["supports_vllm_eval"] = False
            if manifest.get("export_stage") == "hf_roundtrip":
                report["reason"] = "hf_roundtrip_export_not_vllm_ready"
            else:
                report["reason"] = "metadata_only_export_not_vllm_ready"
    return report


def evaluate_with_vllm(
    model_path: str,
    fit_cfg,
    limit_per_task: int = 32,
    *,
    model_impl: str | None = None,
    dtype: str | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    enforce_eager: bool | None = None,
    seed: int | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    compatibility = inspect_vllm_compatibility(model_path)
    if not compatibility["supports_vllm_eval"]:
        if compatibility.get("reason") == "raw_fitmotn_checkpoint_requires_export_hf":
            raise NotImplementedError(
                "Raw FitMoTN checkpoint is not directly supported by vLLM. Run "
                "`python -m fitmotn.cli.export_hf --checkpoint_dir ... --output_dir ...` first."
            )
        if compatibility.get("reason") == "metadata_only_export_not_vllm_ready":
            raise NotImplementedError(
                "This exported FitMoTN directory is metadata-only. Stage 4B/4C must complete HF roundtrip "
                "and vLLM support before inference."
            )
        if compatibility.get("reason") == "hf_roundtrip_export_not_vllm_ready":
            raise NotImplementedError(
                "This exported FitMoTN directory is HF roundtrip-ready, but vLLM support is not implemented yet. "
                "Stage 4C will add vLLM offline runner support."
            )
        raise NotImplementedError(
            "vLLM evaluation for patched FitMoTN checkpoints is intentionally not implemented in MVP. "
            f"reason={compatibility['reason']}"
        )
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as exc:
        raise ImportError("vLLM is required for eval_vllm.py") from exc

    is_fitmotn_export = compatibility.get("is_exported_fitmotn_dir") is True
    llm_kwargs: Dict[str, Any] = {"model": str(model_path), "trust_remote_code": True}
    if is_fitmotn_export:
        llm_kwargs["tokenizer"] = str(model_path)
        llm_kwargs["model_impl"] = model_impl or "transformers"
        llm_kwargs["enforce_eager"] = True if enforce_eager is None else bool(enforce_eager)
    else:
        if model_impl is not None:
            llm_kwargs["model_impl"] = model_impl
        if enforce_eager is not None:
            llm_kwargs["enforce_eager"] = bool(enforce_eager)
    for key, value in [
        ("dtype", dtype),
        ("tensor_parallel_size", tensor_parallel_size),
        ("gpu_memory_utilization", gpu_memory_utilization),
        ("max_model_len", max_model_len),
        ("seed", seed),
    ]:
        if value is not None:
            llm_kwargs[key] = value
    load_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - load_start
    sampling = SamplingParams(temperature=float(temperature), top_p=float(top_p), max_tokens=int(max_tokens))
    results: Dict[str, Any] = {
        "tasks": {},
        "summary": {},
        "capability": compatibility,
        "vllm_options": llm_kwargs,
        "timing": {"load_seconds": load_seconds},
        "throughput": {},
    }
    total_prompts = 0
    total_generated_tokens = 0
    generate_seconds = 0.0
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
        gen_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        generate_seconds += time.perf_counter() - gen_start
        total_prompts += len(prompts)
        for output, ref, eval_type in zip(outputs, refs, eval_types):
            text = output.outputs[0].text if output.outputs else ""
            if output.outputs:
                token_ids = getattr(output.outputs[0], "token_ids", None)
                if token_ids is not None:
                    total_generated_tokens += len(token_ids)
            if eval_type == "mcq":
                correct += _mcq_choice_match(text, ref)
            else:
                correct += _numeric_match(extract_final_answer(text), extract_final_answer(ref))
            n += 1
        acc = correct / max(1, n)
        results["tasks"][task.name] = {"acc": acc, "n": n}
        results["summary"][task.name] = {"primary_metric": "acc", "primary_score": acc}
    results["timing"]["generate_seconds"] = generate_seconds
    if generate_seconds > 0:
        results["throughput"]["prompts_per_second"] = float(total_prompts) / generate_seconds
        if total_generated_tokens:
            results["throughput"]["generated_tokens_per_second"] = float(total_generated_tokens) / generate_seconds
    return results
