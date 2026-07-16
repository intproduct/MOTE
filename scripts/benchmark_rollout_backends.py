#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Solve the problem step by step. Put the final numeric answer after ####.\n"
    "Answer:\n"
)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_env(item) for key, item in value.items()}
    return value


def _read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("benchmark config must be a JSON object")
    return dict(_expand_env(payload))


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _load_questions(cfg: Dict[str, Any]) -> List[str]:
    data = dict(cfg.get("data") or {})
    prompts_jsonl = data.get("prompts_jsonl")
    rows: Iterable[Dict[str, Any]]
    if prompts_jsonl:
        parsed: List[Dict[str, Any]] = []
        with Path(prompts_jsonl).expanduser().open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        parsed.append(value)
        rows = parsed
    else:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover - dependency error is runtime-specific
            raise ImportError("datasets is required when data.prompts_jsonl is not set") from exc
        rows = load_dataset(
            str(data.get("dataset_name", "openai/gsm8k")),
            data.get("dataset_config", "main"),
            split=str(data.get("split", "train")),
        )

    questions: List[str] = []
    for row in rows:
        value = row.get("question", row.get("prompt"))
        if value is not None and str(value).strip():
            questions.append(str(value))
        if len(questions) >= int(cfg["benchmark"]["num_prompts"]):
            break
    if not questions:
        raise ValueError("no usable question/prompt rows were found")
    return questions


def _prepare_prompt_manifest(cfg: Dict[str, Any], output_dir: Path) -> Path:
    from transformers import AutoTokenizer

    model_path = str(cfg["model"]["path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
        fix_mistral_regex=True,
    )
    max_prompt_tokens = int(cfg["benchmark"].get("max_prompt_tokens", 0))
    group_size = int(cfg["benchmark"]["group_size"])
    template = str(cfg["data"].get("prompt_template") or DEFAULT_PROMPT_TEMPLATE)
    token_rows: List[Dict[str, Any]] = []
    for prompt_index, question in enumerate(_load_questions(cfg)):
        prompt = template.format(question=question) if "{question}" in template else f"{template}{question}"
        ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        if max_prompt_tokens > 0:
            ids = ids[-max_prompt_tokens:]
        if not ids:
            raise ValueError(f"tokenized prompt {prompt_index} is empty")
        for sample_index in range(group_size):
            token_rows.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "prompt_token_ids": [int(token) for token in ids],
                }
            )
    manifest = {
        "model_path": model_path,
        "num_unique_prompts": len(token_rows) // group_size,
        "group_size": group_size,
        "num_rollout_rows": len(token_rows),
        "prompt_token_count": sum(len(row["prompt_token_ids"]) for row in token_rows),
        "rows": token_rows,
    }
    path = output_dir / "fixed_prompt_tokens.json"
    _write_json(path, manifest)
    return path


def _cuda_sync(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _response_length(tokens: Sequence[int], eos_token_id: int | None, pad_token_id: int | None) -> int:
    count = 0
    for token in tokens:
        token = int(token)
        if pad_token_id is not None and token == int(pad_token_id) and eos_token_id != pad_token_id:
            break
        count += 1
        if eos_token_id is not None and token == int(eos_token_id):
            break
    return count


def _metric_summary(seconds: Sequence[float], tokens: Sequence[int], row_count: int) -> Dict[str, Any]:
    runs = []
    for index, (elapsed, generated) in enumerate(zip(seconds, tokens), start=1):
        runs.append(
            {
                "repeat": index,
                "generate_seconds": float(elapsed),
                "generated_tokens": int(generated),
                "prompts_per_second": float(row_count) / float(elapsed),
                "generated_tokens_per_second": float(generated) / float(elapsed),
                "average_generated_tokens": float(generated) / float(row_count),
            }
        )
    return {
        "runs": runs,
        "median_generate_seconds": statistics.median(seconds),
        "median_prompts_per_second": statistics.median(row["prompts_per_second"] for row in runs),
        "median_generated_tokens_per_second": statistics.median(
            row["generated_tokens_per_second"] for row in runs
        ),
        "median_average_generated_tokens": statistics.median(row["average_generated_tokens"] for row in runs),
    }


def _hf_worker(cfg: Dict[str, Any], prompt_manifest: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    from fitmotn.runtime import load_causal_lm_and_tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("HF benchmark requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(int(cfg["benchmark"]["seed"]))
    torch.cuda.manual_seed_all(int(cfg["benchmark"]["seed"]))
    load_started = time.perf_counter()
    model, tokenizer, resolved_dtype = load_causal_lm_and_tokenizer(
        cfg["model"]["path"],
        device=device,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
        torch_dtype=str(cfg["model"].get("dtype", "bfloat16")),
        use_cache=True,
    )
    model.eval()
    _cuda_sync(torch)
    load_seconds = time.perf_counter() - load_started

    rows = [list(row["prompt_token_ids"]) for row in prompt_manifest["rows"]]
    batch_size = int(cfg["hf"]["batch_size"])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    eos_id = tokenizer.eos_token_id
    generation = cfg["generation"]

    def run(selected_rows: Sequence[Sequence[int]]) -> int:
        total_generated = 0
        for start in range(0, len(selected_rows), batch_size):
            chunk = selected_rows[start : start + batch_size]
            width = max(len(ids) for ids in chunk)
            input_ids = torch.full((len(chunk), width), int(pad_id), dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
            for row_index, ids in enumerate(chunk):
                input_ids[row_index, width - len(ids) :] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[row_index, width - len(ids) :] = 1
            kwargs: Dict[str, Any] = {
                "max_new_tokens": int(generation["max_new_tokens"]),
                "do_sample": float(generation["temperature"]) > 0,
                "pad_token_id": int(pad_id),
                "eos_token_id": eos_id,
                "use_cache": True,
            }
            if kwargs["do_sample"]:
                kwargs["temperature"] = float(generation["temperature"])
                kwargs["top_p"] = float(generation["top_p"])
            with torch.inference_mode():
                output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            for output_row in output[:, width:].detach().cpu().tolist():
                total_generated += _response_length(output_row, eos_id, int(pad_id))
        _cuda_sync(torch)
        return total_generated

    warmup_rows = rows[: min(len(rows), int(cfg["benchmark"]["warmup_rows"]))]
    run(warmup_rows)
    torch.cuda.reset_peak_memory_stats(device)
    seconds: List[float] = []
    tokens: List[int] = []
    for repeat in range(int(cfg["benchmark"]["repeats"])):
        torch.manual_seed(int(cfg["benchmark"]["seed"]) + repeat)
        torch.cuda.manual_seed_all(int(cfg["benchmark"]["seed"]) + repeat)
        _cuda_sync(torch)
        started = time.perf_counter()
        tokens.append(run(rows))
        seconds.append(time.perf_counter() - started)
    result = {
        "backend": "hf",
        "load_seconds": load_seconds,
        "dtype": str(resolved_dtype).replace("torch.", ""),
        "num_rollout_rows": len(rows),
        "hf_batch_size": batch_size,
        "generation_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "generation_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        **_metric_summary(seconds, tokens, len(rows)),
    }
    return result


def _vllm_generate(llm, prompts, sampling_params):
    try:
        return llm.generate(prompts=[{"prompt_token_ids": ids} for ids in prompts], sampling_params=sampling_params)
    except TypeError:
        return llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)


def _vllm_worker(cfg: Dict[str, Any], prompt_manifest: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM benchmark requires CUDA")
    vcfg = cfg["vllm"]
    kwargs: Dict[str, Any] = {
        "model": str(cfg["model"]["path"]),
        "tokenizer": str(cfg["model"]["path"]),
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", True)),
        "dtype": str(cfg["model"].get("dtype", "bfloat16")),
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(vcfg["gpu_memory_utilization"]),
        "max_model_len": int(vcfg["max_model_len"]),
        "max_num_seqs": int(vcfg["max_num_seqs"]),
        "enforce_eager": bool(vcfg["enforce_eager"]),
        "disable_log_stats": bool(vcfg.get("disable_log_stats", True)),
        "seed": int(cfg["benchmark"]["seed"]),
    }
    if vcfg.get("model_impl"):
        kwargs["model_impl"] = str(vcfg["model_impl"])
    load_started = time.perf_counter()
    llm = LLM(**kwargs)
    load_seconds = time.perf_counter() - load_started
    generation = cfg["generation"]
    sampling = SamplingParams(
        max_tokens=int(generation["max_new_tokens"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
    )
    rows = [list(row["prompt_token_ids"]) for row in prompt_manifest["rows"]]

    def run(selected_rows: Sequence[Sequence[int]]) -> int:
        outputs = _vllm_generate(llm, list(selected_rows), sampling)
        total = 0
        for output in outputs:
            if output.outputs:
                total += len(list(output.outputs[0].token_ids))
        return total

    run(rows[: min(len(rows), int(cfg["benchmark"]["warmup_rows"]))])
    seconds: List[float] = []
    tokens: List[int] = []
    for _ in range(int(cfg["benchmark"]["repeats"])):
        started = time.perf_counter()
        tokens.append(run(rows))
        seconds.append(time.perf_counter() - started)
    return {
        "backend": "vllm",
        "load_seconds": load_seconds,
        "dtype": str(cfg["model"].get("dtype", "bfloat16")),
        "num_rollout_rows": len(rows),
        "vllm_options": kwargs,
        **_metric_summary(seconds, tokens, len(rows)),
    }


def _query_gpu_memory() -> List[float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _run_worker(backend: str, config_path: Path, prompt_path: Path, output_dir: Path) -> Dict[str, Any]:
    result_path = output_dir / f"{backend}_result.json"
    log_path = output_dir / f"{backend}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-backend",
        backend,
        "--config",
        str(config_path),
        "--prompt-manifest",
        str(prompt_path),
        "--worker-output",
        str(result_path),
    ]
    peaks: List[float] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            memory = _query_gpu_memory()
            if memory:
                peaks.append(max(memory))
            time.sleep(0.2)
    if process.returncode != 0:
        raise RuntimeError(f"{backend} worker failed with exit code {process.returncode}; see {log_path}")
    result = _read_json(result_path)
    result["process_lifetime_peak_gpu_memory_mib"] = max(peaks) if peaks else None
    _write_json(result_path, result)
    return result


def build_comparison(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {"available_backends": sorted(results)}
    if "hf" in results and "vllm" in results:
        hf = results["hf"]
        vllm = results["vllm"]
        comparison.update(
            {
                "vllm_vs_hf_prompts_per_second_speedup": (
                    float(vllm["median_prompts_per_second"]) / float(hf["median_prompts_per_second"])
                ),
                "vllm_vs_hf_generated_tokens_per_second_speedup": (
                    float(vllm["median_generated_tokens_per_second"])
                    / float(hf["median_generated_tokens_per_second"])
                ),
                "vllm_minus_hf_load_seconds": float(vllm["load_seconds"]) - float(hf["load_seconds"]),
            }
        )
    return comparison


def _validate_config(cfg: Dict[str, Any]) -> None:
    for section in ("model", "data", "benchmark", "generation", "hf", "vllm"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"missing config section: {section}")
    if not cfg["model"].get("path"):
        raise ValueError("model.path is required")
    for key in ("num_prompts", "group_size", "warmup_rows", "repeats"):
        if int(cfg["benchmark"].get(key, 0)) <= 0:
            raise ValueError(f"benchmark.{key} must be > 0")
    if int(cfg["generation"].get("max_new_tokens", 0)) <= 0:
        raise ValueError("generation.max_new_tokens must be > 0")
    if int(cfg["hf"].get("batch_size", 0)) <= 0:
        raise ValueError("hf.batch_size must be > 0")
    if int(cfg["vllm"].get("max_num_seqs", 0)) <= 0:
        raise ValueError("vllm.max_num_seqs must be > 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair single-GPU HF/vLLM rollout throughput benchmark")
    parser.add_argument("--config", required=True, help="Benchmark JSON configuration")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir")
    parser.add_argument("--backends", nargs="+", choices=("hf", "vllm"), default=None)
    parser.add_argument("--worker-backend", choices=("hf", "vllm"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--prompt-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = _read_json(config_path)
    _validate_config(cfg)
    if args.worker_backend:
        if not args.prompt_manifest or not args.worker_output:
            raise ValueError("worker mode requires --prompt-manifest and --worker-output")
        prompt_manifest = _read_json(args.prompt_manifest)
        if args.worker_backend == "hf":
            result = _hf_worker(cfg, prompt_manifest)
        else:
            result = _vllm_worker(cfg, prompt_manifest)
        _write_json(args.worker_output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    output_dir = Path(args.output_dir or cfg.get("output_dir") or "outputs/rollout_backend_benchmark")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backends = list(args.backends or cfg.get("backends") or ["hf", "vllm"])
    if "vllm" in backends:
        from fitmotn.diagnostics.vllm_export_preflight import validate_vllm_export_preflight

        repository_remote_code = Path(__file__).resolve().parents[1] / "export" / "remote_code" / "modeling_fitmotn.py"
        preflight = validate_vllm_export_preflight(
            cfg["model"]["path"],
            expected_remote_code=repository_remote_code,
            require_vllm=True,
        )
        _write_json(output_dir / "vllm_export_preflight.json", preflight)
        if not preflight["ok"]:
            raise RuntimeError(
                "vLLM export preflight failed before GPU engine startup: "
                + "; ".join(str(error) for error in preflight["errors"])
            )
    effective_config_path = output_dir / "effective_config.json"
    cfg["output_dir"] = str(output_dir)
    _write_json(effective_config_path, cfg)
    prompt_path = _prepare_prompt_manifest(cfg, output_dir)
    results: Dict[str, Dict[str, Any]] = {}
    for backend in backends:
        if backend not in {"hf", "vllm"}:
            raise ValueError(f"unsupported backend: {backend}")
        print(f"[benchmark] starting {backend} worker", flush=True)
        results[backend] = _run_worker(backend, effective_config_path, prompt_path, output_dir)
        print(f"[benchmark] completed {backend}", flush=True)
    final = {
        "config": cfg,
        "prompt_manifest": str(prompt_path),
        "results": results,
        "comparison": build_comparison(results),
        "notes": [
            "Both backends consume the exact same pre-tokenized prompt rows in isolated processes.",
            "Generation medians exclude model loading and warmup.",
            "Different samplers may stop at EOS at different lengths; compare token/s as the normalized throughput metric.",
            "This benchmark excludes FitMoTN policy export, vLLM engine rebuild, reward, logprob, and optimizer time.",
        ],
    }
    final_path = output_dir / "comparison.json"
    _write_json(final_path, final)
    print(json.dumps(final["comparison"], ensure_ascii=False, indent=2))
    print(f"[benchmark] evidence: {final_path}")


if __name__ == "__main__":
    main()
