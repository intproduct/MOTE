#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


DEFAULT_PROMPTS = [
    "Compute 17 + 25. Give only the final answer.",
    "A box has 9 red balls and 4 blue balls. How many balls are there?",
    "Explain briefly why the sky appears blue.",
]


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4 CUDA validation requires an NVIDIA CUDA environment; this is not a CPU/Mac test")


def _load_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        prompts.append(str(value.get("prompt") if isinstance(value, dict) else value))
    if not prompts:
        raise ValueError("prompt JSONL did not contain any prompts")
    return prompts


def _effective_prompt_ids(encoded) -> list[list[int]]:
    rows = []
    for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        rows.append(ids[mask.to(dtype=bool)].detach().cpu().tolist())
    return rows


def _generate_hf(model, encoded, *, max_new_tokens: int) -> list[list[int]]:
    import torch

    input_len = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
        )
    return [row[input_len:].detach().cpu().tolist() for row in output]


def validate(args) -> dict[str, Any]:
    _require_cuda()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = _load_prompts(args.prompts_jsonl)
    export_dir = Path(args.export_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    encoded_cpu = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_token_ids = _effective_prompt_ids(encoded_cpu)
    report: dict[str, Any] = {
        "export_dir": str(export_dir),
        "raw_checkpoint": args.raw_checkpoint,
        "prompt_count": len(prompts),
        "versions": {
            "torch": torch.__version__,
        },
        "thresholds": {
            "max_logit_abs_error": float(args.max_logit_abs_error),
            "min_greedy_token_match": float(args.min_greedy_token_match),
        },
    }

    export_start = time.perf_counter()
    exported_model = AutoModelForCausalLM.from_pretrained(
        export_dir,
        trust_remote_code=True,
        torch_dtype=args.torch_dtype,
    ).to(args.device)
    exported_model.eval()
    report["hf_export_load_sec"] = time.perf_counter() - export_start
    encoded = {key: value.to(args.device) for key, value in encoded_cpu.items()}
    with torch.inference_mode():
        export_logits = exported_model(**encoded).logits[:, -1, :].float().cpu()
    export_tokens = _generate_hf(exported_model, encoded, max_new_tokens=args.max_new_tokens)
    del exported_model
    del encoded
    torch.cuda.empty_cache()

    if args.raw_checkpoint:
        from fitmotn.eval.restore import restore_fitmotn_model

        raw_model, raw_tokenizer, _ = restore_fitmotn_model(args.raw_checkpoint, device=args.device)
        raw_model.eval()
        raw_encoded = raw_tokenizer(prompts, return_tensors="pt", padding=True)
        raw_encoded = {key: value.to(args.device) for key, value in raw_encoded.items()}
        with torch.inference_mode():
            raw_logits = raw_model(**raw_encoded).logits[:, -1, :].float().cpu()
        raw_tokens = _generate_hf(raw_model, raw_encoded, max_new_tokens=args.max_new_tokens)
        logit_error = float((raw_logits - export_logits).abs().max().item())
        report["raw_vs_export"] = {
            "max_logit_abs_error": logit_error,
            "greedy_exact_match": [a == b for a, b in zip(raw_tokens, export_tokens)],
            "ok": logit_error <= float(args.max_logit_abs_error) and raw_tokens == export_tokens,
        }
        del raw_model
        torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    import vllm

    report["versions"]["vllm"] = getattr(vllm, "__version__", None)
    vllm_start = time.perf_counter()
    llm = LLM(
        model=str(export_dir),
        tokenizer=str(export_dir),
        trust_remote_code=True,
        model_impl="transformers",
        enforce_eager=bool(args.enforce_eager),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )
    report["vllm_load_sec"] = time.perf_counter() - vllm_start
    sampling = SamplingParams(max_tokens=int(args.max_new_tokens), temperature=0.0, top_p=1.0)
    gen_start = time.perf_counter()
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": ids} for ids in prompt_token_ids],
        sampling_params=sampling,
    )
    report["vllm_generate_sec"] = time.perf_counter() - gen_start
    vllm_tokens = [list(output.outputs[0].token_ids) if output.outputs else [] for output in outputs]
    total = sum(max(len(a), len(b)) for a, b in zip(export_tokens, vllm_tokens))
    matched = sum(sum(x == y for x, y in zip(a, b)) for a, b in zip(export_tokens, vllm_tokens))
    token_match = float(matched / max(1, total))
    report["export_hf_vs_vllm"] = {
        "hf_tokens": export_tokens,
        "vllm_tokens": vllm_tokens,
        "exact_match": [a == b for a, b in zip(export_tokens, vllm_tokens)],
        "token_match_ratio": token_match,
        "ok": token_match >= float(args.min_greedy_token_match),
    }
    report["ok"] = bool(report["export_hf_vs_vllm"]["ok"]) and bool(
        report.get("raw_vs_export", {"ok": True})["ok"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA-only Stage 4 HF export and vLLM parity validator")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--raw-checkpoint")
    parser.add_argument("--prompts-jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-logit-abs-error", type=float, default=0.02)
    parser.add_argument("--min-greedy-token-match", type=float, default=0.99)
    args = parser.parse_args()
    report = validate(args)
    target = Path(args.output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
