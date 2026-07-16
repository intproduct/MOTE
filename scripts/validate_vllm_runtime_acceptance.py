#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fitmotn.diagnostics.vllm_export_preflight import validate_vllm_export_preflight


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _generate_token_prompts(llm, prompt_token_ids: Sequence[Sequence[int]], sampling_params):
    try:
        return llm.generate(
            prompts=[{"prompt_token_ids": list(ids)} for ids in prompt_token_ids],
            sampling_params=sampling_params,
        )
    except TypeError:
        return llm.generate(prompt_token_ids=[list(ids) for ids in prompt_token_ids], sampling_params=sampling_params)


def _output_tokens(outputs) -> list[list[int]]:
    result: list[list[int]] = []
    for output in outputs:
        if not output.outputs:
            raise RuntimeError("vLLM returned a request with no completion")
        token_ids = getattr(output.outputs[0], "token_ids", None)
        if token_ids is None:
            raise RuntimeError("vLLM completion has no token_ids")
        result.append([int(token) for token in token_ids])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-engine FitMoTN vLLM 0.19 GPU acceptance")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--skip-hf-parity",
        action="store_true",
        help="Skip the sequential HF-vLLM first-token parity gate (not recommended for acceptance).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    repo_code = Path(__file__).resolve().parents[1] / "export" / "remote_code" / "modeling_fitmotn.py"
    report: Dict[str, Any] = {
        "ok": False,
        "model": str(model_path),
        "stages": {},
        "limitations": [
            "This gate validates one vLLM engine, not online policy export/reload latency.",
            "TP2+ acceptance validates runtime correctness but not ideal ADTN parameter sharding efficiency.",
        ],
    }
    try:
        preflight = validate_vllm_export_preflight(
            model_path,
            expected_remote_code=repo_code,
            require_vllm=True,
        )
        report["stages"]["static_preflight"] = preflight
        if not preflight["ok"]:
            raise RuntimeError("static preflight failed: " + "; ".join(preflight["errors"]))

        import torch
        import transformers
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if int(args.tensor_parallel_size) < 1:
            raise ValueError("--tensor-parallel-size must be >= 1")
        if torch.cuda.device_count() < int(args.tensor_parallel_size):
            raise RuntimeError(
                f"visible CUDA device count={torch.cuda.device_count()} is below "
                f"tensor_parallel_size={args.tensor_parallel_size}"
            )
        report["environment"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_device_count": int(torch.cuda.device_count()),
            "tensor_parallel_size": int(args.tensor_parallel_size),
        }
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        prompts = [
            "Question:\nWhat is 1 + 1?\n\nAnswer:\n",
            "Question:\nA box has 3 red balls and 4 blue balls. How many balls are there?\n\nAnswer:\n",
        ]
        token_prompts = [list(tokenizer(prompt, add_special_tokens=True)["input_ids"]) for prompt in prompts]

        hf_first_token = None
        if not args.skip_hf_parity:
            from fitmotn.runtime import load_causal_lm_and_tokenizer

            hf_started = time.perf_counter()
            hf_model, _, _ = load_causal_lm_and_tokenizer(
                model_path,
                device=torch.device("cuda:0"),
                trust_remote_code=True,
                torch_dtype=str(args.dtype),
                use_cache=True,
            )
            input_ids = torch.tensor([token_prompts[0]], dtype=torch.long, device="cuda:0")
            with torch.inference_mode():
                hf_logits = hf_model(input_ids=input_ids, use_cache=True).logits[:, -1, :]
            hf_first_token = int(torch.argmax(hf_logits, dim=-1).item())
            report["stages"]["hf_first_token_reference"] = {
                "ok": True,
                "token_id": hf_first_token,
                "seconds": time.perf_counter() - hf_started,
            }
            del hf_logits, input_ids, hf_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        load_started = time.perf_counter()
        llm = LLM(
            model=str(model_path),
            tokenizer=str(model_path),
            trust_remote_code=True,
            dtype=str(args.dtype),
            tensor_parallel_size=int(args.tensor_parallel_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            max_num_seqs=int(args.max_num_seqs),
            enforce_eager=True,
            disable_log_stats=True,
            model_impl="transformers",
            seed=int(args.seed),
        )
        report["stages"]["engine_load"] = {
            "ok": True,
            "seconds": time.perf_counter() - load_started,
            "tensor_parallel_size": int(args.tensor_parallel_size),
        }

        greedy_one = SamplingParams(temperature=0.0, max_tokens=1)
        started = time.perf_counter()
        first = _output_tokens(_generate_token_prompts(llm, [token_prompts[0]], greedy_one))
        second = _output_tokens(_generate_token_prompts(llm, [token_prompts[0]], greedy_one))
        if first != second or len(first) != 1 or len(first[0]) != 1:
            raise RuntimeError(f"one-token greedy determinism failed: first={first}, second={second}")
        if hf_first_token is not None and first[0][0] != hf_first_token:
            raise RuntimeError(
                "HF-vLLM first-token parity failed: "
                f"hf={hf_first_token}, vllm={first[0][0]}"
            )
        report["stages"]["greedy_one_token"] = {
            "ok": True,
            "token_ids": first,
            "matches_hf": hf_first_token is None or first[0][0] == hf_first_token,
            "seconds": time.perf_counter() - started,
        }

        greedy_batch = SamplingParams(temperature=0.0, max_tokens=16)
        started = time.perf_counter()
        batch_tokens = _output_tokens(_generate_token_prompts(llm, token_prompts, greedy_batch))
        if len(batch_tokens) != len(token_prompts) or any(not tokens for tokens in batch_tokens):
            raise RuntimeError(f"variable-length greedy batch failed: {batch_tokens}")
        report["stages"]["variable_length_batch"] = {
            "ok": True,
            "lengths": [len(tokens) for tokens in batch_tokens],
            "seconds": time.perf_counter() - started,
        }

        sampled = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=16)
        group_prompts = [token_prompts[0] for _ in range(4)]
        started = time.perf_counter()
        sampled_tokens = _output_tokens(_generate_token_prompts(llm, group_prompts, sampled))
        if len(sampled_tokens) != 4 or any(not tokens for tokens in sampled_tokens):
            raise RuntimeError(f"sampled rollout group failed: {sampled_tokens}")
        report["stages"]["sampled_group"] = {
            "ok": True,
            "lengths": [len(tokens) for tokens in sampled_tokens],
            "unique_sequences": len({tuple(tokens) for tokens in sampled_tokens}),
            "seconds": time.perf_counter() - started,
        }
        report["ok"] = True
        _write(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        _write(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
