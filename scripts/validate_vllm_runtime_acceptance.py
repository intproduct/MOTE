#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
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


def _progress(message: str) -> None:
    print(f"[vllm-acceptance] {message}", flush=True)


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


def _observed_prompt_token_ids(outputs) -> list[list[int] | None]:
    observed: list[list[int] | None] = []
    for output in outputs:
        token_ids = getattr(output, "prompt_token_ids", None)
        observed.append(None if token_ids is None else [int(token) for token in token_ids])
    return observed


def _token_description(tokenizer, token_id: int) -> Dict[str, Any]:
    token_id = int(token_id)
    try:
        token_piece = tokenizer.convert_ids_to_tokens(token_id)
    except Exception:
        token_piece = None
    try:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        decoded = None
    return {"token_id": token_id, "token_piece": token_piece, "decoded": decoded}


def _vllm_first_position_logprobs(outputs, tokenizer) -> list[Dict[str, Any]]:
    if not outputs or not getattr(outputs[0], "outputs", None):
        return []
    positions = getattr(outputs[0].outputs[0], "logprobs", None)
    if not positions:
        return []
    mapping = positions[0] or {}
    rows: list[Dict[str, Any]] = []
    for raw_token_id, value in mapping.items():
        try:
            token_id = int(getattr(value, "token_id", raw_token_id))
        except (TypeError, ValueError):
            continue
        raw_logprob = getattr(value, "logprob", value)
        try:
            logprob = float(raw_logprob)
        except (TypeError, ValueError):
            continue
        rank = getattr(value, "rank", None)
        row = _token_description(tokenizer, token_id)
        row.update(
            {
                "logprob": logprob,
                "rank": None if rank is None else int(rank),
                "vllm_decoded_token": getattr(value, "decoded_token", None),
            }
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (row["rank"] is None, row["rank"] or 10**9, -row["logprob"]))


def _hf_token_rank(logprobs, token_id: int) -> int:
    value = logprobs[int(token_id)]
    return int((logprobs > value).sum().item()) + 1


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
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    hf_logprobs_cpu = None
    try:
        _progress("running static export/weight preflight")
        preflight = validate_vllm_export_preflight(
            model_path,
            expected_remote_code=repo_code,
            require_vllm=True,
        )
        report["stages"]["static_preflight"] = preflight
        _write(output_path, report)
        if not preflight["ok"]:
            raise RuntimeError("static preflight failed: " + "; ".join(preflight["errors"]))
        _progress("static preflight passed")

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
        report["stages"]["prompt_contract"] = {
            "prompt_text": prompts[0],
            "expected_prompt_token_ids": token_prompts[0],
            "expected_prompt_token_count": len(token_prompts[0]),
            "decoded_prompt": tokenizer.decode(token_prompts[0], skip_special_tokens=False),
        }
        _write(output_path, report)

        hf_first_token = None
        if not args.skip_hf_parity:
            from fitmotn.runtime import load_causal_lm_and_tokenizer

            _progress("loading HF reference and computing first-token top-20")
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
            hf_logprobs_cpu = torch.log_softmax(hf_logits[0].float(), dim=-1).cpu()
            hf_first_token = int(torch.argmax(hf_logprobs_cpu).item())
            top_values, top_indices = torch.topk(hf_logprobs_cpu, k=min(20, int(hf_logprobs_cpu.numel())))
            hf_top = []
            for rank, (token_id, logprob) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
                row = _token_description(tokenizer, int(token_id))
                row.update({"rank": rank, "logprob": float(logprob)})
                hf_top.append(row)
            report["stages"]["hf_first_token_reference"] = {
                "ok": True,
                "token_id": hf_first_token,
                "token": _token_description(tokenizer, hf_first_token),
                "top_logprobs": hf_top,
                "seconds": time.perf_counter() - hf_started,
            }
            _write(output_path, report)
            _progress(f"HF reference ready: first_token={hf_first_token!r}")
            del hf_logits, input_ids, hf_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        _progress(
            f"loading vLLM engine with tensor_parallel_size={int(args.tensor_parallel_size)}"
        )
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
        _write(output_path, report)
        _progress("vLLM engine loaded; running one-token greedy diagnostics")

        greedy_one = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        started = time.perf_counter()
        first_outputs = _generate_token_prompts(llm, [token_prompts[0]], greedy_one)
        second_outputs = _generate_token_prompts(llm, [token_prompts[0]], greedy_one)
        first = _output_tokens(first_outputs)
        second = _output_tokens(second_outputs)
        observed_prompts = _observed_prompt_token_ids(first_outputs)
        vllm_top = _vllm_first_position_logprobs(first_outputs, tokenizer)
        deterministic = first == second and len(first) == 1 and len(first[0]) == 1
        observed_prompt = observed_prompts[0] if observed_prompts else None
        prompt_matches = observed_prompt == token_prompts[0]
        vllm_first_token = first[0][0] if len(first) == 1 and len(first[0]) == 1 else None
        parity: Dict[str, Any] = {
            "exact_match": hf_first_token is None or vllm_first_token == hf_first_token,
            "hf_token": None if hf_first_token is None else _token_description(tokenizer, hf_first_token),
            "vllm_token": (
                None if vllm_first_token is None else _token_description(tokenizer, vllm_first_token)
            ),
            "vllm_top_logprobs": vllm_top,
        }
        if hf_logprobs_cpu is not None and vllm_first_token is not None:
            hf_best_logprob = float(hf_logprobs_cpu[hf_first_token].item())
            vllm_token_hf_logprob = float(hf_logprobs_cpu[vllm_first_token].item())
            parity.update(
                {
                    "vllm_token_rank_under_hf": _hf_token_rank(hf_logprobs_cpu, vllm_first_token),
                    "vllm_token_logprob_under_hf": vllm_token_hf_logprob,
                    "hf_best_minus_vllm_token_logprob": hf_best_logprob - vllm_token_hf_logprob,
                }
            )
            vllm_by_token = {int(row["token_id"]): row for row in vllm_top}
            parity["hf_token_under_vllm"] = vllm_by_token.get(hf_first_token)
            parity["hf_token_returned_in_vllm_top_logprobs"] = hf_first_token in vllm_by_token
        report["stages"]["greedy_one_token"] = {
            "ok": bool(
                deterministic
                and prompt_matches
                and (hf_first_token is None or vllm_first_token == hf_first_token)
            ),
            "first_token_ids": first,
            "second_token_ids": second,
            "deterministic": deterministic,
            "expected_prompt_token_ids": token_prompts[0],
            "observed_prompt_token_ids": observed_prompt,
            "prompt_token_ids_available": observed_prompt is not None,
            "prompt_token_ids_match": prompt_matches,
            "parity": parity,
            "seconds": time.perf_counter() - started,
        }
        _write(output_path, report)
        if not deterministic:
            raise RuntimeError(f"one-token greedy determinism failed: first={first}, second={second}")
        if not prompt_matches:
            raise RuntimeError(
                "vLLM prompt token contract failed: "
                f"expected={token_prompts[0]}, observed={observed_prompt}"
            )
        if hf_first_token is not None and vllm_first_token != hf_first_token:
            raise RuntimeError(
                "HF-vLLM first-token parity failed: "
                f"hf={hf_first_token}, vllm={vllm_first_token}; "
                "see stages.greedy_one_token.parity for cross-backend ranks/logprobs"
            )
        _progress("one-token parity passed; running variable-length batch")

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
        _write(output_path, report)
        _progress("variable-length batch passed; running sampled rollout group")

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
        _write(output_path, report)
        report["ok"] = True
        _write(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        _write(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        _progress(f"acceptance failed; evidence was written to {output_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
