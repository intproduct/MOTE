from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import signal
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch

from ..audit import to_jsonable
from ..config import load_config
from ..eval.config_utils import resolve_task_eval_settings
from ..eval.lm_eval_gsm8k_protocol import LMEvalGSM8KProtocol
from ..rl.boundary import (
    RolloutAuditAccumulator,
    aggregate_spectrum,
    build_prompt_audit_record,
    is_boundary_prompt,
    reward_match_type,
)
from ..rl.data import load_gsm8k_rl_records
from ..rl.device_topology import rollout_actor_specs
from ..rl.generation import tokenize_rollout_prompts
from ..rl.rewards_gsm8k import extract_gsm8k_answer, gsm8k_reward, normalize_number_answer
from ..rl.rollout_backends import HFRolloutBackend, RolloutGenerationConfig
from ..rl.runtime import load_policy_for_rl
from ..rl.vllm_rollout import VLLMRolloutBackend, assert_multi_actor_dispatch
from ..runtime import load_hf_tokenizer
from ..train.rl_controller import build_rl_prompt_text, build_rollout_attention_and_response_mask


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Build GSM8K boundary data and an evaluation-only multi-rollout spectrum audit"
    )
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--model_source",
        choices=("auto", "fitmotn", "hf"),
        default="auto",
        help="static model format; auto detects authoritative checkpoint artifacts",
    )
    parser.add_argument("--rollout_backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--protocol", choices=("rl", "lm_eval"), default="rl")
    parser.add_argument("--split", choices=("train", "test"), default=None)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--decoding", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--dump_prompts_jsonl", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--verified_traces_jsonl", type=str, required=True)
    parser.add_argument("--all_rollouts_jsonl", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--pass_k", type=int, nargs="+", default=None)
    parser.add_argument("--min_correct_rate", type=float, default=0.25)
    parser.add_argument("--max_correct_rate", type=float, default=0.75)
    parser.add_argument("--audit_boundary_min", type=float, default=0.25)
    parser.add_argument("--audit_boundary_max", type=float, default=0.75)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument(
        "--assert_multi_actor_dispatch",
        action="store_true",
        help="fail fast when multi-actor dispatch counts, placement, or concurrency fail audit checks",
    )
    return parser.parse_args(argv)


def _validate_args(args) -> list[int]:
    if int(args.num_rollouts) <= 0:
        raise ValueError(f"--num_rollouts must be > 0, got {args.num_rollouts}")
    if int(args.prompt_batch_size) <= 0:
        raise ValueError("--prompt_batch_size must be > 0")
    if args.max_prompts is not None and int(args.max_prompts) <= 0:
        raise ValueError(f"--max_prompts must be > 0 when set, got {args.max_prompts}")
    if args.num_fewshot is not None and int(args.num_fewshot) < 0:
        raise ValueError("--num_fewshot must be >= 0")
    if args.decoding == "greedy" and int(args.num_rollouts) != 1:
        raise ValueError("--decoding greedy requires --num_rollouts 1")
    if args.protocol == "lm_eval" and args.split not in (None, "test"):
        raise ValueError("--protocol lm_eval uses the formal GSM8K test split")
    if args.protocol == "rl" and args.split not in (None, "train"):
        raise ValueError("--protocol rl preserves the existing GSM8K train split")
    for low, high, label in (
        (args.min_correct_rate, args.max_correct_rate, "selection"),
        (args.audit_boundary_min, args.audit_boundary_max, "audit boundary"),
    ):
        if not 0.0 <= float(low) <= float(high) <= 1.0:
            raise ValueError(f"{label} rates must satisfy 0 <= min <= max <= 1")
    pass_k = list(args.pass_k or sorted({1, min(4, args.num_rollouts), min(8, args.num_rollouts), args.num_rollouts}))
    if any(int(k) <= 0 or int(k) > int(args.num_rollouts) for k in pass_k):
        raise ValueError("every --pass_k must be in [1, --num_rollouts]")
    return [int(k) for k in pass_k]


def _open_jsonl(path: str | Path | None):
    if path is None:
        return None
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("w", encoding="utf-8")


def _write_line(handle, value: Mapping[str, Any]) -> None:
    if handle is not None:
        handle.write(json.dumps(to_jsonable(dict(value)), ensure_ascii=False) + "\n")


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


_HF_WEIGHT_ARTIFACTS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_HF_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
)


def detect_checkpoint_kind(path: str | Path) -> str:
    """Classify a local static model from authoritative checkpoint artifacts."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"static model directory not found: {root}")
    if (root / "fitmotn_state.pt").is_file():
        return "fitmotn"
    if (root / "fitmotn_state.json").is_file():
        raise ValueError(
            "incomplete FitMoTN checkpoint: fitmotn_state.json exists but "
            f"fitmotn_state.pt is missing under {root}"
        )

    has_config = (root / "config.json").is_file()
    has_weights = any((root / name).is_file() for name in _HF_WEIGHT_ARTIFACTS) or any(
        any(root.glob(pattern))
        for pattern in ("model-*.safetensors", "pytorch_model-*.bin")
    )
    has_tokenizer = any(
        (root / name).is_file() for name in _HF_TOKENIZER_ARTIFACTS
    )
    if has_config and has_weights and has_tokenizer:
        return "hf"
    missing = [
        label
        for label, present in (
            ("config.json", has_config),
            ("HF model weights", has_weights),
            ("tokenizer artifacts", has_tokenizer),
        )
        if not present
    ]
    raise ValueError(
        f"unsupported static model directory {root}: missing {', '.join(missing)}"
    )


def _resolve_checkpoint_kind(path: str | Path, requested_kind: str) -> str:
    detected_kind = detect_checkpoint_kind(path)
    requested = str(requested_kind or "auto").strip().lower()
    if requested not in {"auto", "fitmotn", "hf"}:
        raise ValueError(f"unsupported --model_source {requested_kind!r}")
    if requested != "auto" and requested != detected_kind:
        raise ValueError(
            f"--model_source={requested} conflicts with detected checkpoint kind "
            f"{detected_kind!r} for {Path(path).expanduser().resolve()}"
        )
    return detected_kind


def _resolve_static_model_selection(cfg, args) -> Dict[str, Any]:
    cli_path = getattr(args, "resume_from", None)
    config_resume = getattr(cfg.rl, "resume_from", None)
    config_weights = getattr(cfg.rl, "resume_weights_from", None)
    configured_model = getattr(cfg.model, "model_path", None)
    requested_path = cli_path or config_resume or configured_model
    if not requested_path:
        raise ValueError("static evaluation requires --resume_from or model.model_path")
    resolved_path = str(Path(str(requested_path)).expanduser().resolve())
    ignored: Dict[str, str] = {}
    if cli_path is not None:
        for field, value in (
            ("rl.resume_from", config_resume),
            ("rl.resume_weights_from", config_weights),
        ):
            if value and str(Path(str(value)).expanduser().resolve()) != resolved_path:
                ignored[field] = str(Path(str(value)).expanduser().resolve())
    return {
        "requested_model_path": str(requested_path),
        "resolved_model_path": resolved_path,
        "model_path_source": "cli.resume_from" if cli_path is not None else (
            "rl.resume_from" if config_resume else "model.model_path"
        ),
        "ignored_static_resume_fields": ignored,
        "ignored_for_static_evaluation": bool(ignored),
    }


def _token_ids_hash(token_ids) -> str:
    payload = ",".join(str(int(value)) for value in token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _tokenize_static_prompts(
    tokenizer,
    prompts,
    *,
    max_prompt_tokens: int,
    hflm_compatible: bool = False,
):
    prompt_values = list(prompts)
    had_padding_side = hasattr(tokenizer, "padding_side")
    original_padding_side = getattr(tokenizer, "padding_side", None)
    if hflm_compatible and had_padding_side:
        tokenizer.padding_side = "left"
    try:
        raw_enc = tokenize_rollout_prompts(
            tokenizer, prompt_values, max_prompt_tokens=0
        )
    finally:
        if hflm_compatible and had_padding_side:
            tokenizer.padding_side = original_padding_side
    raw_ids = [
        row[mask.to(dtype=torch.bool)].tolist()
        for row, mask in zip(raw_enc["input_ids"], raw_enc["attention_mask"])
    ]
    limit = int(max_prompt_tokens)
    needs_truncation = limit > 0 and any(len(row) > limit for row in raw_ids)
    if needs_truncation and hflm_compatible:
        # HFLM.tok_batch_encode first left-pads the complete batch, then slices
        # every encoded tensor from the left.  Reproduce that exact ordering
        # instead of tokenizer truncation, which can retain/reinsert BOS tokens.
        effective_enc = {
            key: (
                value[:, -limit:]
                if hasattr(value, "shape")
                and len(value.shape) >= 2
                and int(value.shape[1]) > limit
                else value
            )
            for key, value in raw_enc.items()
        }
    elif needs_truncation:
        effective_enc = tokenize_rollout_prompts(
            tokenizer, prompt_values, max_prompt_tokens=limit
        )
    else:
        effective_enc = raw_enc
    effective_ids = [
        row[mask.to(dtype=torch.bool)].tolist()
        for row, mask in zip(
            effective_enc["input_ids"], effective_enc["attention_mask"]
        )
    ]
    prompt_audits = []
    for prompt, original, effective in zip(
        prompt_values, raw_ids, effective_ids
    ):
        removed = max(0, len(original) - len(effective))
        prompt_audits.append(
            {
                "prompt_hash": _prompt_hash(prompt),
                "raw_prompt_token_count": len(original),
                "effective_prompt_token_count": len(effective),
                "prompt_truncated": removed > 0,
                "truncated_prompt_token_count": removed,
                "effective_prompt_token_hash": _token_ids_hash(effective),
            }
        )
    return effective_enc, effective_ids, {
        "max_prompt_tokens": None if limit <= 0 else limit,
        "truncation_side": "left" if limit > 0 else None,
        "prompt_count": len(prompt_values),
        "truncated_prompt_count": sum(
            int(item["prompt_truncated"]) for item in prompt_audits
        ),
        "prompts": prompt_audits,
    }


def _lm_eval_max_prompt_tokens(cfg, generation_config, *, rollout_backend: str) -> int:
    formal_max_length = int(cfg.data.seq_len_run)
    max_gen_toks = int(generation_config.max_new_tokens)
    max_prompt_tokens = formal_max_length - max_gen_toks
    if max_prompt_tokens <= 0:
        raise ValueError(
            "lm_eval protocol requires data.seq_len_run > max_gen_toks: "
            f"seq_len_run={formal_max_length}, max_gen_toks={max_gen_toks}"
        )
    if rollout_backend == "vllm":
        undersized = {
            str(spec["name"]): int(spec["max_model_len"])
            for spec in rollout_actor_specs(cfg.rl)
            if 0 < int(spec["max_model_len"]) < formal_max_length
        }
        if undersized:
            raise ValueError(
                "lm_eval protocol requires every configured vLLM max_model_len to be at least "
                "data.seq_len_run so HFLM-equivalent truncation is possible: "
                f"seq_len_run={formal_max_length}, undersized_actors={undersized}"
            )
    return max_prompt_tokens


def _resolved_split(args) -> str:
    return str(args.split or ("test" if args.protocol == "lm_eval" else "train"))


def _chat_template_callback(tokenizer, runtime: Mapping[str, Any]):
    if not bool(runtime.get("apply_chat_template", False)):
        return None
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        raise ValueError("lm_eval apply_chat_template=true requires tokenizer.apply_chat_template")
    extra = dict(runtime.get("chat_template_args") or {})
    if runtime.get("enable_thinking") is not None:
        extra.setdefault("enable_thinking", bool(runtime["enable_thinking"]))

    def render(messages):
        kwargs = {"tokenize": False, "add_generation_prompt": True, **extra}
        try:
            return apply(messages, **kwargs)
        except TypeError as exc:
            if "enable_thinking" not in kwargs:
                raise
            kwargs.pop("enable_thinking")
            try:
                return apply(messages, **kwargs)
            except TypeError:
                raise exc

    return render


def _prepare_lm_eval_protocol(cfg, args, tokenizer, seed: int) -> LMEvalGSM8KProtocol:
    settings = resolve_task_eval_settings(cfg, "gsm8k", "final", backend="lm_eval")
    runtime = dict(settings.get("runtime") or {})
    return LMEvalGSM8KProtocol.from_fit_config(
        cfg,
        num_fewshot=args.num_fewshot,
        seed=seed,
        max_prompts=args.max_prompts,
        chat_template=_chat_template_callback(tokenizer, runtime),
        tokenizer_name=str(getattr(tokenizer, "name_or_path", "") or ""),
    )


def _uniform_lm_eval_generation_kwargs(records) -> dict[str, Any]:
    values = [
        dict(row["_lm_eval_record"].generation_kwargs)
        for row in records
    ]
    if not values:
        return {}
    first = values[0]
    if any(value != first for value in values[1:]):
        raise RuntimeError("lm_eval GSM8K produced non-uniform generation kwargs across requests")
    return first


def _make_generation_config(cfg, args, seed: int, records) -> tuple[RolloutGenerationConfig, dict[str, Any]]:
    formal = _uniform_lm_eval_generation_kwargs(records) if args.protocol == "lm_eval" else {}
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else formal.get("max_gen_toks", cfg.rl.max_new_tokens)
    )
    decoding = str(args.decoding)
    do_sample = decoding == "sample"
    temperature = float(
        args.temperature
        if args.temperature is not None
        else formal.get("temperature", cfg.rl.temperature)
    )
    top_p = float(
        args.top_p if args.top_p is not None else formal.get("top_p", cfg.rl.top_p)
    )
    stop_sequences = tuple(str(value) for value in (formal.get("until") or []))
    top_k = formal.get("top_k")
    generation_config = RolloutGenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        rollout_micro_batch_size=int(getattr(cfg.rl, "rollout_micro_batch_size", 0)),
        rollout_use_cache=bool(getattr(cfg.rl, "rollout_use_cache", True)),
        rollout_inference_mode=True,
        seed=seed,
        do_sample=do_sample,
        stop_sequences=stop_sequences,
        top_k=None if top_k is None else int(top_k),
    )
    effective = {
        "decoding": decoding,
        "do_sample": do_sample,
        "num_rollouts": int(args.num_rollouts),
        "temperature": 0.0 if not do_sample else temperature,
        "top_p": None if not do_sample else top_p,
        "top_k": None if not do_sample else top_k,
        "max_new_tokens": max_new_tokens,
        "until": list(stop_sequences),
        "task_request_generation_kwargs": formal,
    }
    return generation_config, effective


def _score_rollout(protocol: str, row, completion: str, lm_eval_protocol):
    rl_reward, rl_debug = gsm8k_reward(completion, row["answer"])
    if protocol == "rl":
        return float(rl_reward), rl_debug, reward_match_type(rl_debug, rl_reward), {
            "rl_parser_answer": rl_debug.get("pred_answer"),
            "lm_eval_parser_answer": None,
            "parser_disagreement": False,
            "rl_correct": bool(rl_reward == 1.0),
            "lm_eval_correct": None,
        }
    formal = lm_eval_protocol.score(row["_lm_eval_record"], completion)
    lm_answer = formal.get("extracted_answer")
    rl_answer = extract_gsm8k_answer(completion)
    disagreement = (None if lm_answer is None else normalize_number_answer(lm_answer)) != (
        None if rl_answer is None else normalize_number_answer(rl_answer)
    )
    reward = float(bool(formal["correct"]))
    debug = {
        "pred_answer": lm_answer,
        "gold_answer": formal.get("target"),
        "strict_match": bool(formal["correct"]),
        "fallback_match": False,
        "match_type": "lm_eval_strict_match" if reward == 1.0 else "lm_eval_no_match",
        "lm_eval": formal,
        "rl_reward_debug": rl_debug,
    }
    diagnostic = {
        "rl_parser_answer": rl_answer,
        "lm_eval_parser_answer": lm_answer,
        "parser_disagreement": bool(disagreement),
        "rl_correct": bool(rl_reward == 1.0),
        "lm_eval_correct": bool(formal["correct"]),
    }
    return reward, debug, debug["match_type"], diagnostic


def _emit_vllm_actor_dispatch(
    dispatch: Mapping[str, Any],
    *,
    configured_prompt_batch_size: int,
    configured_num_rollouts: int,
    strict: bool,
) -> None:
    payload = dict(dispatch)
    actual_prompt_count = int(payload.get("prompt_count", payload.get("prompt_batch_size", 0)))
    expected_rows = actual_prompt_count * int(configured_num_rollouts)
    payload["cli_configured_prompt_batch_size"] = int(configured_prompt_batch_size)
    payload["cli_configured_num_rollouts"] = int(configured_num_rollouts)
    payload["cli_expected_prompt_count"] = actual_prompt_count
    payload["cli_expected_completion_count"] = expected_rows
    print(
        "[VLLMActorDispatch] "
        + json.dumps(to_jsonable({"vllm_actor_dispatch": payload}), ensure_ascii=False),
        flush=True,
    )
    if strict:
        assert_multi_actor_dispatch(
            payload,
            expected_row_count=expected_rows,
            expected_prompt_count=actual_prompt_count,
            expected_num_samples=int(configured_num_rollouts),
            require_overlap=True,
        )


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(to_jsonable(dict(value)), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _hf_samples(*, backend, model, tokenizer, prompt, num_rollouts, generation_config, prompt_seed, max_prompt_tokens, hflm_compatible=False):
    del prompt_seed
    prompts = [prompt] * int(num_rollouts)
    enc, _, tokenization_audit = _tokenize_static_prompts(
        tokenizer,
        prompts,
        max_prompt_tokens=max_prompt_tokens,
        hflm_compatible=hflm_compatible,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    batch = backend.generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        update_step=0,
    )
    if batch.metadata.get("vllm_actor_dispatch"):
        print(
            "[VLLMActorDispatch] "
            + json.dumps(
                to_jsonable(
                    {"vllm_actor_dispatch": batch.metadata["vllm_actor_dispatch"]}
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
    _, response_mask = build_rollout_attention_and_response_mask(
        batch.sequences,
        attention_mask,
        response_start=batch.response_start,
        original_seq_lens=batch.original_seq_lens,
        eos_token_id=tokenizer.eos_token_id,
    )
    texts = tokenizer.batch_decode(batch.sequences[:, batch.response_start:], skip_special_tokens=True)
    result = []
    for index, text in enumerate(texts):
        count = int(response_mask[index].sum().item())
        reached_limit = count >= int(generation_config.max_new_tokens)
        result.append({
            "text": text,
            "token_ids": batch.sequences[index, batch.response_start:batch.original_seq_lens[index]].detach().cpu().tolist(),
            "response_token_count": count,
            "finish_reason": "length" if reached_limit else "stop",
            "truncated": reached_limit,
        })
    return result, tokenization_audit


def _vllm_samples(*, backend, tokenizer, prompt, num_rollouts, generation_config, prompt_seed, prompt_index, max_prompt_tokens):
    results, dispatch = _vllm_samples_batch(
        backend=backend,
        tokenizer=tokenizer,
        prompts=[prompt],
        num_rollouts=num_rollouts,
        generation_config=generation_config,
        prompt_seeds=[prompt_seed],
        prompt_indices=[prompt_index],
        max_prompt_tokens=max_prompt_tokens,
    )
    return results[0], dispatch


def _vllm_samples_batch(
    *,
    backend,
    tokenizer,
    prompts,
    num_rollouts,
    generation_config,
    prompt_seeds,
    prompt_indices,
    max_prompt_tokens,
    decoding="sample",
    log_prompt_tokenization=False,
    hflm_compatible=False,
):
    _, effective_prompt_ids, tokenization_audit = _tokenize_static_prompts(
        tokenizer,
        list(prompts),
        max_prompt_tokens=max_prompt_tokens,
        hflm_compatible=hflm_compatible,
    )
    for prompt_index, item in zip(prompt_indices, tokenization_audit["prompts"]):
        item["prompt_index"] = int(prompt_index)
    if log_prompt_tokenization:
        print(
            "[PromptTokenization] "
            + json.dumps(to_jsonable(tokenization_audit), ensure_ascii=False),
            flush=True,
        )
    raw_batch = backend.generate_static_samples_batch(
        prompt_token_ids=effective_prompt_ids,
        num_samples=num_rollouts,
        max_new_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        seeds=list(prompt_seeds),
        prompt_indices=list(prompt_indices),
        eos_token_id=tokenizer.eos_token_id,
        decoding=decoding,
        stop=list(generation_config.stop_sequences),
        top_k=generation_config.top_k,
    )
    results = []
    for raw in raw_batch:
        prompt_results = []
        for item in raw:
            token_ids = list(item["token_ids"])
            finish_reason = item.get("finish_reason")
            truncated = str(finish_reason).lower() in {"length", "max_tokens", "max_token"} or (
                finish_reason is None and len(token_ids) >= int(generation_config.max_new_tokens)
            )
            prompt_results.append({
                **item,
                "response_token_count": len(token_ids),
                "truncated": bool(truncated),
            })
        results.append(prompt_results)
    dispatch = dict(backend.last_actor_dispatch_metadata or {})
    dispatch["prompt_tokenization"] = tokenization_audit
    return results, dispatch


def _copy_checkpoint_callback(source: Path):
    def save(output_dir: Path, update_step: int, checkpoint_name: str, extra: dict[str, Any]) -> Path:
        del update_step, checkpoint_name, extra
        shutil.copytree(source, output_dir, dirs_exist_ok=True)
        return output_dir
    return save


def _immutable_static_checkpoint_callback(*_args, **_kwargs):
    raise RuntimeError("immutable native HF static evaluation cannot export policy checkpoints")


def _prepare_backend(
    cfg,
    args,
    output_root: Path,
    *,
    model_selection: Mapping[str, Any] | None = None,
):
    selection = dict(model_selection or _resolve_static_model_selection(cfg, args))
    resolved_model_path = str(selection["resolved_model_path"])
    checkpoint_kind = _resolve_checkpoint_kind(
        resolved_model_path,
        getattr(args, "model_source", "auto"),
    )
    source_metadata = {
        **selection,
        "checkpoint_kind": checkpoint_kind,
        "model_source": str(getattr(args, "model_source", "auto")),
    }
    print(
        "[StaticModelSource] "
        + json.dumps(
            to_jsonable(
                {
                    "path": resolved_model_path,
                    "detected_kind": checkpoint_kind,
                    "initialization": (
                        "native_hf_direct"
                        if checkpoint_kind == "hf" and args.rollout_backend == "vllm"
                        else (
                            "fitmotn_restore"
                            if checkpoint_kind == "fitmotn"
                            else "hf_policy_restore"
                        )
                    ),
                    "requested_model_path": selection["requested_model_path"],
                    "ignored_static_resume_fields": selection[
                        "ignored_static_resume_fields"
                    ],
                    "ignored_for_static_evaluation": selection[
                        "ignored_for_static_evaluation"
                    ],
                }
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.rollout_backend == "hf":
        model, tokenizer, load_info = load_policy_for_rl(
            cfg, resume_from=resolved_model_path
        )
        model.eval()
        return HFRolloutBackend(), model, tokenizer, {
            **source_metadata,
            "initialization": "hf_checkpoint_restore",
            "checkpoint_metadata_authoritative": bool(load_info.metadata),
            "checkpoint_patch_cfg": load_info.patch_cfg,
            "export_dir": None,
        }

    # Static immutable engines never join online weight-transfer sessions.
    cfg.rl.vllm_sync_strategy = "export_reload"
    cfg.rl.vllm_fallback_to_hf = False
    if checkpoint_kind == "hf":
        tokenizer = load_hf_tokenizer(
            resolved_model_path,
            trust_remote_code=bool(getattr(cfg.model, "trust_remote_code", True)),
            padding_side="left",
        )
        backend = VLLMRolloutBackend(
            fit_cfg=cfg,
            rl_dir=output_root / ".vllm_static_runtime",
            save_policy_checkpoint=_immutable_static_checkpoint_callback,
        )
        try:
            startup = backend.initialize_static_model(
                model_path=resolved_model_path,
                tokenizer_path=resolved_model_path,
                source_kind="hf",
            )
        except BaseException:
            backend.close()
            raise
        return backend, None, tokenizer, {
            **source_metadata,
            "initialization": "native_hf_direct",
            "export_dir": None,
            "policy_version": None,
            "checkpoint_metadata_authoritative": False,
            "checkpoint_patch_cfg": None,
            "vllm_actor_resources": startup.get("vllm_actor_resources"),
            "vllm_rollout_actor_count": startup.get("vllm_rollout_actor_count"),
            "vllm_engine_load_sec": startup.get("engine_load_sec"),
            "vllm_static_model_descriptor": startup.get("policy_descriptor"),
        }

    model, tokenizer, load_info = load_policy_for_rl(
        cfg, resume_from=resolved_model_path
    )
    model.eval()

    # Static evaluation always bootstraps by export/reload exactly once.  It
    # never joins the trainer's native NCCL weight-transfer session.
    cfg.rl.vllm_sync_strategy = "export_reload"
    cfg.rl.vllm_fallback_to_hf = False
    cfg.rl.vllm_export_root = str(output_root / ".vllm_static_exports")
    # The checkpoint metadata wins over a conflicting runtime JSON base path.
    cfg.model.model_path = str(load_info.base_model_path)
    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=output_root / ".vllm_static_runtime",
        save_policy_checkpoint=_copy_checkpoint_callback(
            Path(resolved_model_path).resolve()
        ),
    )
    try:
        sync = backend.sync_policy(model=model, tokenizer=tokenizer, update_step=0, force=True)
    except BaseException:
        backend.close()
        raise
    metadata = {
        **source_metadata,
        "initialization": "static_export_reload_once",
        "export_dir": sync.export_dir,
        "policy_version": sync.policy_version,
        "checkpoint_metadata_authoritative": bool(load_info.metadata),
        "checkpoint_patch_cfg": load_info.patch_cfg,
        "vllm_actor_resources": sync.metadata.get("vllm_actor_resources"),
        "vllm_rollout_actor_count": sync.metadata.get("vllm_rollout_actor_count"),
    }
    return backend, model, tokenizer, metadata


def _iter_prompt_samples(
    *,
    cfg,
    args,
    records,
    backend,
    model,
    tokenizer,
    generation_config,
    seed: int,
    prompt_observer=None,
):
    protocol = str(getattr(args, "protocol", "rl"))
    decoding = str(getattr(args, "decoding", "sample"))
    max_prompt_tokens = (
        _lm_eval_max_prompt_tokens(
            cfg, generation_config, rollout_backend=args.rollout_backend
        )
        if protocol == "lm_eval"
        else int(getattr(cfg.rl, "rollout_max_prompt_tokens", 0))
    )
    if args.rollout_backend == "hf":
        for prompt_index, row in enumerate(records):
            prompt = (
                str(row["_lm_eval_prompt"])
                if protocol == "lm_eval"
                else build_rl_prompt_text(cfg, tokenizer, row["question"])
            )
            prompt_seed = seed + prompt_index
            if prompt_observer is not None:
                prompt_observer(prompt_index, row, prompt)
            samples, tokenization_audit = _hf_samples(
                backend=backend,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                num_rollouts=args.num_rollouts,
                generation_config=generation_config,
                prompt_seed=prompt_seed,
                max_prompt_tokens=max_prompt_tokens,
                hflm_compatible=protocol == "lm_eval",
            )
            row["_prompt_token_audit"] = dict(tokenization_audit["prompts"][0])
            yield prompt_index, row, prompt, prompt_seed, samples
        return

    batch_size = int(args.prompt_batch_size)
    for batch_start in range(0, len(records), batch_size):
        batch_rows = records[batch_start : batch_start + batch_size]
        prompt_indices = list(range(batch_start, batch_start + len(batch_rows)))
        prompts = [
            (
                str(row["_lm_eval_prompt"])
                if protocol == "lm_eval"
                else build_rl_prompt_text(cfg, tokenizer, row["question"])
            )
            for row in batch_rows
        ]
        if prompt_observer is not None:
            for prompt_index, row, prompt in zip(
                prompt_indices, batch_rows, prompts
            ):
                prompt_observer(prompt_index, row, prompt)
        prompt_seeds = [seed + prompt_index for prompt_index in prompt_indices]
        samples_batch, actor_dispatch = _vllm_samples_batch(
            backend=backend,
            tokenizer=tokenizer,
            prompts=prompts,
            num_rollouts=args.num_rollouts,
            generation_config=generation_config,
            prompt_seeds=prompt_seeds,
            prompt_indices=prompt_indices,
            max_prompt_tokens=max_prompt_tokens,
            decoding=decoding,
            log_prompt_tokenization=protocol == "lm_eval",
            hflm_compatible=protocol == "lm_eval",
        )
        if len(samples_batch) != len(batch_rows):
            raise RuntimeError(
                f"static vLLM batch returned {len(samples_batch)} prompt groups; "
                f"expected {len(batch_rows)}"
            )
        _emit_vllm_actor_dispatch(
            actor_dispatch,
            configured_prompt_batch_size=batch_size,
            configured_num_rollouts=int(args.num_rollouts),
            strict=bool(args.assert_multi_actor_dispatch),
        )
        token_audits = list(
            (actor_dispatch.get("prompt_tokenization") or {}).get("prompts") or []
        )
        if token_audits and len(token_audits) != len(batch_rows):
            raise RuntimeError(
                "static vLLM prompt tokenization audit count does not match batch: "
                f"audits={len(token_audits)}, prompts={len(batch_rows)}"
            )
        if token_audits:
            for row, audit in zip(batch_rows, token_audits):
                row["_prompt_token_audit"] = dict(audit)
        for local_index, row in enumerate(batch_rows):
            yield (
                prompt_indices[local_index],
                row,
                prompts[local_index],
                prompt_seeds[local_index],
                samples_batch[local_index],
            )


def _terminate_static_audit(signum, _frame):
    raise SystemExit(128 + int(signum))


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    pass_k = _validate_args(args)
    cfg = load_config(config_json=args.config_json)
    model_selection = _resolve_static_model_selection(cfg, args)
    # Keep legacy output records coherent while making the separately resolved
    # static path authoritative over both RL resume fields for initialization.
    cfg.rl.resume_from = str(model_selection["resolved_model_path"])
    seed = int(args.seed if args.seed is not None else (getattr(cfg.rl, "seed", None) or getattr(cfg.train, "seed", 0)))
    # Preserve the original HF boundary builder's one-time global seeding.
    # vLLM additionally receives the required stable per-prompt request seed.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    records = load_gsm8k_rl_records(cfg) if args.protocol == "rl" else []
    if args.max_prompts is not None and args.protocol == "rl":
        records = records[: int(args.max_prompts)]

    output_root = Path(args.summary_json or args.output_jsonl).expanduser().resolve().parent
    backend = None
    model = None
    handles = []
    prompt_records: list[dict[str, Any]] = []
    rollout_accumulator = RolloutAuditAccumulator()
    verified_count = 0
    selected_count = 0
    init_metadata: dict[str, Any] = {}
    protocol_metadata: dict[str, Any] = {}
    effective_generation: dict[str, Any] = {}
    lm_eval_protocol = None
    parser_disagreement_count = 0
    lm_eval_correct_rl_wrong_count = 0
    rl_correct_lm_eval_wrong_count = 0
    total_scored_rollouts = 0
    prompt_truncated_count = 0
    truncated_prompt_token_count = 0
    maximum_raw_prompt_tokens = 0
    maximum_effective_prompt_tokens = 0
    completed = False
    previous_sigterm_handler = None
    try:
        if args.rollout_backend == "vllm":
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _terminate_static_audit)
        backend, model, tokenizer, init_metadata = _prepare_backend(
            cfg,
            args,
            output_root,
            model_selection=model_selection,
        )
        if args.protocol == "lm_eval":
            lm_eval_protocol = _prepare_lm_eval_protocol(cfg, args, tokenizer, seed)
            records = lm_eval_protocol.boundary_rows()
            protocol_metadata = lm_eval_protocol.metadata()
        else:
            protocol_metadata = {
                "task": "gsm8k",
                "split": "train",
                "num_fewshot": None,
                "prompt_source": "fitmotn.train.rl_controller.build_rl_prompt_text",
                "scoring_source": "fitmotn.rl.rewards_gsm8k.gsm8k_reward",
                "scorer": "gsm8k_reward",
                "apply_chat_template": None,
                "enable_thinking": None,
            }
        generation_config, effective_generation = _make_generation_config(
            cfg, args, seed, records
        )
        if args.protocol == "lm_eval":
            effective_generation["formal_model_max_length"] = int(
                cfg.data.seq_len_run
            )
            effective_generation["max_prompt_tokens"] = _lm_eval_max_prompt_tokens(
                cfg,
                generation_config,
                rollout_backend=args.rollout_backend,
            )
            effective_generation["prompt_truncation"] = "left"
        if args.rollout_backend == "vllm":
            print(
                "[VLLMActorStartup] "
                + json.dumps(
                    to_jsonable(
                        {
                            "actor_count": init_metadata.get("vllm_rollout_actor_count"),
                            "actors": init_metadata.get("vllm_actor_resources"),
                        }
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )
        selected_file = _open_jsonl(args.output_jsonl)
        verified_file = _open_jsonl(args.verified_traces_jsonl)
        all_file = _open_jsonl(args.all_rollouts_jsonl)
        prompts_file = _open_jsonl(args.dump_prompts_jsonl)
        handles = [handle for handle in (selected_file, verified_file, all_file, prompts_file) if handle is not None]

        def observe_prompt(prompt_index, row, prompt):
            prompt_hash = _prompt_hash(prompt)
            _write_line(
                prompts_file,
                {
                    "index": prompt_index,
                    "dataset_index": row.get("dataset_index", row.get("idx", prompt_index)),
                    "question": row["question"],
                    "protocol": args.protocol,
                    "lm_eval_prompt_hash": prompt_hash if args.protocol == "lm_eval" else None,
                    "passk_prompt_hash": prompt_hash,
                    "prompt": prompt,
                },
            )
            if prompts_file is not None:
                prompts_file.flush()

        for prompt_index, row, prompt, prompt_seed, samples in _iter_prompt_samples(
            cfg=cfg,
            args=args,
            records=records,
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            seed=seed,
            prompt_observer=observe_prompt,
        ):
            prompt_hash = _prompt_hash(prompt)
            prompt_token_audit = dict(row.get("_prompt_token_audit") or {})
            prompt_truncated_count += int(
                bool(prompt_token_audit.get("prompt_truncated", False))
            )
            truncated_prompt_token_count += int(
                prompt_token_audit.get("truncated_prompt_token_count", 0)
            )
            maximum_raw_prompt_tokens = max(
                maximum_raw_prompt_tokens,
                int(prompt_token_audit.get("raw_prompt_token_count", 0)),
            )
            maximum_effective_prompt_tokens = max(
                maximum_effective_prompt_tokens,
                int(prompt_token_audit.get("effective_prompt_token_count", 0)),
            )
            rollout_rows = []
            for rollout_index, sample in enumerate(samples):
                reward, debug, match_type, parser_diagnostic = _score_rollout(
                    args.protocol, row, sample["text"], lm_eval_protocol
                )
                total_scored_rollouts += 1
                parser_disagreement_count += int(parser_diagnostic["parser_disagreement"])
                lm_eval_correct_rl_wrong_count += int(
                    parser_diagnostic["lm_eval_correct"] is True
                    and parser_diagnostic["rl_correct"] is False
                )
                rl_correct_lm_eval_wrong_count += int(
                    parser_diagnostic["rl_correct"] is True
                    and parser_diagnostic["lm_eval_correct"] is False
                )
                rollout = {
                    "index": prompt_index,
                    "prompt_index": prompt_index,
                    "dataset_index": row.get("dataset_index", row.get("idx", prompt_index)),
                    "idx": row.get("idx"),
                    "rollout_index": rollout_index,
                    "protocol": args.protocol,
                    "prompt_hash": prompt_hash,
                    "lm_eval_prompt_hash": prompt_hash if args.protocol == "lm_eval" else None,
                    "passk_prompt_hash": prompt_hash,
                    **prompt_token_audit,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "gold": row["answer"],
                    "response": sample["text"],
                    "generated_text": sample["text"],
                    "response_token_count": int(sample["response_token_count"]),
                    "finish_reason": sample.get("finish_reason"),
                    "truncated": bool(sample["truncated"]),
                    "reward": float(reward),
                    "correct": bool(reward == 1.0),
                    "extracted_answer": parser_diagnostic[
                        "lm_eval_parser_answer" if args.protocol == "lm_eval" else "rl_parser_answer"
                    ],
                    "reward_debug": debug,
                    "reward_match_type": match_type,
                    **parser_diagnostic,
                    "seed": prompt_seed,
                    "checkpoint": str(init_metadata["resolved_model_path"]),
                    "sampling": dict(effective_generation),
                }
                rollout_rows.append(rollout)
                rollout_accumulator.add(rollout)
                _write_line(all_file, rollout)
            prompt_record = build_prompt_audit_record(
                row, rollout_rows, pass_k=pass_k,
                boundary_min=args.audit_boundary_min, boundary_max=args.audit_boundary_max,
                mgpo_cfg=cfg.rl,
            )
            prompt_record.update(
                {
                    "dataset_index": row.get("dataset_index", row.get("idx", prompt_index)),
                    "protocol": args.protocol,
                    "prompt_hash": prompt_hash,
                    "lm_eval_prompt_hash": prompt_hash if args.protocol == "lm_eval" else None,
                    "passk_prompt_hash": prompt_hash,
                    **prompt_token_audit,
                    "num_fewshot": protocol_metadata.get("num_fewshot"),
                    "lm_eval_correctness": (
                        [bool(item["lm_eval_correct"]) for item in rollout_rows]
                        if args.protocol == "lm_eval"
                        else None
                    ),
                }
            )
            prompt_records.append(prompt_record)
            if is_boundary_prompt(prompt_record["p_correct"], args.min_correct_rate, args.max_correct_rate):
                _write_line(selected_file, prompt_record)
                selected_count += 1
            for rollout in rollout_rows:
                if rollout["reward"] == 1.0:
                    _write_line(verified_file, {
                        **rollout,
                        "solution": rollout["response"],
                        "final_answer": str(rollout["reward_debug"].get("pred_answer") or ""),
                        "source": "boundary_rollout",
                        "p_correct": prompt_record["p_correct"],
                        "num_rollouts": prompt_record["num_rollouts"],
                        "match_quality_warning": (
                            "fallback-only correctness; inspect before distillation"
                            if rollout["reward_match_type"] == "fallback_last_number_only" else None
                        ),
                    })
                    verified_count += 1
            if args.progress_every > 0 and ((prompt_index + 1) % args.progress_every == 0 or prompt_index + 1 == len(records)):
                partial = aggregate_spectrum(prompt_records, rollout_accumulator, pass_k=pass_k, mgpo_enabled=cfg.rl.mgpo_enabled)
                spectrum = partial["spectrum"]
                print(
                    f"[SpectrumAudit] processed={prompt_index + 1}/{len(records)} "
                    f"sampled_pass1={spectrum['sampled_pass_at_1']:.4f} "
                    f"pass{pass_k[-1]}={spectrum[f'pass_at_{pass_k[-1]}']:.4f} "
                    f"mixed={spectrum['mixed_ratio']:.4f} all_wrong={spectrum['all_wrong_ratio']:.4f} "
                    f"all_correct={spectrum['all_correct_ratio']:.4f} "
                    f"truncated={partial['length_diagnostics']['truncated_rollout_rate']:.4f}",
                    flush=True,
                )
                for handle in handles:
                    handle.flush()
        completed = True
    finally:
        if previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
        try:
            for handle in handles:
                try:
                    handle.flush()
                finally:
                    handle.close()
        finally:
            try:
                if backend is not None:
                    backend.close()
            finally:
                if model is not None:
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    aggregates = aggregate_spectrum(
        prompt_records, rollout_accumulator, pass_k=pass_k, mgpo_enabled=bool(cfg.rl.mgpo_enabled)
    )
    parser_diagnostics = {
        "parser_disagreement_count": parser_disagreement_count,
        "parser_disagreement_rate": (
            float(parser_disagreement_count / total_scored_rollouts)
            if total_scored_rollouts
            else 0.0
        ),
        "lm_eval_correct_rl_wrong_count": lm_eval_correct_rl_wrong_count,
        "rl_correct_lm_eval_wrong_count": rl_correct_lm_eval_wrong_count,
        "num_scored_rollouts": total_scored_rollouts,
    }
    summary = {
        "metadata": {
            "model_or_checkpoint": str(init_metadata["resolved_model_path"]),
            "dataset": "gsm8k",
            "protocol": args.protocol,
            "base_protocol": args.protocol,
            "stochastic_passk": bool(args.protocol == "lm_eval" and args.decoding == "sample"),
            "split": _resolved_split(args),
            "rollout_backend": args.rollout_backend,
            "num_prompts": len(prompt_records),
            "num_rollouts_per_prompt": int(args.num_rollouts),
            "decoding": args.decoding,
            "temperature": effective_generation.get("temperature"),
            "top_p": effective_generation.get("top_p"),
            "max_new_tokens": effective_generation.get("max_new_tokens"),
            "generation": effective_generation,
            "prompt_protocol": protocol_metadata.get("prompt_source"),
            "seed": seed,
            "pass_k": pass_k,
            "selection_correct_rate_range": [args.min_correct_rate, args.max_correct_rate],
            "audit_boundary_range": [args.audit_boundary_min, args.audit_boundary_max],
            "prompt_batch_size": int(args.prompt_batch_size),
            **protocol_metadata,
            **init_metadata,
        },
        **aggregates,
        "parser_diagnostics": parser_diagnostics,
        "prompt_token_diagnostics": {
            "truncation_side": (
                effective_generation.get("prompt_truncation")
                if args.protocol == "lm_eval"
                else (
                    "left"
                    if int(getattr(cfg.rl, "rollout_max_prompt_tokens", 0)) > 0
                    else None
                )
            ),
            "max_prompt_tokens": effective_generation.get("max_prompt_tokens"),
            "truncated_prompt_count": prompt_truncated_count,
            "truncated_prompt_rate": (
                float(prompt_truncated_count / len(prompt_records))
                if prompt_records
                else 0.0
            ),
            "truncated_prompt_token_count": truncated_prompt_token_count,
            "maximum_raw_prompt_tokens": maximum_raw_prompt_tokens,
            "maximum_effective_prompt_tokens": maximum_effective_prompt_tokens,
        },
        "outputs": {
            "selected_prompt_count": selected_count,
            "verified_trace_count": verified_count,
            "output_jsonl": str(Path(args.output_jsonl).expanduser().resolve()),
            "verified_traces_jsonl": str(Path(args.verified_traces_jsonl).expanduser().resolve()),
            "all_rollouts_jsonl": None if args.all_rollouts_jsonl is None else str(Path(args.all_rollouts_jsonl).expanduser().resolve()),
            "dump_prompts_jsonl": None if args.dump_prompts_jsonl is None else str(Path(args.dump_prompts_jsonl).expanduser().resolve()),
        },
        "complete": completed,
    }
    summary_path = args.summary_json or str(Path(args.output_jsonl).with_suffix(".summary.json"))
    _atomic_json(summary_path, summary)
    print(json.dumps({"complete": True, "summary_json": str(Path(summary_path).resolve()), **summary["outputs"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
