from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from .answer_extraction import (
    AnswerValidationError,
    clean_text,
    extract_final_answer,
    split_reasoning_and_final_answer,
    validate_final_answer,
)


class ReasoningNormalizationError(ValueError):
    def __init__(self, reason_code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


def approx_token_len(text: str | None) -> int:
    text = clean_text(text)
    if not text:
        return 0
    words = len(text.split())
    chars = max(1, len(text) // 4)
    return max(words, chars)


def _get_nested(ex: Mapping[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in ex and ex[key] is not None:
            return ex[key]
    return None


def _extract_messages_text(messages: Any, role: str | None = None) -> str:
    if not isinstance(messages, list):
        return clean_text(messages)
    selected: List[str] = []
    for item in messages:
        if not isinstance(item, Mapping):
            continue
        item_role = str(item.get("role") or item.get("from") or "").strip().lower()
        if role is not None and item_role != role:
            continue
        selected.append(clean_text(item.get("content") or item.get("value") or item.get("text")))
    return clean_text(selected)


def _extract_conversation_turns(conversations: Any) -> Dict[str, str]:
    if not isinstance(conversations, list):
        return {"question": "", "answer": clean_text(conversations)}
    user_parts: List[str] = []
    assistant_parts: List[str] = []
    for item in conversations:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or item.get("from") or "").strip().lower()
        text = clean_text(item.get("content") or item.get("value") or item.get("text"))
        if not text:
            continue
        if role in {"user", "human"}:
            user_parts.append(text)
        elif role in {"assistant", "gpt", "model"}:
            assistant_parts.append(text)
    return {"question": clean_text(user_parts), "answer": clean_text(assistant_parts[-1] if assistant_parts else "")}

def _verification_bool(value: Any, *, field: str, index: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ReasoningNormalizationError(
        "invalid_verification_value",
        f"{field}[{index}] is not a strict boolean",
        details={"field": field, "index": index, "value": repr(value)},
    )


def _generation_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return clean_text(
            value.get("solution")
            or value.get("reasoning")
            or value.get("response")
            or value.get("content")
            or value.get("text")
        )
    return clean_text(value)


def select_openr1_trace(ex: Mapping[str, Any], *, tokenizer=None) -> Dict[str, Any]:
    if tokenizer is None:
        raise ReasoningNormalizationError(
            "production_tokenizer_required",
            "OpenR1 trace selection requires the production tokenizer and must run in the offline release builder",
        )
    generations = ex.get("generations")
    math_verify = ex.get("correctness_math_verify")
    llama_verify = ex.get("correctness_llama")
    if not isinstance(generations, list) or not isinstance(math_verify, list):
        raise ReasoningNormalizationError(
            "missing_parallel_verification_arrays",
            "OpenR1 requires generations and correctness_math_verify arrays",
        )
    if len(generations) != len(math_verify) or (llama_verify is not None and (not isinstance(llama_verify, list) or len(llama_verify) != len(generations))):
        raise ReasoningNormalizationError(
            "parallel_array_length_mismatch",
            "OpenR1 parallel arrays have different lengths",
            details={
                "generations": len(generations),
                "correctness_math_verify": len(math_verify),
                "correctness_llama": None if llama_verify is None or not isinstance(llama_verify, list) else len(llama_verify),
            },
        )
    candidates: list[dict[str, Any]] = []
    for index, generation in enumerate(generations):
        solution = _generation_text(generation)
        math_verified = _verification_bool(math_verify[index], field="correctness_math_verify", index=index)
        llama_verified = None
        if llama_verify is not None:
            llama_verified = _verification_bool(llama_verify[index], field="correctness_llama", index=index)
        token_length = len(tokenizer.encode(solution, add_special_tokens=False))
        candidates.append(
            {
                "index": index,
                "solution": solution,
                "math_verified": math_verified,
                "llama_verified": llama_verified,
                "format_complete": bool(extract_final_answer(solution)),
                "token_length": int(token_length),
            }
        )
    verified = [candidate for candidate in candidates if candidate["math_verified"]]
    if not verified:
        raise ReasoningNormalizationError(
            "no_verified_candidate",
            "OpenR1 row has no correctness_math_verify=true candidate",
            details={"available_generation_count": len(candidates)},
        )
    complete = [candidate for candidate in verified if candidate["solution"] and candidate["format_complete"]]
    if not complete:
        raise ReasoningNormalizationError(
            "no_format_complete_verified_candidate",
            "OpenR1 row has verified candidates but none has a parseable final answer",
            details={"available_verified_count": len(verified)},
        )
    selected = min(complete, key=lambda candidate: (candidate["token_length"], candidate["index"]))
    strategy = "math_verified_format_complete_shortest_tokenized"
    return {
        "solution": selected["solution"],
        "selected_generation_index": int(selected["index"]),
        "selected_math_verified": True,
        "selected_llama_verified": selected["llama_verified"],
        "available_generation_count": len(candidates),
        "available_verified_count": len(verified),
        "trace_selection_strategy": strategy,
        "selected_token_length": int(selected["token_length"]),
    }


def extract_question_solution_answer(
    ex: Mapping[str, Any], dataset_name: str, prefer_short: bool = True, *, tokenizer=None
) -> Dict[str, Any]:
    dataset_name = str(dataset_name)
    question = clean_text(_get_nested(ex, ["question", "problem", "prompt", "query"]))
    answer = clean_text(_get_nested(ex, ["answer", "final_answer", "boxed", "ground_truth"]))
    solution = clean_text(_get_nested(ex, ["solution", "reasoning", "response", "cot"]))
    strategy = "direct_fields"
    selection_metadata: Dict[str, Any] = {}

    if dataset_name == "openr1_math":
        selected = select_openr1_trace(ex, tokenizer=tokenizer)
        solution = selected["solution"]
        strategy = selected["trace_selection_strategy"]
        selection_metadata = {key: value for key, value in selected.items() if key != "solution"}
        question = question or clean_text(_extract_messages_text(ex.get("messages"), role="user"))
        if not answer:
            answer = clean_text(_get_nested(ex, ["answer", "final_answer", "ground_truth", "expected_answer"]))
        if not answer and solution:
            _, answer = split_reasoning_and_final_answer(solution)
    elif dataset_name == "numinamath_cot":
        if not question:
            question = clean_text(_extract_messages_text(ex.get("messages"), role="user"))
        if not solution:
            solution = clean_text(_extract_messages_text(ex.get("messages"), role="assistant"))
        if not answer and solution:
            _, answer = split_reasoning_and_final_answer(solution)
        strategy = "problem_solution_or_messages"
    elif dataset_name == "openthoughts_math":
        conv = _extract_conversation_turns(ex.get("conversations"))
        question = question or conv["question"] or clean_text(_extract_messages_text(ex.get("messages"), role="user"))
        if not solution:
            solution = conv["answer"] or clean_text(_extract_messages_text(ex.get("messages"), role="assistant"))
        if not answer and solution:
            _, answer = split_reasoning_and_final_answer(solution)
        strategy = "conversation_or_messages"
    elif dataset_name == "bespoke_stratos":
        conv = _extract_conversation_turns(ex.get("conversations"))
        question = question or conv["question"] or clean_text(_extract_messages_text(ex.get("messages"), role="user"))
        assistant_text = conv["answer"] or clean_text(_extract_messages_text(ex.get("messages"), role="assistant"))
        if assistant_text:
            solution, parsed_answer = split_reasoning_and_final_answer(assistant_text)
            answer = answer or parsed_answer
        strategy = "conversation_trace"
    else:
        if not question:
            question = clean_text(_extract_messages_text(ex.get("messages"), role="user"))
        if not solution:
            solution = clean_text(_extract_messages_text(ex.get("messages"), role="assistant"))
        if not solution and answer:
            solution, parsed_answer = split_reasoning_and_final_answer(answer)
            answer = parsed_answer or answer
        if not answer and solution:
            _, answer = split_reasoning_and_final_answer(solution)

    if not question and ex.get("conversations") is not None:
        question = question or _extract_conversation_turns(ex.get("conversations"))["question"]
    if not solution and ex.get("conversations") is not None:
        solution = solution or _extract_conversation_turns(ex.get("conversations"))["answer"]
    if not answer and solution:
        _, answer = split_reasoning_and_final_answer(solution)
    return {
        "question": clean_text(question),
        "solution": clean_text(solution),
        "answer": clean_text(answer) or extract_final_answer(solution),
        "trace_strategy": strategy,
        "selection_metadata": selection_metadata,
    }


def is_reasoning_sample_too_long(question: str, solution: str, answer: str, limits: Mapping[str, Any] | None) -> bool:
    limits = limits or {}
    full_text = "\n\n".join([part for part in [clean_text(question), clean_text(solution), clean_text(answer)] if part])
    max_chars = limits.get("max_chars")
    if max_chars is not None and len(full_text) > int(max_chars):
        return True
    max_tokens = limits.get("max_approx_tokens")
    if max_tokens is not None and approx_token_len(full_text) > int(max_tokens):
        return True
    return False


def extract_reasoning_record(
    ex: Mapping[str, Any], dataset_name: str, prefer_short: bool = True, *, tokenizer=None
) -> Dict[str, Any]:
    extracted = extract_question_solution_answer(ex, dataset_name, prefer_short=prefer_short, tokenizer=tokenizer)
    return {
        "question": clean_text(extracted.get("question")),
        "solution_text": clean_text(extracted.get("solution")),
        "final_answer": clean_text(extracted.get("answer")),
        "dataset_name": str(dataset_name),
        "metadata": {
            "trace_strategy": extracted.get("trace_strategy", "unknown"),
            **dict(extracted.get("selection_metadata") or {}),
        },
        "trace_strategy": extracted.get("trace_strategy", "unknown"),
    }


def format_reasoning_prompt_target(record: Mapping[str, Any], supervision_mode: str) -> tuple[str, str]:
    question = clean_text(record.get("question"))
    solution = clean_text(record.get("solution_text"))
    answer = clean_text(record.get("final_answer"))
    mode = str(supervision_mode or "answer_only").strip().lower()
    if mode == "full_trace":
        prompt = f"Question:\n{question}"
        target = f"Solution:\n{solution}\n\nFinal Answer:\n{answer}"
        return prompt, target
    if mode == "answer_only":
        prompt = f"Question:\n{question}\n\nSolution:\n{solution}\n\nFinal Answer:"
        return prompt, answer
    raise ValueError(f"Unknown reasoning supervision mode: {supervision_mode}")


def normalize_reasoning_sample(
    ex: Mapping[str, Any],
    dataset_name: str,
    limits: Mapping[str, Any] | None,
    supervision_mode: str = "answer_only",
) -> Dict[str, Any]:
    limits = dict(limits or {})
    correct_flag = ex.get("correct")
    if dataset_name == "openthoughts_math" and (correct_flag is False or str(correct_flag).strip().lower() == "false"):
        return {"ok": False, "reason": "incorrect_trace", "trace_strategy": "correctness_filter"}
    try:
        record = extract_reasoning_record(ex, dataset_name, prefer_short=bool(limits.get("prefer_short_reasoning", True)))
    except ReasoningNormalizationError as exc:
        return {
            "ok": False,
            "reason": exc.reason_code,
            "trace_strategy": "strict_openr1_rejection",
            "details": exc.details,
        }
    question = record["question"]
    solution = record["solution_text"]
    answer = record["final_answer"]
    if not question or not solution or not answer:
        return {
            "ok": False,
            "reason": "missing_fields",
            "trace_strategy": record.get("trace_strategy", "unknown"),
            "question": question,
            "solution": solution,
            "answer": answer,
        }
    task_type = clean_text(ex.get("task_type") or ex.get("type") or "numeric").lower()
    if "proof" in task_type:
        task_type = "proof"
    try:
        validated_answer = validate_final_answer(solution, answer, task_type=task_type)
    except AnswerValidationError as exc:
        return {
            "ok": False,
            "reason": exc.reason_code,
            "trace_strategy": record.get("trace_strategy", "unknown"),
            "question": question,
            "solution": solution,
            "answer": answer,
        }
    answer = validated_answer["final_answer"]
    record["final_answer"] = answer
    record["answer_type"] = validated_answer["answer_type"]
    too_long = is_reasoning_sample_too_long(question, solution, answer, limits=limits)
    if too_long and bool(limits.get("skip_overlong_reasoning_samples", True)):
        return {
            "ok": False,
            "reason": "overlong",
            "trace_strategy": record.get("trace_strategy", "unknown"),
            "question": question,
            "solution": solution,
            "answer": answer,
        }
    prompt, target = format_reasoning_prompt_target(record, supervision_mode)
    return {
        "ok": True,
        "reason": "ok",
        "prompt": prompt,
        "target": target,
        "eval_type": "numeric",
        "trace_strategy": record.get("trace_strategy", "unknown"),
        "question": question,
        "solution": solution,
        "answer": answer,
        "answer_type": validated_answer["answer_type"],
        "dataset_name": record.get("dataset_name"),
        "reasoning_record": record,
        "reasoning_supervision_mode": str(supervision_mode),
        "approx_tokens": approx_token_len("\n\n".join([question, solution, answer])),
        "char_length": len("\n\n".join([question, solution, answer])),
    }
