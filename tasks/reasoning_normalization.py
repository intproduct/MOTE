from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from .answer_extraction import clean_text, extract_final_answer, split_reasoning_and_final_answer


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

def _coerce_trace_candidates(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    candidates: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            candidates.append(dict(item))
        elif clean_text(item):
            candidates.append({"solution": clean_text(item)})
    return candidates


def _select_openr1_trace(ex: Mapping[str, Any], prefer_short: bool) -> tuple[str, str]:
    candidates = _coerce_trace_candidates(ex.get("generations") or ex.get("solutions") or ex.get("traces"))
    if not candidates:
        direct_solution = clean_text(_get_nested(ex, ["solution", "reasoning", "response", "cot"]))
        if direct_solution:
            return direct_solution, "direct_solution"
        return "", "no_trace"

    def score(candidate: Mapping[str, Any]) -> tuple[int, int]:
        verified = 1 if bool(candidate.get("correctness_math_verify")) else 0
        judged = 1 if bool(candidate.get("correctness_llama")) else 0
        solution_text = clean_text(
            candidate.get("solution")
            or candidate.get("reasoning")
            or candidate.get("response")
            or candidate.get("content")
            or candidate.get("text")
        )
        length_score = approx_token_len(solution_text)
        if prefer_short:
            return (verified * 4 + judged * 2, -length_score)
        return (verified * 4 + judged * 2, length_score)

    usable = [c for c in candidates if clean_text(c.get("solution") or c.get("reasoning") or c.get("response") or c.get("content") or c.get("text"))]
    if not usable:
        return "", "empty_trace_list"
    best = sorted(usable, key=score, reverse=True)[0]
    solution = clean_text(best.get("solution") or best.get("reasoning") or best.get("response") or best.get("content") or best.get("text"))
    if bool(best.get("correctness_math_verify")):
        strategy = "verified_trace"
    elif bool(best.get("correctness_llama")):
        strategy = "llama_judged_trace"
    else:
        strategy = "nonempty_trace"
    return solution, strategy


def extract_question_solution_answer(ex: Mapping[str, Any], dataset_name: str, prefer_short: bool = True) -> Dict[str, Any]:
    dataset_name = str(dataset_name)
    question = clean_text(_get_nested(ex, ["question", "problem", "prompt", "query"]))
    answer = clean_text(_get_nested(ex, ["answer", "final_answer", "boxed", "ground_truth"]))
    solution = clean_text(_get_nested(ex, ["solution", "reasoning", "response", "cot"]))
    strategy = "direct_fields"

    if dataset_name == "openr1_math":
        solution, strategy = _select_openr1_trace(ex, prefer_short=prefer_short)
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


def extract_reasoning_record(ex: Mapping[str, Any], dataset_name: str, prefer_short: bool = True) -> Dict[str, Any]:
    extracted = extract_question_solution_answer(ex, dataset_name, prefer_short=prefer_short)
    return {
        "question": clean_text(extracted.get("question")),
        "solution_text": clean_text(extracted.get("solution")),
        "final_answer": clean_text(extracted.get("answer")),
        "dataset_name": str(dataset_name),
        "metadata": {
            "trace_strategy": extracted.get("trace_strategy", "unknown"),
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
    record = extract_reasoning_record(ex, dataset_name, prefer_short=bool(limits.get("prefer_short_reasoning", True)))
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
        "dataset_name": record.get("dataset_name"),
        "reasoning_record": record,
        "reasoning_supervision_mode": str(supervision_mode),
        "approx_tokens": approx_token_len("\n\n".join([question, solution, answer])),
        "char_length": len("\n\n".join([question, solution, answer])),
    }
