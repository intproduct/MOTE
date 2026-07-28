from __future__ import annotations

import re
from typing import Any, Mapping

from .text_normalization import normalize_text


DEFAULT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
}


def clean_text(value: Any) -> str:
    return normalize_text(value)


def _as_text(value: Any) -> str:
    return clean_text(value)


def _first_present(ex: Mapping[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key and key in ex and ex[key] is not None:
            return ex[key]
    return None


def normalize_text_example(ex: Mapping[str, Any], *, text_field: str | None = None) -> str | None:
    keys = [text_field] if text_field else []
    keys += ["text", "content", "body", "document", "code", "completion"]
    text = _as_text(_first_present(ex, [key for key in keys if key]))
    return text or None


def _normalize_role(value: Any, role_map: Mapping[str, str] | None = None) -> str:
    merged = dict(DEFAULT_ROLE_MAP)
    merged.update({str(k).strip().lower(): str(v).strip().lower() for k, v in dict(role_map or {}).items()})
    role = str(value or "").strip().lower()
    return merged.get(role, role)


def _message_from_mapping(
    msg: Mapping[str, Any],
    *,
    role_key: str,
    content_key: str,
    role_map: Mapping[str, str] | None,
) -> dict[str, str] | None:
    role = _normalize_role(msg.get(role_key, msg.get("role", msg.get("from"))), role_map)
    content = _as_text(msg.get(content_key, msg.get("content", msg.get("value"))))
    if role not in {"user", "assistant", "system"} or not content:
        return None
    return {"role": role, "content": content}


def _finalize_messages(messages: list[dict[str, str]], *, skip_if_no_assistant: bool, skip_empty: bool) -> list[dict[str, str]] | None:
    if skip_empty:
        messages = [msg for msg in messages if _as_text(msg.get("content"))]
    if not messages:
        return None
    if skip_if_no_assistant and not any(msg.get("role") == "assistant" for msg in messages):
        return None
    return messages


def normalize_chat_messages(
    ex: Mapping[str, Any],
    *,
    messages_field: str | None = None,
    role_key: str = "role",
    content_key: str = "content",
    role_map: Mapping[str, str] | None = None,
    system_field: str | None = None,
    prompt_field: str | None = None,
    response_field: str | None = None,
    skip_if_no_assistant: bool = True,
    skip_empty: bool = True,
) -> list[dict[str, str]] | None:
    raw_messages = ex.get(messages_field) if messages_field else None
    if raw_messages is None:
        raw_messages = _first_present(ex, ["messages", "conversations", "conversation", "dialogue", "dialog"])
    messages: list[dict[str, str]] = []
    if isinstance(raw_messages, list):
        for raw in raw_messages:
            if isinstance(raw, Mapping):
                msg = _message_from_mapping(raw, role_key=role_key, content_key=content_key, role_map=role_map)
                if msg is not None:
                    messages.append(msg)
    if not messages:
        sys_text = _as_text(ex.get(system_field)) if system_field else ""
        prompt = _as_text(ex.get(prompt_field)) if prompt_field else ""
        response = _as_text(ex.get(response_field)) if response_field else ""
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if response:
            messages.append({"role": "assistant", "content": response})
    return _finalize_messages(messages, skip_if_no_assistant=skip_if_no_assistant, skip_empty=skip_empty)


def normalize_prompt_response(
    ex: Mapping[str, Any],
    *,
    instruction_field: str | None = None,
    input_field: str | None = None,
    prompt_field: str | None = None,
    output_field: str | None = None,
    response_field: str | None = None,
    system_field: str | None = None,
    skip_empty: bool = True,
) -> list[dict[str, str]] | None:
    instruction = _as_text(_first_present(ex, [instruction_field or "", "instruction"]))
    input_text = _as_text(_first_present(ex, [input_field or "", "input"]))
    prompt = _as_text(_first_present(ex, [prompt_field or "", "prompt", "question"]))
    response = _as_text(_first_present(ex, [response_field or "", output_field or "", "response", "output", "answer"]))
    user = "\n\n".join(part for part in [instruction, input_text] if part) or prompt
    messages: list[dict[str, str]] = []
    sys_text = _as_text(ex.get(system_field)) if system_field else ""
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    if user:
        messages.append({"role": "user", "content": user})
    if response:
        messages.append({"role": "assistant", "content": response})
    return _finalize_messages(messages, skip_if_no_assistant=True, skip_empty=skip_empty)


def _extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text or "", flags=re.IGNORECASE | re.DOTALL)
    return _as_text(match.group(1)) if match else ""


def normalize_reasoning_qa(
    ex: Mapping[str, Any],
    *,
    question_field: str | None = None,
    solution_field: str | None = None,
    answer_field: str | None = None,
    answer_extraction: str | None = None,
    system_field: str | None = None,
    skip_empty: bool = True,
) -> list[dict[str, str]] | None:
    question = _as_text(_first_present(ex, [question_field or "", "question", "problem", "prompt", "instruction"]))
    solution = _as_text(_first_present(ex, [solution_field or "", "solution", "reasoning", "rationale", "response", "output"]))
    answer = _as_text(_first_present(ex, [answer_field or "", "final_answer", "answer"]))
    if not answer and str(answer_extraction or "").strip().lower() == "xml_answer":
        answer = _extract_xml_answer(solution)
    assistant = solution
    if answer and answer not in assistant:
        assistant = "\n\n".join(part for part in [solution, f"Final Answer:\n{answer}"] if part)
    messages: list[dict[str, str]] = []
    sys_text = _as_text(ex.get(system_field)) if system_field else ""
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    if question:
        messages.append({"role": "user", "content": question})
    if assistant:
        messages.append({"role": "assistant", "content": assistant})
    return _finalize_messages(messages, skip_if_no_assistant=True, skip_empty=skip_empty)


def normalize_chat_like_example(ex: Mapping[str, Any], task) -> list[dict[str, str]] | None:
    dataset_format = str(getattr(task, "dataset_format", "chat_messages") or "chat_messages")
    if dataset_format == "prompt_response":
        return normalize_prompt_response(
            ex,
            instruction_field=getattr(task, "instruction_field", None),
            input_field=getattr(task, "input_field", None),
            prompt_field=getattr(task, "prompt_field", None),
            output_field=getattr(task, "output_field", None),
            response_field=getattr(task, "response_field", None),
            system_field=getattr(task, "system_field", None),
            skip_empty=bool(getattr(task, "skip_empty", True)),
        )
    if dataset_format == "reasoning_qa":
        return normalize_reasoning_qa(
            ex,
            question_field=getattr(task, "question_field", None),
            solution_field=getattr(task, "solution_field", None),
            answer_field=getattr(task, "answer_field", None),
            answer_extraction=getattr(task, "answer_extraction", None),
            system_field=getattr(task, "system_field", None),
            skip_empty=bool(getattr(task, "skip_empty", True)),
        )
    return normalize_chat_messages(
        ex,
        messages_field=getattr(task, "messages_field", None),
        role_key=str(getattr(task, "role_key", "role") or "role"),
        content_key=str(getattr(task, "content_key", "content") or "content"),
        role_map=getattr(task, "role_map", None),
        system_field=getattr(task, "system_field", None),
        prompt_field=getattr(task, "prompt_field", None),
        response_field=getattr(task, "response_field", None),
        skip_if_no_assistant=bool(getattr(task, "skip_if_no_assistant", True)),
        skip_empty=bool(getattr(task, "skip_empty", True)),
    )
