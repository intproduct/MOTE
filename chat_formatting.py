from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch as tc


ASSISTANT_MASK_KEYS = (
    "assistant_masks",
    "assistant_mask",
    "assistant_tokens_mask",
    "assistant_token_mask",
    "assistant_masks",
)


def tokenizer_supports_chat_template(tokenizer) -> bool:
    return callable(getattr(tokenizer, "apply_chat_template", None))


def require_chat_template(tokenizer) -> None:
    if not tokenizer_supports_chat_template(tokenizer):
        raise ValueError("chat format requires tokenizer.apply_chat_template; refusing to fall back to raw formatting")


def apply_standard_chat_template(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    **extra_kwargs,
):
    require_chat_template(tokenizer)
    kwargs = dict(extra_kwargs)
    kwargs.update(
        {
            "tokenize": bool(tokenize),
            "add_generation_prompt": bool(add_generation_prompt),
            "enable_thinking": bool(enable_thinking),
        }
    )
    try:
        return tokenizer.apply_chat_template(list(messages), **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in kwargs:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(list(messages), **retry_kwargs)
        except TypeError:
            raise exc


def build_reasoning_messages(question, target: Optional[str] = None, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    question_content = f"Question:\n{'' if question is None else str(question).strip()}"
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": question_content})
    if target is not None:
        messages.append({"role": "assistant", "content": str(target)})
    return messages


def build_chat_prompt_text_for_generation(
    tokenizer,
    user_content: str,
    system_prompt: Optional[str] = None,
    enable_thinking: bool = False,
    **kwargs,
) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": str(user_content)})
    return apply_standard_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        **kwargs,
    )


def _as_list(value: Any) -> List[int]:
    if isinstance(value, tc.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in list(value or [])]


def _extract_assistant_mask(tokenized: Mapping[str, Any]) -> Optional[List[int]]:
    for key in ASSISTANT_MASK_KEYS:
        if key in tokenized and tokenized[key] is not None:
            return _as_list(tokenized[key])
    return None


def _prefix_len_or_raise(full_ids: List[int], prompt_ids: List[int]) -> int:
    if len(prompt_ids) <= len(full_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    max_probe = min(len(prompt_ids), len(full_ids))
    for trim in range(1, min(8, max_probe) + 1):
        candidate = prompt_ids[:-trim]
        if candidate and full_ids[: len(candidate)] == candidate:
            return len(candidate)
    raise ValueError(
        "chat template fallback could not align prompt prefix with full conversation; "
        "use a tokenizer that supports return_assistant_tokens_mask"
    )


def make_standard_chat_supervised_example(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    max_len: int,
    add_eos: bool = True,
    *,
    enable_thinking: bool = False,
    use_generation_prompt_for_labels: bool = True,
    **extra_kwargs,
) -> Dict[str, tc.Tensor]:
    require_chat_template(tokenizer)
    full_messages = [dict(msg) for msg in messages if msg.get("content") is not None]
    tokenized = None
    try:
        tokenized = apply_standard_chat_template(
            tokenizer,
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **extra_kwargs,
        )
    except (TypeError, ValueError):
        tokenized = None

    if isinstance(tokenized, Mapping) and _extract_assistant_mask(tokenized) is not None:
        input_ids = _as_list(tokenized.get("input_ids"))
        assistant_mask = _extract_assistant_mask(tokenized) or []
        if add_eos and getattr(tokenizer, "eos_token_id", None) is not None:
            input_ids.append(int(tokenizer.eos_token_id))
            assistant_mask.append(1)
        labels = [tok if idx < len(assistant_mask) and int(assistant_mask[idx]) else -100 for idx, tok in enumerate(input_ids)]
    else:
        assistant_idx = next((idx for idx, msg in enumerate(full_messages) if str(msg.get("role", "")).lower() == "assistant"), None)
        if assistant_idx is None:
            prompt_messages = full_messages
        else:
            prompt_messages = full_messages[:assistant_idx]
        prompt_text = apply_standard_chat_template(
            tokenizer,
            prompt_messages,
            tokenize=False,
            add_generation_prompt=bool(use_generation_prompt_for_labels),
            enable_thinking=enable_thinking,
            **extra_kwargs,
        )
        full_text = apply_standard_chat_template(
            tokenizer,
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            **extra_kwargs,
        )
        if add_eos and getattr(tokenizer, "eos_token", None):
            full_text = full_text + tokenizer.eos_token
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prefix_len = _prefix_len_or_raise(input_ids, prompt_ids)
        labels = [-100] * prefix_len + input_ids[prefix_len:]

    if len(input_ids) > int(max_len):
        input_ids = input_ids[-int(max_len) :]
        labels = labels[-int(max_len) :]
    if not input_ids:
        fallback = getattr(tokenizer, "eos_token_id", None) or 0
        input_ids = [int(fallback)]
        labels = [int(fallback)]
    return {
        "input_ids": tc.tensor(input_ids, dtype=tc.long),
        "labels": tc.tensor(labels, dtype=tc.long),
    }
