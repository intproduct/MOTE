from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence
from functools import lru_cache
import re

import torch as tc

from .data.supervision import validate_supervision


ASSISTANT_MASK_KEYS = (
    "assistant_masks",
    "assistant_mask",
    "assistant_tokens_mask",
    "assistant_token_mask",
    "assistant_masks",
)


@lru_cache(maxsize=32)
def supervision_chat_template(template: str) -> str:
    """Annotate the native Qwen3.5/MixT assistant branch without changing text.

    Restrict the rewrite to the recognized template structure. The enclosing
    generation block covers thinking, tool calls and the assistant terminator;
    user and tool-response branches remain outside it.
    """
    if re.search(r"\{%[-+]?\s*generation\b", template):
        return template
    start = '{%- elif message.role == "assistant" %}'
    end = '{%- elif message.role == "tool" %}'
    if (template.count(start) == 1 and template.count(end) == 1 and
            "ns.last_query_index" in template and "<tool_response>" in template and
            template.index(start) < template.index(end)):
        return template.replace(start, start + "{%- generation -%}").replace(end, "{%- endgeneration -%}" + end)
    return template


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
) -> Dict[str, Any]:
    require_chat_template(tokenizer)
    full_messages = [dict(msg, content=msg.get("content") or "") for msg in messages
                     if msg.get("content") is not None or msg.get("tool_calls") or msg.get("reasoning_content")]
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and "chat_template" not in extra_kwargs:
        annotated = supervision_chat_template(template)
        if annotated != template:
            extra_kwargs["chat_template"] = annotated
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

    assistant_mask = _extract_assistant_mask(tokenized) if isinstance(tokenized, Mapping) else None
    # HF can return an all-zero mask when the template has no generation tags.
    # Treat that as unavailable: appending EOS must not turn an entirely masked
    # answer into an apparently valid one-token training example.
    if assistant_mask is not None and any(assistant_mask):
        input_ids = _as_list(tokenized.get("input_ids"))
        labels = [tok if idx < len(assistant_mask) and int(assistant_mask[idx]) else -100 for idx, tok in enumerate(input_ids)]
    else:
        assistant_idx = next((idx for idx, msg in enumerate(full_messages) if str(msg.get("role", "")).lower() == "assistant"), None)
        if assistant_idx is None:
            prompt_messages = full_messages
        else:
            trailing_non_assistant = [
                str(msg.get("role", "")).lower()
                for msg in full_messages[assistant_idx + 1 :]
                if str(msg.get("role", "")).lower() != "assistant"
            ]
            if trailing_non_assistant:
                raise ValueError(
                    "multi-turn chat supervision requires a tokenizer that supports "
                    "return_assistant_tokens_mask; refusing to supervise user/tool tokens"
                )
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
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prefix_len = _prefix_len_or_raise(input_ids, prompt_ids)
        labels = [-100] * prefix_len + input_ids[prefix_len:]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    already_terminated = bool(input_ids and eos_id is not None and int(input_ids[-1]) == int(eos_id))
    if not already_terminated and eos_id is not None and eos_id in input_ids and hasattr(tokenizer, "decode"):
        last_eos = len(input_ids) - 1 - input_ids[::-1].index(eos_id)
        already_terminated = not tokenizer.decode(input_ids[last_eos + 1:], skip_special_tokens=False).strip()
    if add_eos and eos_id is not None and not already_terminated:
        input_ids.append(int(eos_id))
        labels.append(int(eos_id))

    supervised_tokens = sum(int(label) != -100 for label in labels)
    diagnostics = validate_supervision(
        input_ids,
        labels,
        max_length=max_len,
        prompt_tokens=len(input_ids) - supervised_tokens,
        target_tokens=supervised_tokens,
    )
    return {
        "input_ids": tc.tensor(input_ids, dtype=tc.long),
        "labels": tc.tensor(labels, dtype=tc.long),
        "data_diagnostics": diagnostics.to_dict(),
    }
