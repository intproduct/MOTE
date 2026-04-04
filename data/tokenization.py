from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch as tc


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s


def make_supervised_example(tokenizer, prompt: str, answer: str, max_len: int, add_eos: bool = True) -> Dict[str, tc.Tensor]:
    prompt = normalize_text(prompt)
    answer = normalize_text(answer)
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_text = answer + (tokenizer.eos_token if (add_eos and tokenizer.eos_token) else "")
    a_ids = tokenizer.encode(a_text, add_special_tokens=False)
    input_ids = p_ids + a_ids
    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        if overflow >= len(p_ids):
            cut_from_answer = overflow - len(p_ids)
            p_ids = []
            a_ids = a_ids[cut_from_answer:]
        else:
            p_ids = p_ids[overflow:]
        input_ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids[:]
    if not input_ids:
        fallback = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        input_ids = [fallback]
        labels = [fallback]
    return {
        "input_ids": tc.tensor(input_ids, dtype=tc.long),
        "labels": tc.tensor(labels, dtype=tc.long),
    }


def make_causal_lm_example_from_text(tokenizer, text: str, max_len: int, add_eos: bool = True) -> Dict[str, tc.Tensor]:
    text = normalize_text(text)
    if add_eos and tokenizer.eos_token:
        text = text + tokenizer.eos_token
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    return {
        "input_ids": tc.tensor(ids, dtype=tc.long),
        "labels": tc.tensor(ids, dtype=tc.long),
    }


def make_chat_supervised_example(tokenizer, messages: List[Dict[str, str]], max_len: int, add_eos: bool = True) -> Dict[str, tc.Tensor]:
    input_ids: List[int] = []
    labels: List[int] = []

    def _append_text(text: str, trainable: bool):
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        labels.extend(ids if trainable else ([-100] * len(ids)))

    for msg in messages:
        role = normalize_text(msg.get("role", "")).lower()
        content = normalize_text(msg.get("content", ""))
        if not content:
            continue
        if role == "system":
            _append_text(f"<|im_start|>system\n{content}\n<|im_end|>\n", trainable=False)
        elif role == "user":
            _append_text(f"<|im_start|>user\n{content}\n<|im_end|>\n", trainable=False)
        elif role == "assistant":
            _append_text("<|im_start|>assistant\n", trainable=False)
            _append_text(content, trainable=True)
            _append_text("\n<|im_end|>\n", trainable=True)
        else:
            _append_text(f"{content}\n", trainable=False)

    if add_eos and tokenizer.eos_token:
        eos_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        input_ids.extend(eos_ids)
        labels.extend(eos_ids)

    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]

    if not input_ids:
        fallback = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        input_ids = [fallback]
        labels = [fallback]

    return {
        "input_ids": tc.tensor(input_ids, dtype=tc.long),
        "labels": tc.tensor(labels, dtype=tc.long),
    }


def build_example_from_token_ids(token_ids: List[int], max_len: int, eos_id: Optional[int] = None) -> Dict[str, tc.Tensor]:
    ids = [int(x) for x in token_ids if x is not None][:max_len]
    if not ids:
        ids = [eos_id or 0]
    return {
        "input_ids": tc.tensor(ids, dtype=tc.long),
        "labels": tc.tensor(ids, dtype=tc.long),
    }
