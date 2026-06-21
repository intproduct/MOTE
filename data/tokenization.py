from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch as tc

from ..chat_formatting import make_standard_chat_supervised_example


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s


def make_supervised_example(
    tokenizer,
    prompt: str,
    answer: str,
    max_len: int,
    add_eos: bool = True,
    *,
    final_answer_weight_enabled: bool = False,
    final_answer_weight: float = 1.0,
    final_answer_marker: str = "####",
) -> Dict[str, tc.Tensor]:
    prompt = normalize_text(prompt)
    answer = normalize_text(answer)
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_text = answer + (tokenizer.eos_token if (add_eos and tokenizer.eos_token) else "")
    a_ids = tokenizer.encode(a_text, add_special_tokens=False)
    answer_weights = [1.0] * len(a_ids)
    if final_answer_weight_enabled and float(final_answer_weight) > 1.0 and final_answer_marker:
        marker_idx = answer.find(str(final_answer_marker))
        if marker_idx >= 0:
            marker_end = marker_idx + len(str(final_answer_marker))
            weight_start = len(tokenizer.encode(answer[:marker_end], add_special_tokens=False))
            for idx in range(min(weight_start, len(answer_weights)), len(answer_weights)):
                answer_weights[idx] = float(final_answer_weight)
    input_ids = p_ids + a_ids
    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        if overflow >= len(p_ids):
            cut_from_answer = overflow - len(p_ids)
            p_ids = []
            a_ids = a_ids[cut_from_answer:]
            answer_weights = answer_weights[cut_from_answer:]
        else:
            p_ids = p_ids[overflow:]
        input_ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids[:]
    loss_weights = [0.0] * len(p_ids) + answer_weights[:]
    if not input_ids:
        fallback = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        input_ids = [fallback]
        labels = [fallback]
        loss_weights = [1.0]
    result = {
        "input_ids": tc.tensor(input_ids, dtype=tc.long),
        "labels": tc.tensor(labels, dtype=tc.long),
    }
    if final_answer_weight_enabled and float(final_answer_weight) != 1.0:
        result["loss_weights"] = tc.tensor(loss_weights, dtype=tc.float32)
    return result


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
    return make_standard_chat_supervised_example(tokenizer, messages=messages, max_len=max_len, add_eos=add_eos)


def build_example_from_token_ids(token_ids: List[int], max_len: int, eos_id: Optional[int] = None) -> Dict[str, tc.Tensor]:
    ids = [int(x) for x in token_ids if x is not None][:max_len]
    if not ids:
        ids = [eos_id or 0]
    return {
        "input_ids": tc.tensor(ids, dtype=tc.long),
        "labels": tc.tensor(ids, dtype=tc.long),
    }
