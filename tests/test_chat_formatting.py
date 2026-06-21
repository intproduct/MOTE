from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg

from MOTE.chat_formatting import (
    apply_standard_chat_template,
    build_chat_prompt_text_for_generation,
    build_reasoning_messages,
    make_standard_chat_supervised_example,
)
from MOTE.data.tokenization import make_supervised_example


class FakeChatTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    chat_template = "fake-template"

    def __init__(self, support_enable_thinking=True, support_assistant_mask=True):
        self.support_enable_thinking = support_enable_thinking
        self.support_assistant_mask = support_assistant_mask
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) % 251 + 3 for ch in str(text)]

    def apply_chat_template(self, messages, **kwargs):
        if not self.support_enable_thinking and "enable_thinking" in kwargs:
            raise TypeError("enable_thinking unsupported")
        self.calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        text = "".join(f"[{msg['role'].upper()}]{msg['content']}[/]" for msg in messages)
        if kwargs.get("add_generation_prompt"):
            text += "[ASSISTANT]"
        if not kwargs.get("tokenize"):
            return text
        input_ids = self.encode(text, add_special_tokens=False)
        if kwargs.get("return_dict"):
            payload = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
            if self.support_assistant_mask and kwargs.get("return_assistant_tokens_mask"):
                assistant_start = text.find("[ASSISTANT]")
                payload["assistant_tokens_mask"] = [1 if idx >= assistant_start and assistant_start >= 0 else 0 for idx in range(len(input_ids))]
            return payload
        return input_ids


class NoChatTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2


def test_apply_standard_chat_template_uses_tokenizer_and_falls_back_without_enable_thinking():
    tokenizer = FakeChatTokenizer(support_enable_thinking=False)
    text = apply_standard_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert "[USER]hello" in text
    assert "<|im_start|>" not in text
    assert "enable_thinking" not in tokenizer.calls[-1]["kwargs"]


def test_chat_template_required_for_chat_helpers():
    try:
        build_chat_prompt_text_for_generation(NoChatTokenizer(), "Question:\n1+1?")
    except ValueError as exc:
        assert "apply_chat_template" in str(exc)
    else:
        raise AssertionError("missing chat template should fail")


def test_chat_supervised_assistant_mask_labels_only_assistant_tokens():
    tokenizer = FakeChatTokenizer(support_assistant_mask=True)
    messages = build_reasoning_messages("q", target="Solution:\ns\n\nFinal Answer:\n2", system_prompt="sys")
    ex = make_standard_chat_supervised_example(tokenizer, messages, max_len=4096, add_eos=False)
    labels = ex["labels"].tolist()
    assert any(value == -100 for value in labels)
    assert any(value != -100 for value in labels)
    first_trainable = next(idx for idx, value in enumerate(labels) if value != -100)
    assert all(value == -100 for value in labels[:first_trainable])


def test_chat_supervised_prefix_fallback_labels_only_suffix():
    tokenizer = FakeChatTokenizer(support_assistant_mask=False)
    messages = build_reasoning_messages("q", target="Solution:\ns\n\nFinal Answer:\n2")
    ex = make_standard_chat_supervised_example(tokenizer, messages, max_len=4096, add_eos=False)
    labels = ex["labels"].tolist()
    assert any(value == -100 for value in labels)
    assert any(value != -100 for value in labels)


def test_raw_supervised_example_still_masks_prompt():
    tokenizer = FakeChatTokenizer()
    ex = make_supervised_example(tokenizer, "Question:\nq", "Solution:\ns", max_len=4096, add_eos=False)
    labels = ex["labels"].tolist()
    assert labels[: len(tokenizer.encode("Question:\nq", add_special_tokens=False))] == [-100] * len(
        tokenizer.encode("Question:\nq", add_special_tokens=False)
    )
