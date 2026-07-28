from __future__ import annotations

import pytest

from MOTE.chat_formatting import build_reasoning_messages, make_standard_chat_supervised_example
from MOTE.data.supervision import SupervisionError
from MOTE.data.tokenization import make_supervised_example


class CharacterTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    chat_template = "fake"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) + 10 for char in str(text)]

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"[{item['role']}]{item['content']}" for item in messages)
        if kwargs.get("add_generation_prompt"):
            text += "[assistant]"
        if not kwargs.get("tokenize"):
            return text
        ids = self.encode(text)
        if kwargs.get("return_dict"):
            assistant_at = text.find("[assistant]")
            mask = [int(assistant_at >= 0 and index >= assistant_at) for index in range(len(ids))]
            return {"input_ids": ids, "assistant_tokens_mask": mask}
        return ids


def test_raw_sft_rejects_overflow_without_truncating_prompt_or_target():
    tokenizer = CharacterTokenizer()
    with pytest.raises(SupervisionError) as exc:
        make_supervised_example(tokenizer, "question", "final-answer", max_len=8, add_eos=False)
    assert exc.value.reason_code == "overlong"
    assert exc.value.diagnostics.prompt_tokens == len("question")
    assert exc.value.diagnostics.target_tokens == len("final-answer")
    assert exc.value.diagnostics.overflow_tokens > 0


def test_raw_sft_reports_exact_supervision_diagnostics_and_single_eos():
    tokenizer = CharacterTokenizer()
    example = make_supervised_example(tokenizer, "Q\n  code", "A", max_len=64, add_eos=True)
    diagnostics = example["data_diagnostics"]
    assert diagnostics["prompt_tokens"] == len("Q\n  code")
    assert diagnostics["target_tokens"] == 2
    assert diagnostics["supervised_tokens"] == 2
    assert example["input_ids"][-1].item() == tokenizer.eos_token_id
    assert example["labels"][: diagnostics["prompt_tokens"]].tolist() == [-100] * diagnostics["prompt_tokens"]


def test_chat_sft_rejects_overflow_instead_of_keeping_only_sequence_tail():
    tokenizer = CharacterTokenizer()
    messages = build_reasoning_messages("long-question", target="final-answer")
    with pytest.raises(SupervisionError) as exc:
        make_standard_chat_supervised_example(tokenizer, messages, max_len=10, add_eos=False)
    assert exc.value.reason_code == "overlong"
    assert exc.value.diagnostics.total_tokens > 10


def test_chat_sft_does_not_duplicate_template_eos():
    class EosTerminatedTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if kwargs.get("tokenize") and kwargs.get("return_dict"):
                result["input_ids"].append(self.eos_token_id)
                result["assistant_tokens_mask"].append(1)
            return result

    tokenizer = EosTerminatedTokenizer()
    messages = build_reasoning_messages("question", target="answer")
    example = make_standard_chat_supervised_example(tokenizer, messages, max_len=256, add_eos=True)
    assert example["input_ids"].tolist().count(tokenizer.eos_token_id) == 1
    assert example["input_ids"][-1].item() == tokenizer.eos_token_id
    assert example["labels"][-1].item() == tokenizer.eos_token_id


def test_multiturn_fallback_refuses_to_supervise_user_tokens():
    class NoMaskTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("return_assistant_tokens_mask"):
                raise ValueError("mask unsupported")
            return super().apply_chat_template(messages, **kwargs)

    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    with pytest.raises(ValueError, match="refusing to supervise user/tool tokens"):
        make_standard_chat_supervised_example(NoMaskTokenizer(), messages, max_len=256)
