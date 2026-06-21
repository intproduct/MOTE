from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg

from MOTE.config.defaults import make_default_config
from MOTE.eval import lm_eval_hf


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, **kwargs):
        return "chat"


class FakeHFLM:
    last_kwargs = None

    def __init__(self, pretrained, tokenizer, batch_size, max_length, trust_remote_code, device, enable_thinking=None, chat_template_args=None):
        FakeHFLM.last_kwargs = dict(
            pretrained=pretrained,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            device=device,
            enable_thinking=enable_thinking,
            chat_template_args=chat_template_args,
        )


class FakeEvaluator:
    last_kwargs = None

    @staticmethod
    def simple_evaluate(
        model,
        tasks,
        num_fewshot,
        batch_size,
        log_samples,
        apply_chat_template=None,
        fewshot_as_multiturn=None,
        system_instruction=None,
        metadata=None,
        confirm_run_unsafe_code=None,
        gen_kwargs=None,
        limit=None,
    ):
        FakeEvaluator.last_kwargs = dict(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            log_samples=log_samples,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            system_instruction=system_instruction,
            metadata=metadata,
            confirm_run_unsafe_code=confirm_run_unsafe_code,
            gen_kwargs=gen_kwargs,
            limit=limit,
        )
        return {"results": {tasks[0]: {"exact_match,strict-match": 0.5}}}


class FakeModel:
    def eval(self):
        return self


def test_lm_eval_chat_runtime_kwargs_and_summary_are_recorded():
    cfg = make_default_config()
    cfg.eval.runtime.apply_chat_template = True
    cfg.eval.runtime.enable_thinking = False
    cfg.eval.runtime.system_instruction = "sys"
    cfg.eval.runtime.chat_template_args = {"foo": "bar"}
    cfg.eval.final_tasks = ["gsm8k"]

    with patch.object(lm_eval_hf, "import_lm_eval", return_value=(FakeEvaluator, FakeHFLM)):
        result = lm_eval_hf.run_lm_eval_tasks(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            fit_cfg=cfg,
            tasks=["gsm8k"],
            eval_name="chat_eval_test",
            eval_mode="final",
        )

    assert FakeHFLM.last_kwargs["enable_thinking"] is False
    assert FakeHFLM.last_kwargs["chat_template_args"] == {"foo": "bar"}
    assert FakeEvaluator.last_kwargs["apply_chat_template"] is True
    assert FakeEvaluator.last_kwargs["system_instruction"] == "sys"
    assert result["summary"]["gsm8k"]["apply_chat_template"] is True
    assert result["summary"]["gsm8k"]["enable_thinking"] is False
