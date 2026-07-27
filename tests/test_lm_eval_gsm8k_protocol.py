from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg

from MOTE.cli import build_boundary_gsm8k as boundary_cli
from MOTE.config.defaults import make_default_config
from MOTE.eval.lm_eval_gsm8k_protocol import (
    LMEvalGSM8KProtocol,
    resolve_lm_eval_num_fewshot,
)
from MOTE.rl.boundary import pass_at_k_estimate
from MOTE.rl.rollout_backends import HFRolloutBackend, RolloutGenerationConfig


class FakeInstance:
    def __init__(self, doc_id, prompt, generation_kwargs):
        self.doc_id = doc_id
        self.idx = 0
        self.arguments = (prompt, dict(generation_kwargs))
        self.resps = []
        self.filtered_resps = {}


class SyntheticGSM8KTask:
    def __init__(self, count=16):
        self.docs = [
            {"question": f"question-{index}", "answer": f"worked #### {index}", "target": str(index)}
            for index in range(count)
        ]
        self.config = types.SimpleNamespace(
            task="gsm8k",
            dataset_path="synthetic/gsm8k",
            dataset_name="main",
            test_split="test",
            fewshot_split="train",
            output_type="generate_until",
            generation_kwargs={"until": ["Question:"], "do_sample": False},
            num_fewshot=5,
            repeats=1,
        )
        self.instances = []

    def get_config(self, key):
        return getattr(self.config, key, None)

    def set_config(self, key, value, update=False):
        if update and isinstance(value, dict):
            current = dict(getattr(self.config, key, {}) or {})
            current.update(value)
            value = current
        setattr(self.config, key, value)

    def set_fewshot_seed(self, seed):
        self.fewshot_seed = seed

    def build_all_requests(self, limit, **kwargs):
        del kwargs
        count = len(self.docs) if limit is None else min(int(limit), len(self.docs))
        self.instances = [
            FakeInstance(
                index,
                f"EXACT-FEW-SHOT-{self.config.num_fewshot}\nQuestion: {doc['question']}\nAnswer:",
                self.config.generation_kwargs,
            )
            for index, doc in enumerate(self.docs[:count])
        ]

    def doc_iterator(self, rank, limit, world_size, samples=None):
        del rank, world_size, samples
        count = len(self.docs) if limit is None else min(int(limit), len(self.docs))
        yield from enumerate(self.docs[:count])

    def apply_filters(self):
        for instance in self.instances:
            text = instance.resps[0]
            match = re.search(r"FORMAL:(-?[0-9]+)", text)
            instance.filtered_resps["strict-match"] = match.group(1) if match else "[invalid]"
            instance.filtered_resps["flexible-extract"] = "unused"

    def process_results(self, doc, results):
        return {"exact_match": float(results[0] == doc["target"])}

    def doc_to_target(self, doc):
        return doc["target"]


def make_protocol(*, count=16, fewshot=8, max_prompts=None):
    return LMEvalGSM8KProtocol(
        task=SyntheticGSM8KTask(count=count),
        num_fewshot=fewshot,
        runtime={"apply_chat_template": False, "enable_thinking": False},
        generation_overrides={"max_gen_toks": 256},
        seed=1234,
        max_prompts=max_prompts,
        lm_eval_version="synthetic",
    )


def test_protocol_cli_defaults_preserve_rl_and_resolve_lm_eval_test_split():
    common = ["--config_json", "c.json", "--output_jsonl", "o.jsonl", "--verified_traces_jsonl", "v.jsonl"]
    rl = boundary_cli.parse_args(common)
    lm_eval = boundary_cli.parse_args(common + ["--protocol", "lm_eval"])
    assert rl.protocol == "rl"
    assert boundary_cli._resolved_split(rl) == "train"
    assert boundary_cli._resolved_split(lm_eval) == "test"


def test_num_fewshot_uses_project_config_and_cli_override():
    cfg = make_default_config()
    cfg.eval.fewshot.task_overrides["gsm8k"] = 8
    assert resolve_lm_eval_num_fewshot(cfg, None) == 8
    assert resolve_lm_eval_num_fewshot(cfg, 3) == 3


def test_synthetic_task_prompt_and_stable_test_subset_are_exact():
    protocol = make_protocol(count=4, fewshot=8, max_prompts=2)
    rows = protocol.boundary_rows()
    assert [row["dataset_index"] for row in rows] == [0, 1]
    assert rows[0]["_lm_eval_prompt"] == "EXACT-FEW-SHOT-8\nQuestion: question-0\nAnswer:"
    assert protocol.metadata()["scorer"] == "exact_match,strict-match"


def test_lm_eval_scoring_reuses_filter_and_isolated_from_rl_reward():
    protocol = make_protocol(count=1)
    row = protocol.boundary_rows()[0]
    completion = "reasoning FORMAL:0\nthen overwritten #### 96"
    reward, _debug, match_type, diagnostic = boundary_cli._score_rollout(
        "lm_eval", row, completion, protocol
    )
    assert reward == 1.0
    assert match_type == "lm_eval_strict_match"
    assert diagnostic["lm_eval_correct"] is True
    assert diagnostic["rl_correct"] is False
    assert diagnostic["parser_disagreement"] is True


def test_parser_disagreement_is_diagnostic_only():
    protocol = make_protocol(count=1)
    row = protocol.boundary_rows()[0]
    reward, _debug, _match_type, diagnostic = boundary_cli._score_rollout(
        "lm_eval", row, "FORMAL:99\n#### 0", protocol
    )
    assert reward == 0.0
    assert diagnostic["rl_correct"] is True
    assert diagnostic["lm_eval_correct"] is False


def test_lm_eval_prompt_iteration_never_calls_rl_builder(monkeypatch):
    protocol = make_protocol(count=2)
    records = protocol.boundary_rows()
    monkeypatch.setattr(
        boundary_cli,
        "build_rl_prompt_text",
        lambda *_args: (_ for _ in ()).throw(AssertionError("RL prompt builder called")),
    )
    monkeypatch.setattr(
        boundary_cli,
        "_vllm_samples_batch",
        lambda **kwargs: (
            [[{"text": "", "token_ids": [], "response_token_count": 0, "finish_reason": "stop", "truncated": False}] for _ in kwargs["prompts"]],
            {"prompt_count": len(kwargs["prompts"]), "row_count": len(kwargs["prompts"]), "actors": {}},
        ),
    )
    args = types.SimpleNamespace(
        protocol="lm_eval", decoding="greedy", rollout_backend="vllm", prompt_batch_size=2,
        num_rollouts=1, assert_multi_actor_dispatch=False,
    )
    cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(seq_len_run=2048),
        rl=types.SimpleNamespace(rollout_max_prompt_tokens=9),
    )
    values = list(boundary_cli._iter_prompt_samples(
        cfg=cfg, args=args, records=records, backend=object(), model=None, tokenizer=object(),
        generation_config=types.SimpleNamespace(max_new_tokens=256), seed=7,
    ))
    assert [value[2] for value in values] == [row["_lm_eval_prompt"] for row in records]


def test_rl_prompt_iteration_still_calls_original_builder(monkeypatch):
    calls = []
    monkeypatch.setattr(boundary_cli, "build_rl_prompt_text", lambda _c, _t, q: calls.append(q) or f"RL:{q}")
    monkeypatch.setattr(
        boundary_cli,
        "_vllm_samples_batch",
        lambda **kwargs: (
            [[{"text": "", "token_ids": [], "response_token_count": 0, "finish_reason": "stop", "truncated": False}] for _ in kwargs["prompts"]],
            {"prompt_count": len(kwargs["prompts"]), "row_count": len(kwargs["prompts"]), "actors": {}},
        ),
    )
    args = types.SimpleNamespace(
        protocol="rl", decoding="sample", rollout_backend="vllm", prompt_batch_size=2,
        num_rollouts=1, assert_multi_actor_dispatch=False,
    )
    cfg = types.SimpleNamespace(rl=types.SimpleNamespace(rollout_max_prompt_tokens=9))
    records = [{"question": "q0"}, {"question": "q1"}]
    values = list(boundary_cli._iter_prompt_samples(
        cfg=cfg, args=args, records=records, backend=object(), model=None, tokenizer=object(),
        generation_config=types.SimpleNamespace(), seed=7,
    ))
    assert calls == ["q0", "q1"]
    assert [value[2] for value in values] == ["RL:q0", "RL:q1"]


def test_generation_config_true_greedy_and_stochastic_sample():
    cfg = make_default_config()
    protocol = make_protocol(count=1)
    rows = protocol.boundary_rows()
    greedy_args = types.SimpleNamespace(
        protocol="lm_eval", decoding="greedy", num_rollouts=1,
        max_new_tokens=None, temperature=None, top_p=None,
    )
    greedy, greedy_meta = boundary_cli._make_generation_config(cfg, greedy_args, 1, rows)
    assert greedy.do_sample is False
    assert greedy_meta["temperature"] == 0.0
    assert greedy.stop_sequences == ("Question:",)
    sample_args = types.SimpleNamespace(
        protocol="lm_eval", decoding="sample", num_rollouts=16,
        max_new_tokens=None, temperature=0.7, top_p=0.95,
    )
    sample, sample_meta = boundary_cli._make_generation_config(cfg, sample_args, 1, rows)
    assert sample.do_sample is True
    assert sample_meta["temperature"] == 0.7
    assert sample_meta["top_p"] == 0.95


def test_hf_backend_true_greedy_omits_sampling_controls_and_passes_stops():
    import torch

    observed = {}

    class Model:
        def generate(self, input_ids, attention_mask, **kwargs):
            del attention_mask
            observed.update(kwargs)
            suffix = torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype)
            return torch.cat([input_ids, suffix], dim=1)

    tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=2)
    backend = HFRolloutBackend()
    input_ids = torch.tensor([[4, 5]])
    backend.generate(
        model=Model(), tokenizer=tokenizer, prompts=["p"], input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        generation_config=RolloutGenerationConfig(
            max_new_tokens=8, temperature=0.7, top_p=0.95, do_sample=False,
            stop_sequences=("Question:",),
        ),
        update_step=0,
    )
    assert observed["do_sample"] is False
    assert "temperature" not in observed
    assert "top_p" not in observed
    assert observed["stop_strings"] == ["Question:"]
    assert observed["tokenizer"] is tokenizer


def test_pass_at_k_formula_is_unchanged():
    assert pass_at_k_estimate(16, 6, 4) == pytest.approx(1 - 210 / 1820)


class TokenLengthTokenizer:
    eos_token_id = 2
    truncation_side = "right"
    padding_side = "right"

    def __call__(
        self,
        prompts,
        *,
        return_tensors,
        padding,
        add_special_tokens,
        truncation=False,
        max_length=None,
    ):
        import torch

        del return_tensors, padding, add_special_tokens
        rows = [list(range(1, len(prompt) + 1)) for prompt in prompts]
        if truncation:
            assert self.truncation_side == "left"
            rows = [row[-int(max_length):] for row in rows]
        width = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad = [0] * (width - len(row))
            input_ids.append(pad + row)
            attention_mask.append([0] * len(pad) + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


def test_lm_eval_static_prompt_uses_hflm_left_truncation_budget():
    observed = {}

    class Backend:
        last_actor_dispatch_metadata = {
            "actor_count": 2,
            "active_actor_count": 2,
            "actors": {},
        }

        def generate_static_samples_batch(self, **kwargs):
            observed.update(kwargs)
            return [[{"token_ids": [9], "text": "x", "finish_reason": "stop"}]]

    tokenizer = TokenLengthTokenizer()
    results, dispatch = boundary_cli._vllm_samples_batch(
        backend=Backend(), tokenizer=tokenizer, prompts=["x" * 2115],
        num_rollouts=1,
        generation_config=types.SimpleNamespace(
            max_new_tokens=256, temperature=0.0, top_p=1.0,
            stop_sequences=(), top_k=None,
        ),
        prompt_seeds=[7], prompt_indices=[3], max_prompt_tokens=1792,
        decoding="greedy", hflm_compatible=True,
    )
    assert len(observed["prompt_token_ids"][0]) == 1792
    assert observed["prompt_token_ids"][0][0] == 324
    assert tokenizer.padding_side == "right"
    assert results[0][0]["text"] == "x"
    audit = dispatch["prompt_tokenization"]["prompts"][0]
    assert audit["prompt_index"] == 3
    assert audit["raw_prompt_token_count"] == 2115
    assert audit["effective_prompt_token_count"] == 1792
    assert audit["truncated_prompt_token_count"] == 323
    assert audit["prompt_truncated"] is True


def test_lm_eval_prompt_budget_matches_hflm_and_rejects_undersized_actor():
    cfg = make_default_config()
    cfg.data.seq_len_run = 2048
    cfg.rl.vllm_max_model_len = 2048
    generation = types.SimpleNamespace(max_new_tokens=256)
    assert boundary_cli._lm_eval_max_prompt_tokens(
        cfg, generation, rollout_backend="vllm"
    ) == 1792
    cfg.rl.vllm_max_model_len = 1024
    with pytest.raises(ValueError, match="undersized_actors"):
        boundary_cli._lm_eval_max_prompt_tokens(
            cfg, generation, rollout_backend="vllm"
        )


def test_prompt_dump_observer_runs_before_vllm_generation_failure(monkeypatch):
    protocol = make_protocol(count=1)
    records = protocol.boundary_rows()
    observed = []
    monkeypatch.setattr(
        boundary_cli,
        "_vllm_samples_batch",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected generate failure")),
    )
    cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(seq_len_run=2048),
        rl=types.SimpleNamespace(rollout_max_prompt_tokens=0),
    )
    args = types.SimpleNamespace(
        protocol="lm_eval", decoding="greedy", rollout_backend="vllm",
        prompt_batch_size=1, num_rollouts=1, assert_multi_actor_dispatch=False,
    )
    with pytest.raises(RuntimeError, match="injected generate failure"):
        list(boundary_cli._iter_prompt_samples(
            cfg=cfg, args=args, records=records, backend=object(), model=None,
            tokenizer=object(),
            generation_config=types.SimpleNamespace(max_new_tokens=256), seed=7,
            prompt_observer=lambda index, row, prompt: observed.append((index, row, prompt)),
        ))
    assert observed and observed[0][0] == 0
