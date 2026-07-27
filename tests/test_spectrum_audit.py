from __future__ import annotations

import sys
import types
import signal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg

from MOTE.cli.build_boundary_gsm8k import parse_args
from MOTE.cli import build_boundary_gsm8k as boundary_cli
from MOTE.rl.boundary import (
    aggregate_spectrum,
    build_prompt_audit_record,
    classify_prompt,
    pass_at_k_estimate,
    reward_match_type,
)
from MOTE.rl.vllm_actor import ActorCompletion, ActorRequestOutput
from MOTE.rl.vllm_rollout import VLLMRolloutBackend
from MOTE.config.defaults import make_default_config


def _rollout(reward, tokens=4, truncated=False, match_type=None):
    debug = {
        "match_type": match_type or ("strict_hash" if reward else "none"),
        "pred_answer": "1" if reward else "2",
        "strict_match": match_type in (None, "strict_hash") and bool(reward),
        "fallback_match": match_type == "fallback_last_number",
    }
    return {
        "reward": float(reward),
        "reward_debug": debug,
        "reward_match_type": reward_match_type(debug, reward),
        "response_token_count": tokens,
        "truncated": truncated,
    }


def test_pass_at_k_standard_estimator_and_edges():
    assert pass_at_k_estimate(16, 0, 1) == 0.0
    assert pass_at_k_estimate(16, 6, 1) == pytest.approx(6 / 16)
    assert pass_at_k_estimate(16, 6, 4) == pytest.approx(1 - 210 / 1820)
    assert pass_at_k_estimate(16, 1, 16) == 1.0
    with pytest.raises(ValueError):
        pass_at_k_estimate(4, 1, 5)


@pytest.mark.parametrize(
    "correct,expected",
    [(0, (True, False, False)), (2, (False, False, True)), (4, (False, True, False))],
)
def test_prompt_classification(correct, expected):
    value = classify_prompt(correct, 4, boundary_min=.25, boundary_max=.75)
    assert (value["all_wrong"], value["all_correct"], value["mixed"]) == expected
    assert value["effective_rl_prompt"] == value["mixed"]
    assert value["zero_advantage_prompt"] != value["mixed"]


def test_prompt_boundary_is_independent_from_selection_range():
    value = classify_prompt(1, 4, boundary_min=.25, boundary_max=.75)
    assert value["boundary"] is True
    value = classify_prompt(1, 4, boundary_min=.5, boundary_max=.75)
    assert value["boundary"] is False


def test_summary_histogram_reward_length_and_truncation_aggregation():
    rows = []
    compact = []
    for idx, rewards in enumerate(([0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1])):
        rollouts = [_rollout(reward, tokens=idx + 2, truncated=(idx == 1 and j == 0)) for j, reward in enumerate(rewards)]
        rows.append(build_prompt_audit_record(
            {"idx": idx, "question": "q", "answer": "#### 1"}, rollouts,
            pass_k=[1, 4], boundary_min=.25, boundary_max=.75,
        ))
        compact.extend(rollouts)
    result = aggregate_spectrum(rows, compact, pass_k=[1, 4], mgpo_enabled=False)
    assert result["spectrum"]["correct_count_histogram"] == {"0": 1, "1": 1, "4": 1}
    assert result["spectrum"]["all_wrong_count"] == 1
    assert result["spectrum"]["all_correct_count"] == 1
    assert result["length_diagnostics"]["truncated_rollout_count"] == 1
    assert result["length_diagnostics"]["maximum_response_tokens"] == 4
    assert result["reward_diagnostics"]["reward_accuracy"] == pytest.approx(5 / 12)


def test_fallback_only_is_exclusive_from_strict_match():
    strict_and_fallback = {"match_type": "strict_hash", "strict_match": True, "fallback_match": True}
    fallback = {"match_type": "fallback_last_number", "strict_match": False, "fallback_match": True}
    assert reward_match_type(strict_and_fallback, 1.0) == "strict_hash_match"
    assert reward_match_type(fallback, 1.0) == "fallback_last_number_only"
    assert reward_match_type({"match_type": "none", "pred_answer": None}, 0.0) == "unparseable"


def test_mgpo_weight_uses_project_function_and_aggregates():
    cfg = types.SimpleNamespace(
        mgpo_enabled=True, mgpo_p0=.5, mgpo_gamma=2.0,
        mgpo_weight_min=.1, mgpo_weight_max=1.0, mgpo_eps=1e-6,
    )
    prompt = build_prompt_audit_record(
        {"question": "q", "answer": "1"}, [_rollout(1), _rollout(0)],
        pass_k=[1, 2], boundary_min=.25, boundary_max=.75, mgpo_cfg=cfg,
    )
    assert prompt["mgpo_weight"] == pytest.approx(1.0)
    result = aggregate_spectrum([prompt], [_rollout(1), _rollout(0)], pass_k=[1, 2], mgpo_enabled=True)
    assert result["mgpo_diagnostics"]["available"] is True
    assert result["mgpo_diagnostics"]["mgpo_weight_mean_effective_prompts"] == pytest.approx(1.0)


def test_original_cli_defaults_to_hf_and_preserves_required_arguments():
    args = parse_args([
        "--config_json", "c.json", "--resume_from", "model",
        "--output_jsonl", "out.jsonl", "--verified_traces_jsonl", "traces.jsonl",
        "--num_rollouts", "16", "--min_correct_rate", ".25", "--max_correct_rate", ".75",
    ])
    assert args.rollout_backend == "hf"
    assert args.num_rollouts == 16
    assert args.assert_multi_actor_dispatch is False


def test_cli_accepts_strict_multi_actor_dispatch_audit_flag():
    args = parse_args([
        "--config_json", "c.json", "--output_jsonl", "out.jsonl",
        "--verified_traces_jsonl", "traces.jsonl", "--rollout_backend", "vllm",
        "--assert_multi_actor_dispatch",
    ])
    assert args.assert_multi_actor_dispatch is True


def test_vllm_prompt_batch_size_controls_static_backend_batching(monkeypatch):
    observed_batch_sizes = []

    def fake_batch(**kwargs):
        prompt_count = len(kwargs["prompts"])
        observed_batch_sizes.append(prompt_count)
        num_rollouts = int(kwargs["num_rollouts"])
        samples = [
            [
                {
                    "text": "",
                    "token_ids": [],
                    "response_token_count": 0,
                    "finish_reason": "stop",
                    "truncated": False,
                }
                for _ in range(num_rollouts)
            ]
            for _ in range(prompt_count)
        ]
        return samples, {
            "dispatch_path": "static_multi_sample_batch",
            "actor_count": 2,
            "active_actor_count": min(2, prompt_count),
            "prompt_count": prompt_count,
            "prompt_batch_size": prompt_count,
            "num_samples": num_rollouts,
            "row_count": prompt_count * num_rollouts,
            "actors": {},
        }

    monkeypatch.setattr(boundary_cli, "_vllm_samples_batch", fake_batch)
    monkeypatch.setattr(
        boundary_cli,
        "build_rl_prompt_text",
        lambda _cfg, _tokenizer, question: f"prompt:{question}",
    )
    cfg = types.SimpleNamespace(
        rl=types.SimpleNamespace(rollout_max_prompt_tokens=0)
    )
    args = types.SimpleNamespace(
        rollout_backend="vllm",
        prompt_batch_size=16,
        num_rollouts=16,
        assert_multi_actor_dispatch=False,
    )
    records = [{"question": f"q{index}"} for index in range(17)]

    yielded = list(
        boundary_cli._iter_prompt_samples(
            cfg=cfg,
            args=args,
            records=records,
            backend=object(),
            model=None,
            tokenizer=object(),
            generation_config=types.SimpleNamespace(),
            seed=100,
        )
    )

    assert observed_batch_sizes == [16, 1]
    assert [item[0] for item in yielded] == list(range(17))
    assert [item[3] for item in yielded] == list(range(100, 117))


def test_static_audit_sigterm_is_converted_to_finally_unwind():
    with pytest.raises(SystemExit) as exc_info:
        boundary_cli._terminate_static_audit(signal.SIGTERM, None)
    assert exc_info.value.code == 128 + int(signal.SIGTERM)


def test_mock_vllm_true_multi_sample_parsing_and_actor_round_robin(tmp_path):
    cfg = make_default_config()
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.vllm_rollout_actors = [
        {"name": "a0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "a1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]
    descriptor = {"policy_version": 0, "policy_fingerprint": "x", "export_dir": "e"}

    class Client:
        is_alive = True
        def __init__(self, tag): self.tag = tag
        def generate(
            self,
            *,
            prompt_token_ids,
            sampling_kwargs_by_prompt,
            expected_policy_descriptor,
        ):
            assert expected_policy_descriptor == descriptor
            assert len(prompt_token_ids) == 1
            return [ActorRequestOutput([
                ActorCompletion([self.tag, index], text=f"r{index}", finish_reason="stop")
                for index in range(sampling_kwargs_by_prompt[0]["n"])
            ])]

    backend = VLLMRolloutBackend(
        fit_cfg=cfg, rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._actors = {"a0": Client(10), "a1": Client(20)}
    backend._engine_policy_descriptor = descriptor
    first = backend.generate_static_samples(
        prompt_token_ids=[1], num_samples=3, max_new_tokens=8,
        temperature=1.0, top_p=.95, seed=123, prompt_index=0,
    )
    second = backend.generate_static_samples(
        prompt_token_ids=[2], num_samples=3, max_new_tokens=8,
        temperature=1.0, top_p=.95, seed=124, prompt_index=1,
    )
    assert [item["token_ids"][0] for item in first] == [10, 10, 10]
    assert [item["token_ids"][0] for item in second] == [20, 20, 20]
    assert [item["text"] for item in first] == ["r0", "r1", "r2"]
    dispatch = backend.last_actor_dispatch_metadata
    assert dispatch["dispatch_path"] == "static_multi_sample_batch"
    assert dispatch["prompt_batch_size"] == 1
    assert dispatch["row_count"] == 3
    assert dispatch["active_actor_count"] == 1
    assert dispatch["actors"]["a0"]["row_count"] == 0
    assert dispatch["actors"]["a1"]["row_count"] == 3


def test_static_backend_is_closed_when_initialization_raises(tmp_path, monkeypatch):
    closed = []

    class Model:
        def eval(self): return self

    load_info = types.SimpleNamespace(
        base_model_path=str(tmp_path / "base"), metadata={"patch_cfg": {"topk": 8}}, patch_cfg={"topk": 8}
    )
    monkeypatch.setattr(boundary_cli, "load_policy_for_rl", lambda *args, **kwargs: (Model(), object(), load_info))

    class Backend:
        def __init__(self, **kwargs): pass
        def sync_policy(self, **kwargs): raise RuntimeError("injected init failure")
        def close(self): closed.append(True)

    monkeypatch.setattr(boundary_cli, "VLLMRolloutBackend", Backend)
    cfg = make_default_config()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    cfg.rl.resume_from = str(checkpoint)
    args = types.SimpleNamespace(rollout_backend="vllm")
    with pytest.raises(RuntimeError, match="injected init failure"):
        boundary_cli._prepare_backend(cfg, args, tmp_path / "out")
    assert closed == [True]


def test_static_backend_is_closed_when_sigterm_unwinds_initialization(tmp_path, monkeypatch):
    closed = []

    class Model:
        def eval(self): return self

    load_info = types.SimpleNamespace(
        base_model_path=str(tmp_path / "base"), metadata={}, patch_cfg={}
    )
    monkeypatch.setattr(
        boundary_cli,
        "load_policy_for_rl",
        lambda *args, **kwargs: (Model(), object(), load_info),
    )

    class Backend:
        def __init__(self, **kwargs): pass
        def sync_policy(self, **kwargs): raise SystemExit(143)
        def close(self): closed.append(True)

    monkeypatch.setattr(boundary_cli, "VLLMRolloutBackend", Backend)
    cfg = make_default_config()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    cfg.rl.resume_from = str(checkpoint)
    args = types.SimpleNamespace(rollout_backend="vllm")
    with pytest.raises(SystemExit) as exc_info:
        boundary_cli._prepare_backend(cfg, args, tmp_path / "out")
    assert exc_info.value.code == 143
    assert closed == [True]


def test_boundary_cli_contains_no_optimizer_backward_or_training_entrypoint():
    source = (ROOT / "cli" / "build_boundary_gsm8k.py").read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "train_grpo" not in source
    assert "train_rl" not in source
