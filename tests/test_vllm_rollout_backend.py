from __future__ import annotations

import sys
import types
import json
import copy
import threading
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg
if "MOTE.rl" not in sys.modules:
    rl_pkg = types.ModuleType("MOTE.rl")
    rl_pkg.__path__ = [str(ROOT / "rl")]
    sys.modules["MOTE.rl"] = rl_pkg

from MOTE.config.defaults import make_default_config
from MOTE.rl.rollout_backends import RolloutGenerationConfig
from MOTE.rl.rollout_backends import RolloutSyncResult
from MOTE.rl.vllm_rollout import VLLMRolloutBackend, assert_multi_actor_dispatch
from MOTE.rl.vllm_sync import VLLMPolicySyncManager
from MOTE.train.rl_controller import build_rollout_attention_and_response_mask


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2


class FakeCompletion:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.text = "unused"


class FakeRequestOutput:
    def __init__(self, token_ids):
        self.outputs = [FakeCompletion(token_ids)]


class FakeMultiRequestOutput:
    def __init__(self, prompt_token: int, completion_count: int):
        self.outputs = [
            FakeCompletion([int(prompt_token), sample_index])
            for sample_index in range(int(completion_count))
        ]


def _cfg(tmp_path):
    cfg = make_default_config()
    cfg.rl.rollout_backend = "vllm"
    cfg.rl.vllm_export_root = str(tmp_path / "exports")
    cfg.rl.max_new_tokens = 4
    cfg.rl.temperature = 0.7
    cfg.rl.top_p = 0.95
    return cfg


@pytest.fixture(autouse=True)
def _mock_sync_vllm_preflight(monkeypatch):
    monkeypatch.setattr(
        "MOTE.diagnostics.vllm_export_preflight.validate_vllm_export_preflight",
        lambda *args, **kwargs: {"ok": True, "errors": [], "warnings": []},
    )


def test_vllm_backend_missing_import_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    backend = VLLMRolloutBackend(
        fit_cfg=_cfg(tmp_path),
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )

    with pytest.raises(ImportError, match="rl.rollout_backend='vllm'"):
        backend._import_vllm()


def test_vllm_token_prompt_required_unless_text_fallback_enabled(tmp_path):
    class FakeLLM:
        def generate(self, *args, **kwargs):
            if "prompt_token_ids" in kwargs or (
                kwargs.get("prompts") and isinstance(kwargs["prompts"][0], dict)
            ):
                raise TypeError("token prompts unsupported")
            return [FakeRequestOutput([9])]

    backend = VLLMRolloutBackend(
        fit_cfg=_cfg(tmp_path),
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend.llm = FakeLLM()
    backend._vllm = object
    backend._sampling_params_cls = lambda **kwargs: kwargs

    with pytest.raises(RuntimeError, match="token-id prompts"):
        backend._generate_token_ids(
            prompts=["a"],
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            sampling_params={},
        )


def test_vllm_text_prompt_fallback_is_explicit(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_allow_text_prompt_fallback = True

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append({"args": args, "kwargs": kwargs})
            if "prompt_token_ids" in kwargs or (
                kwargs.get("prompts") and isinstance(kwargs["prompts"][0], dict)
            ):
                raise TypeError("token prompts unsupported")
            return [FakeRequestOutput([9])]

    llm = FakeLLM()
    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend.llm = llm
    backend._vllm = object
    backend._sampling_params_cls = lambda **kwargs: kwargs

    outputs = backend._generate_token_ids(
        prompts=["prompt text"],
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        sampling_params={},
    )

    assert outputs[0].outputs[0].token_ids == [9]
    assert llm.calls[-1]["args"] == (["prompt text"], {})


def test_vllm_token_prompts_strip_left_padding(tmp_path):
    captured = {}

    class FakeLLM:
        def generate(self, *args, **kwargs):
            captured.update(kwargs)
            return [FakeRequestOutput([9]), FakeRequestOutput([10])]

    backend = VLLMRolloutBackend(
        fit_cfg=_cfg(tmp_path),
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend.llm = FakeLLM()
    backend._vllm = object
    backend._sampling_params_cls = lambda **kwargs: kwargs

    backend._generate_token_ids(
        prompts=["short", "long"],
        input_ids=torch.tensor([[0, 0, 11, 12], [21, 22, 23, 24]], dtype=torch.long),
        attention_mask=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
        sampling_params={},
    )

    assert captured["prompts"] == [
        {"prompt_token_ids": [11, 12]},
        {"prompt_token_ids": [21, 22, 23, 24]},
    ]


def test_mock_vllm_outputs_preserve_expanded_prompt_order_and_masks(tmp_path):
    input_ids = torch.tensor([[10, 11, 0], [10, 11, 0], [20, 21, 22], [20, 21, 22]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.long)
    generated_ids = [[31, 2], [32], [41, 42, 2], [43]]

    sequences, original_lens = VLLMRolloutBackend.build_full_sequences_from_token_outputs(
        input_ids=input_ids,
        generated_token_ids=generated_ids,
        pad_token_id=0,
    )
    full_attention, response_mask = build_rollout_attention_and_response_mask(
        sequences,
        attention_mask,
        response_start=input_ids.shape[1],
        original_seq_lens=original_lens,
        eos_token_id=2,
    )

    assert torch.equal(sequences[0, :5], torch.tensor([10, 11, 0, 31, 2]))
    assert torch.equal(sequences[1, :4], torch.tensor([10, 11, 0, 32]))
    assert torch.equal(sequences[2, :6], torch.tensor([20, 21, 22, 41, 42, 2]))
    assert torch.equal(sequences[3, :4], torch.tensor([20, 21, 22, 43]))
    assert original_lens == [5, 4, 6, 4]
    assert torch.equal(full_attention[:, : input_ids.shape[1]], attention_mask)
    assert response_mask[0, 3:5].sum().item() == 2.0
    assert response_mask[1, 3:4].sum().item() == 1.0
    assert response_mask[2, 3:6].sum().item() == 3.0
    assert response_mask[3, 3:4].sum().item() == 1.0


def test_stage5c_multi_actor_sharding_uses_all_actors_and_restores_order(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.group_size = 4
    cfg.rl.vllm_rollout_actors = [
        {"name": "r0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "r1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]
    descriptor = {"policy_version": 3, "policy_fingerprint": "abc", "export_dir": "x"}

    class FakeClient:
        is_alive = True

        def __init__(self, name):
            self.name = name

        def generate(self, *, prompt_token_ids, sampling_kwargs, expected_policy_descriptor):
            assert expected_policy_descriptor == descriptor
            return [FakeRequestOutput([ids[-1] + 100]) for ids in prompt_token_ids]

        def ping(self):
            return {"policy_descriptor": descriptor, "actor_resources": {"cuda_device_count": 1}}

    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._actors = {"r0": FakeClient("r0"), "r1": FakeClient("r1")}
    backend._actor_resource_snapshot = {"r0": {}, "r1": {}}
    backend._rollout_request_sequence = 1
    prompt_ids = [[index] for index in range(8)]

    outputs = backend._generate_subprocess_actors(
        prompt_token_ids=prompt_ids,
        sampling_kwargs={"max_tokens": 4},
        expected_policy_descriptor=descriptor,
    )

    assert [output.outputs[0].token_ids[0] for output in outputs] == list(range(100, 108))
    dispatch = backend._last_actor_dispatch_metadata
    assert dispatch["active_actor_count"] == 2
    assert sorted(
        index
        for actor in dispatch["actors"].values()
        for index in actor["row_indices"]
    ) == list(range(8))


def test_multi_actor_audit_256_rows_is_balanced_concurrent_and_ordered(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.group_size = 16
    cfg.rl.vllm_actor_resource_log_every = 0
    cfg.rl.vllm_rollout_actors = [
        {"name": "rollout_0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "rollout_1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]
    descriptor = {"policy_version": 3, "policy_fingerprint": "abc", "export_dir": "x"}
    barrier = threading.Barrier(2)

    class FakeClient:
        is_alive = True

        def __init__(self, device):
            self.device = device
            self.last_generate_info = {}

        def generate(self, *, prompt_token_ids, sampling_kwargs, expected_policy_descriptor):
            assert expected_policy_descriptor == descriptor
            barrier.wait(timeout=2.0)
            actor_start = time.perf_counter()
            time.sleep(0.05)
            actor_end = time.perf_counter()
            self.last_generate_info = {
                "llm_generate_called": True,
                "generate_start_time": actor_start,
                "generate_end_time": actor_end,
                "generate_sec": actor_end - actor_start,
                "sampling_kwargs": dict(sampling_kwargs),
                "engine_config": {"model": "mock", "tokenizer": "mock", "dtype": "bfloat16"},
                "actor_resources": {
                    "cuda_visible_devices": self.device,
                    "process_id": 100 + int(self.device),
                },
                "engine_resources_at_load": {
                    "cuda_device_count": 1,
                    "cuda_current_device": 0,
                    "enginecore_process_ids": [200 + int(self.device)],
                },
            }
            return [FakeRequestOutput([ids[-1] + 1000]) for ids in prompt_token_ids]

    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._actors = {
        "rollout_0": FakeClient("1"),
        "rollout_1": FakeClient("2"),
    }
    backend._actor_resource_snapshot = {"rollout_0": {}, "rollout_1": {}}
    prompt_ids = [[index] for index in range(256)]

    outputs = backend._generate_subprocess_actors(
        prompt_token_ids=prompt_ids,
        sampling_kwargs={"max_tokens": 8, "temperature": 0.7, "top_p": 0.95},
        expected_policy_descriptor=descriptor,
    )

    assert [output.outputs[0].token_ids[0] for output in outputs] == list(range(1000, 1256))
    dispatch = backend.last_actor_dispatch_metadata
    assert dispatch["row_count"] == 256
    assert dispatch["assigned_row_count"] == 256
    assert dispatch["prompt_batch_size"] == 16
    assert dispatch["num_rollouts"] == 16
    assert dispatch["generate_intervals_overlap"] is True
    assert dispatch["generate_overlap_sec"] > 0.0
    assert {name: item["row_count"] for name, item in dispatch["actors"].items()} == {
        "rollout_0": 128,
        "rollout_1": 128,
    }
    assert all(item["llm_generate_called"] for item in dispatch["actors"].values())
    assert all(item["total_generated_tokens"] == 128 for item in dispatch["actors"].values())
    assert_multi_actor_dispatch(dispatch, expected_row_count=256)


def _static_batch_backend(tmp_path, *, barrier=None, fail_device=None, bad_completion_prompt=None):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.vllm_rollout_actors = [
        {"name": "rollout_0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "rollout_1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]
    descriptor = {"policy_version": 0, "policy_fingerprint": "static", "export_dir": "e"}
    calls = {}

    class StaticClient:
        is_alive = True

        def __init__(self, name, device):
            self.name = name
            self.device = device
            self.last_generate_info = {}
            self.close_count = 0

        def generate(
            self,
            *,
            prompt_token_ids,
            sampling_kwargs_by_prompt,
            expected_policy_descriptor,
        ):
            if self.device == fail_device:
                raise RuntimeError(f"injected actor failure on {self.device}")
            assert expected_policy_descriptor == descriptor
            assert len(prompt_token_ids) == len(sampling_kwargs_by_prompt)
            if barrier is not None:
                barrier.wait(timeout=2.0)
            actor_start = time.perf_counter()
            time.sleep(0.03)
            actor_end = time.perf_counter()
            calls[self.name] = {
                "prompt_token_ids": [list(ids) for ids in prompt_token_ids],
                "sampling_kwargs_by_prompt": [
                    dict(item) for item in sampling_kwargs_by_prompt
                ],
            }
            self.last_generate_info = {
                "llm_generate_called": True,
                "generate_start_time": actor_start,
                "generate_end_time": actor_end,
                "generate_sec": actor_end - actor_start,
                "sampling_kwargs_by_prompt": [
                    dict(item) for item in sampling_kwargs_by_prompt
                ],
                "engine_config": {
                    "model": "mock",
                    "tokenizer": "mock",
                    "dtype": "bfloat16",
                    "max_model_len": 1024,
                    "max_num_seqs": 256,
                    "model_impl": "transformers",
                },
                "actor_resources": {
                    "cuda_visible_devices": self.device,
                    "process_id": 100 + int(self.device),
                },
                "engine_resources_at_load": {
                    "cuda_device_count": 1,
                    "cuda_current_device": 0,
                    "enginecore_process_ids": [200 + int(self.device)],
                },
            }
            outputs = []
            for ids, params in zip(prompt_token_ids, sampling_kwargs_by_prompt):
                count = int(params["n"])
                if bad_completion_prompt is not None and ids[-1] == bad_completion_prompt:
                    count -= 1
                outputs.append(FakeMultiRequestOutput(ids[-1], count))
            return outputs

        def close(self):
            self.close_count += 1
            return {
                "closed": True,
                "actor_pid": 100 + int(self.device),
                "enginecore_pids": [200 + int(self.device)],
            }

    clients = {
        "rollout_0": StaticClient("rollout_0", "1"),
        "rollout_1": StaticClient("rollout_1", "2"),
    }
    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._actors = clients
    backend._engine_policy_descriptor = descriptor
    backend._actor_resource_snapshot = {
        name: {
            "engine": {
                "actor_resources": {
                    "cuda_visible_devices": client.device,
                    "process_id": 100 + int(client.device),
                    "cuda_device_count": 1,
                    "cuda_current_device": 0,
                    "enginecore_process_ids": [200 + int(client.device)],
                },
                "engine_config": {
                    "model": "mock",
                    "tokenizer": "mock",
                    "dtype": "bfloat16",
                },
            }
        }
        for name, client in clients.items()
    }
    return backend, clients, calls


def test_static_batch_16_prompts_16_samples_balances_concurrently_and_restores_order(tmp_path):
    backend, _clients, calls = _static_batch_backend(
        tmp_path, barrier=threading.Barrier(2)
    )

    results = backend.generate_static_samples_batch(
        prompt_token_ids=[[index] for index in range(16)],
        num_samples=16,
        max_new_tokens=8,
        temperature=0.7,
        top_p=0.95,
        seeds=[1000 + index for index in range(16)],
        prompt_indices=list(range(16)),
        eos_token_id=2,
    )

    assert len(results) == 16
    assert all(len(completions) == 16 for completions in results)
    assert [
        results[prompt_index][sample_index]["token_ids"]
        for prompt_index in range(16)
        for sample_index in range(16)
    ] == [
        [prompt_index, sample_index]
        for prompt_index in range(16)
        for sample_index in range(16)
    ]
    assert [ids[-1] for ids in calls["rollout_0"]["prompt_token_ids"]] == list(range(0, 16, 2))
    assert [ids[-1] for ids in calls["rollout_1"]["prompt_token_ids"]] == list(range(1, 16, 2))
    assert [
        item["seed"] for item in calls["rollout_0"]["sampling_kwargs_by_prompt"]
    ] == [1000 + index for index in range(0, 16, 2)]
    assert [
        item["seed"] for item in calls["rollout_1"]["sampling_kwargs_by_prompt"]
    ] == [1000 + index for index in range(1, 16, 2)]
    dispatch = backend.last_actor_dispatch_metadata
    assert dispatch["dispatch_path"] == "static_multi_sample_batch"
    assert dispatch["prompt_count"] == 16
    assert dispatch["completion_count"] == 256
    assert dispatch["generate_intervals_overlap"] is True
    assert dispatch["generate_overlap_sec"] > 0.0
    assert {
        name: (item["prompt_count"], item["completion_count"])
        for name, item in dispatch["actors"].items()
    } == {"rollout_0": (8, 128), "rollout_1": (8, 128)}
    assert_multi_actor_dispatch(
        dispatch,
        expected_row_count=256,
        expected_prompt_count=16,
        expected_num_samples=16,
    )


def test_static_batch_16_prompts_true_greedy_balances_8_8_concurrently(tmp_path):
    backend, _clients, calls = _static_batch_backend(
        tmp_path, barrier=threading.Barrier(2)
    )
    results = backend.generate_static_samples_batch(
        prompt_token_ids=[[index] for index in range(16)],
        num_samples=1,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        seeds=[1000 + index for index in range(16)],
        prompt_indices=list(range(16)),
        decoding="greedy",
        stop=["Question:"],
    )
    assert len(results) == 16
    assert all(len(completions) == 1 for completions in results)
    for call in calls.values():
        assert len(call["prompt_token_ids"]) == 8
        assert all(item["n"] == 1 for item in call["sampling_kwargs_by_prompt"])
        assert all(item["temperature"] == 0.0 for item in call["sampling_kwargs_by_prompt"])
        assert all("top_p" not in item for item in call["sampling_kwargs_by_prompt"])
        assert all(item["stop"] == ["Question:"] for item in call["sampling_kwargs_by_prompt"])
    dispatch = backend.last_actor_dispatch_metadata
    assert dispatch["decoding"] == "greedy"
    assert dispatch["stochastic_sampling"] is False
    assert dispatch["generate_intervals_overlap"] is True
    assert {name: item["row_count"] for name, item in dispatch["actors"].items()} == {
        "rollout_0": 8,
        "rollout_1": 8,
    }


def test_static_batch_rejects_multi_sample_greedy(tmp_path):
    backend, _clients, _calls = _static_batch_backend(tmp_path)
    with pytest.raises(ValueError, match="num_samples=1"):
        backend.generate_static_samples_batch(
            prompt_token_ids=[[0]], num_samples=2, max_new_tokens=8,
            temperature=0.7, top_p=0.95, seeds=[1], decoding="greedy",
        )


def test_static_batch_17_prompts_handles_non_divisible_actor_split(tmp_path):
    backend, _clients, _calls = _static_batch_backend(tmp_path)
    results = backend.generate_static_samples_batch(
        prompt_token_ids=[[index] for index in range(17)],
        num_samples=16,
        max_new_tokens=4,
        temperature=1.0,
        top_p=0.9,
        seeds=[2000 + index for index in range(17)],
        prompt_indices=list(range(17)),
    )
    assert len(results) == 17
    assert all(len(completions) == 16 for completions in results)
    assert {
        name: item["prompt_count"]
        for name, item in backend.last_actor_dispatch_metadata["actors"].items()
    } == {"rollout_0": 9, "rollout_1": 8}


def test_static_batch_single_prompt_allows_one_active_actor_and_single_api_compatibility(tmp_path):
    backend, _clients, _calls = _static_batch_backend(tmp_path)
    results = backend.generate_static_samples(
        prompt_token_ids=[77],
        num_samples=16,
        max_new_tokens=4,
        temperature=1.0,
        top_p=0.9,
        seed=3007,
        prompt_index=7,
    )
    assert len(results) == 16
    assert all(item["token_ids"][0] == 77 for item in results)
    dispatch = backend.last_actor_dispatch_metadata
    assert dispatch["active_actor_count"] == 1
    assert dispatch["actors"]["rollout_0"]["prompt_count"] == 0
    assert dispatch["actors"]["rollout_1"]["prompt_count"] == 1
    assert_multi_actor_dispatch(
        dispatch,
        expected_row_count=16,
        expected_prompt_count=1,
        expected_num_samples=16,
    )


def test_static_batch_strict_audit_rejects_one_active_actor_for_large_batch(tmp_path):
    backend, _clients, _calls = _static_batch_backend(tmp_path)
    backend.generate_static_samples_batch(
        prompt_token_ids=[[index] for index in range(16)],
        num_samples=16,
        max_new_tokens=4,
        temperature=1.0,
        top_p=0.9,
        seeds=[4000 + index for index in range(16)],
        prompt_indices=list(range(16)),
    )
    dispatch = copy.deepcopy(backend.last_actor_dispatch_metadata)
    first = dispatch["actors"]["rollout_0"]
    second = dispatch["actors"]["rollout_1"]
    first["prompt_count"] = 16
    first["prompt_group_count"] = 16
    first["row_count"] = 256
    first["completion_count"] = 256
    second["prompt_count"] = 0
    second["prompt_group_count"] = 0
    second["row_count"] = 0
    second["completion_count"] = 0
    dispatch["active_actor_count"] = 1
    with pytest.raises(RuntimeError, match="expected active actors=2.*zero rows"):
        assert_multi_actor_dispatch(
            dispatch,
            expected_row_count=256,
            expected_prompt_count=16,
            expected_num_samples=16,
        )


def test_static_batch_rejects_wrong_completion_count_and_closes_all_actors(tmp_path):
    backend, clients, _calls = _static_batch_backend(
        tmp_path, bad_completion_prompt=3
    )
    with pytest.raises(RuntimeError, match="returned 15 completions"):
        backend.generate_static_samples_batch(
            prompt_token_ids=[[index] for index in range(16)],
            num_samples=16,
            max_new_tokens=4,
            temperature=1.0,
            top_p=0.9,
            seeds=[5000 + index for index in range(16)],
            prompt_indices=list(range(16)),
        )
    assert all(client.close_count == 1 for client in clients.values())


def test_static_batch_actor_exception_closes_other_actor_and_enginecore_path(tmp_path):
    backend, clients, _calls = _static_batch_backend(tmp_path, fail_device="2")
    with pytest.raises(RuntimeError, match="injected actor failure"):
        backend.generate_static_samples_batch(
            prompt_token_ids=[[index] for index in range(16)],
            num_samples=16,
            max_new_tokens=4,
            temperature=1.0,
            top_p=0.9,
            seeds=[6000 + index for index in range(16)],
            prompt_indices=list(range(16)),
        )
    assert clients["rollout_0"].close_count == 1
    assert clients["rollout_1"].close_count == 1


def test_multi_actor_audit_rejects_configured_actor_with_zero_rows():
    dispatch = {
        "actor_count": 2,
        "active_actor_count": 1,
        "row_count": 256,
        "actors": {
            "rollout_0": {
                "row_count": 256,
                "configured_cuda_visible_devices": ["1"],
                "observed_cuda_visible_devices": "1",
            },
            "rollout_1": {
                "row_count": 0,
                "configured_cuda_visible_devices": ["2"],
                "observed_cuda_visible_devices": "2",
            },
        },
    }
    with pytest.raises(RuntimeError, match="active_actor_count=1.*zero rows"):
        assert_multi_actor_dispatch(dispatch, expected_row_count=256)


def test_multi_actor_audit_rejects_row_count_sum_mismatch():
    dispatch = {
        "actor_count": 2,
        "active_actor_count": 2,
        "row_count": 256,
        "actors": {
            "rollout_0": {
                "row_count": 128,
                "configured_cuda_visible_devices": ["1"],
                "observed_cuda_visible_devices": "1",
                "generate_start_time": 10.0,
                "generate_end_time": 20.0,
            },
            "rollout_1": {
                "row_count": 127,
                "configured_cuda_visible_devices": ["2"],
                "observed_cuda_visible_devices": "2",
                "generate_start_time": 10.1,
                "generate_end_time": 20.1,
            },
        },
    }
    with pytest.raises(RuntimeError, match="row_count sum=255"):
        assert_multi_actor_dispatch(dispatch, expected_row_count=256)


def test_multi_actor_audit_rejects_serial_generate_intervals():
    dispatch = {
        "actor_count": 2,
        "active_actor_count": 2,
        "row_count": 256,
        "actors": {
            "rollout_0": {
                "row_count": 128,
                "configured_cuda_visible_devices": ["1"],
                "observed_cuda_visible_devices": "1",
                "generate_start_time": 100.0,
                "generate_end_time": 180.0,
            },
            "rollout_1": {
                "row_count": 128,
                "configured_cuda_visible_devices": ["2"],
                "observed_cuda_visible_devices": "2",
                "generate_start_time": 180.0,
                "generate_end_time": 260.0,
            },
        },
    }
    with pytest.raises(RuntimeError, match="serial execution"):
        assert_multi_actor_dispatch(dispatch, expected_row_count=256)


def test_stage5c_multi_actor_failure_invalidates_entire_batch(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.group_size = 2
    cfg.rl.vllm_rollout_actors = [
        {"name": "ok", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "bad", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]

    class FakeClient:
        is_alive = True

        def __init__(self, fail=False):
            self.fail = fail

        def generate(self, **kwargs):
            if self.fail:
                raise RuntimeError("injected failure")
            return [FakeRequestOutput([9]) for _ in kwargs["prompt_token_ids"]]

        def ping(self):
            return {}

    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._actors = {"ok": FakeClient(), "bad": FakeClient(fail=True)}
    backend._actor_resource_snapshot = {"ok": {}, "bad": {}}
    with pytest.raises(RuntimeError, match="entire rollout batch is invalid"):
        backend._generate_subprocess_actors(
            prompt_token_ids=[[1], [2]],
            sampling_kwargs={},
            expected_policy_descriptor={},
        )


def test_stage5c_parallel_engine_build_enforces_policy_barrier(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.vllm_rollout_actors = [
        {"name": "r0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "r1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]
    descriptor = {"policy_version": 4, "policy_fingerprint": "fp", "export_dir": "export"}

    class FakeClient:
        is_alive = True

        def __init__(self):
            self.startup_info = {"kind": "ready"}
            self.last_engine_info = {}
            self.loaded_descriptor = None
            self.loaded_kwargs = None

        def load_engine(self, kwargs, *, policy_descriptor):
            self.loaded_kwargs = dict(kwargs)
            self.loaded_descriptor = dict(policy_descriptor)
            tp = int(kwargs["tensor_parallel_size"])
            self.last_engine_info = {
                "engine_topology": {
                    "observed_tensor_parallel_size": tp,
                    "tensor_parallel_verified": True,
                },
                "actor_resources": {"cuda_device_count": tp},
            }
            return 0.01

        def ping(self):
            return {"policy_descriptor": self.loaded_descriptor}

        def close(self):
            return None

    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._actors = {"r0": FakeClient(), "r1": FakeClient()}
    elapsed = backend._build_engine("export", policy_descriptor=descriptor)
    assert elapsed >= 0.0
    assert set(backend._actor_resource_snapshot) == {"r0", "r1"}
    assert all(item["policy_verified"] for item in backend._actor_resource_snapshot.values())
    assert backend._actors["r0"].loaded_kwargs["seed"] != backend._actors["r1"].loaded_kwargs["seed"]
    assert "device" not in backend._actors["r0"].loaded_kwargs
    assert "device" not in backend._actors["r1"].loaded_kwargs


def test_subprocess_backend_close_surfaces_actor_failure_and_keeps_failed_client(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.vllm_rollout_actors = [
        {"name": "r0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "r1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]

    class FakeClient:
        def __init__(self, error=None):
            self.error = error

        def close(self):
            if self.error is not None:
                raise self.error
            return {"closed": True}

    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    failed = FakeClient(RuntimeError("engine core survived"))
    backend._actors = {"r0": FakeClient(), "r1": failed}

    with pytest.raises(RuntimeError, match="engine core survived"):
        backend.close()

    assert backend._actors == {"r1": failed}


def test_subprocess_backend_close_returns_actor_reports(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.vllm_rollout_actors = [
        {"name": "r0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
    ]

    class FakeClient:
        def close(self):
            return {"closed": True, "actor_pid": 123}

    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._actors = {"r0": FakeClient()}

    report = backend.close()

    assert report["closed"] is True
    assert report["actors"]["r0"]["actor_pid"] == 123
    assert backend._actors == {}


def test_subprocess_backend_closes_receivers_before_trainer_nccl_groups(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.vllm_rollout_actors = [
        {"name": "r0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "r1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]
    events = []

    class FakeClient:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(f"actor_close:{self.name}")
            return {"closed": True}

    class FakeSyncManager:
        def mark_engines_discarded(self):
            events.append("trainer_groups_release")
            return {"ok": True, "group_count": 2, "groups": {}}

    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._actors = {"r0": FakeClient("r0"), "r1": FakeClient("r1")}
    backend.sync_manager = FakeSyncManager()

    report = backend.close()

    assert set(events[:-1]) == {"actor_close:r0", "actor_close:r1"}
    assert events[-1] == "trainer_groups_release"
    assert report["trainer_nccl_groups"]["group_count"] == 2


def test_native_sync_uses_commit_receipts_without_post_commit_ping(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.vllm_rollout_actors = [
        {"name": "r0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "r1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]
    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )

    class FakeClient:
        is_alive = True

        def ping(self):
            raise AssertionError("native sync must not issue a redundant post-commit ping")

    backend._actors = {"r0": FakeClient(), "r1": FakeClient()}
    backend._actor_resource_snapshot = {"r0": {}, "r1": {}}
    descriptor = {
        "policy_version": 1,
        "policy_fingerprint": "fp",
        "export_dir": "export",
        "weight_transfer_scope": "trainable_patch",
        "weight_transfer_plan_fingerprint": "plan-fp",
    }
    actor_results = {
        name: {
            "committed": True,
            "commit": {
                "committed": True,
                "actor_status": "READY",
                "policy_descriptor": descriptor,
            },
        }
        for name in ("r0", "r1")
    }
    backend.sync_manager.sync = lambda **_kwargs: RolloutSyncResult(
        synced=True,
        policy_version=1,
        policy_lag_updates=0,
        export_dir="export",
        metadata={
            "vllm_weight_transfer_native_sync": True,
            "vllm_policy_fingerprint": "fp",
            "vllm_weight_transfer_scope": "trainable_patch",
            "weight_transfer_plan_fingerprint": "plan-fp",
            "weight_transfer_actor_results": actor_results,
        },
    )

    result = backend.sync_policy(model=None, tokenizer=None, update_step=1)

    assert result.synced is True
    assert result.metadata["vllm_all_actors_policy_verified"] is True
    assert all(
        snapshot["latest"]["source"] == "native_sync_commit_receipt"
        for snapshot in backend._actor_resource_snapshot.values()
    )


def test_post_native_sync_generation_uses_complete_patch_descriptor(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.group_size = 2
    cfg.rl.vllm_actor_resource_log_every = 0
    cfg.rl.vllm_rollout_actors = [
        {"name": "r0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "r1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
    ]
    descriptor = {
        "policy_version": 1,
        "policy_fingerprint": "fp-1",
        "export_dir": "export-u0",
        "weight_transfer_scope": "trainable_patch",
        "weight_transfer_plan_fingerprint": "plan-fp",
    }
    observed = []

    class FakeClient:
        is_alive = True

        def generate(self, *, prompt_token_ids, sampling_kwargs, expected_policy_descriptor):
            observed.append(dict(expected_policy_descriptor))
            return [FakeRequestOutput([9]) for _ in prompt_token_ids]

    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._actors = {"r0": FakeClient(), "r1": FakeClient()}
    backend._actor_resource_snapshot = {
        "r0": {"policy_verified": True},
        "r1": {"policy_verified": True},
    }
    backend._engine_policy_descriptor = dict(descriptor)
    backend.sync_manager.policy_version = 1
    backend.sync_manager.export_dir = Path("export-u0")
    backend._last_sync = RolloutSyncResult(
        synced=True,
        policy_version=1,
        policy_lag_updates=0,
        export_dir="export-u0",
        metadata={"vllm_policy_fingerprint": "fp-1"},
    )

    result = backend.generate(
        model=None,
        tokenizer=FakeTokenizer(),
        prompts=["a", "b"],
        input_ids=torch.tensor([[1, 2], [3, 4]]),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        generation_config=RolloutGenerationConfig(max_new_tokens=1, temperature=0.0, top_p=1.0),
        update_step=1,
    )

    assert observed == [descriptor, descriptor]
    assert result.metadata["vllm_policy_version"] == 1
    assert result.metadata["vllm_policy_fingerprint"] == "fp-1"


def test_in_process_vllm_019_retries_without_legacy_device(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_device = "cuda:0"
    calls = []

    class StrictV019LLM:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            if "device" in kwargs:
                raise TypeError("EngineArgs.__init__() got an unexpected keyword argument 'device'")

    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    backend._vllm = StrictV019LLM
    backend._sampling_params_cls = object

    backend._build_engine("export")

    assert len(calls) == 2
    assert calls[0]["device"] == "cuda:0"
    assert "device" not in calls[1]


def test_stale_vllm_policy_raises_unless_allowed(tmp_path):
    cfg = _cfg(tmp_path)
    manager = VLLMPolicySyncManager(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    manager.policy_version = 1

    with pytest.raises(RuntimeError, match="policy is stale"):
        manager.assert_fresh_or_allowed(2)

    cfg.rl.allow_stale_vllm_policy = True
    assert manager.assert_fresh_or_allowed(3) == 2


def test_sync_due_tracks_optimizer_update_versions_not_microsteps(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_every_updates = 1
    manager = VLLMPolicySyncManager(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )
    manager.policy_version = 0

    assert manager.sync_due(0) is False
    assert manager.sync_due(1) is True


def test_export_reload_sync_calls_stage4c_hooks(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = {"save": [], "export": [], "layout": [], "roundtrip": []}

    def save_policy(output_dir, update_step, checkpoint_name, extra):
        calls["save"].append((Path(output_dir), update_step, checkpoint_name, dict(extra)))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir)

    def fake_export(checkpoint_dir, output_dir, **kwargs):
        calls["export"].append((Path(checkpoint_dir), Path(output_dir), dict(kwargs)))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return types.SimpleNamespace(output_dir=Path(output_dir))

    def fake_layout(path):
        calls["layout"].append(Path(path))
        return types.SimpleNamespace(ok=True, errors=[], warnings=[])

    def fake_roundtrip(path, **kwargs):
        calls["roundtrip"].append((Path(path), dict(kwargs)))
        return types.SimpleNamespace(ok=True, errors=[], warnings=[])

    monkeypatch.setattr("MOTE.export.hf_export.export_fitmotn_hf_roundtrip", fake_export)
    monkeypatch.setattr("MOTE.export.validate.validate_export_layout", fake_layout)
    monkeypatch.setattr("MOTE.export.roundtrip.validate_hf_roundtrip", fake_roundtrip)

    manager = VLLMPolicySyncManager(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=save_policy,
    )
    manager.weight_transfer_initialized = True
    result = manager.sync(model=None, tokenizer=None, update_step=1, force=True)

    assert result.synced is True
    assert result.policy_version == 1
    assert result.policy_lag_updates == 0
    assert manager.weight_transfer_initialized is False
    assert calls["save"]
    assert calls["export"][0][0].name.startswith(".raw-policy-u1-")
    assert ".tmp-" in calls["export"][0][0].name
    assert calls["export"][0][1].name.startswith(".hf-policy-u1-")
    assert result.export_dir is not None
    assert Path(result.export_dir).name.startswith("hf-policy-u1-")
    assert calls["layout"][0].name.startswith(".hf-policy-u1-")
    assert calls["roundtrip"][0][0].name.startswith(".hf-policy-u1-")
    assert result.metadata["vllm_export_transaction_committed"] is True
    assert result.metadata["vllm_export_reused"] is False


def test_export_reload_reuses_committed_matching_policy(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = {"save": 0, "export": 0}

    def save_policy(output_dir, update_step, checkpoint_name, extra):
        calls["save"] += 1
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir)

    def fake_export(checkpoint_dir, output_dir, **kwargs):
        calls["export"] += 1
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return types.SimpleNamespace(output_dir=Path(output_dir))

    monkeypatch.setattr("MOTE.export.hf_export.export_fitmotn_hf_roundtrip", fake_export)
    monkeypatch.setattr(
        "MOTE.export.validate.validate_export_layout",
        lambda path: types.SimpleNamespace(ok=True, errors=[], warnings=[]),
    )
    monkeypatch.setattr(
        "MOTE.export.roundtrip.validate_hf_roundtrip",
        lambda path, **kwargs: types.SimpleNamespace(ok=True, errors=[], warnings=[]),
    )
    model = torch.nn.Linear(2, 2)
    first = VLLMPolicySyncManager(
        fit_cfg=cfg, rl_dir=tmp_path, save_policy_checkpoint=save_policy
    ).sync(model=model, tokenizer=None, update_step=3, force=True)
    second = VLLMPolicySyncManager(
        fit_cfg=cfg, rl_dir=tmp_path, save_policy_checkpoint=save_policy
    ).sync(model=model, tokenizer=None, update_step=3, force=True)

    assert calls == {"save": 1, "export": 1}
    assert first.export_dir == second.export_dir
    assert second.metadata["vllm_export_reused"] is True
    assert second.metadata["vllm_checkpoint_save_sec"] == 0.0
    assert (Path(second.export_dir) / "fitmotn_vllm_sync_manifest.json").is_file()
    assert (Path(cfg.rl.vllm_export_root) / "fitmotn_vllm_sync_state.json").is_file()


def test_export_reload_failure_removes_transaction_directories(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)

    def save_policy(output_dir, update_step, checkpoint_name, extra):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir)

    def fake_export(checkpoint_dir, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True)
        return types.SimpleNamespace(output_dir=Path(output_dir))

    monkeypatch.setattr("MOTE.export.hf_export.export_fitmotn_hf_roundtrip", fake_export)
    monkeypatch.setattr(
        "MOTE.export.validate.validate_export_layout",
        lambda path: types.SimpleNamespace(ok=False, errors=["broken"], warnings=[]),
    )
    manager = VLLMPolicySyncManager(
        fit_cfg=cfg, rl_dir=tmp_path, save_policy_checkpoint=save_policy
    )
    with pytest.raises(RuntimeError, match="layout validation failed"):
        manager.sync(model=torch.nn.Linear(2, 2), tokenizer=None, update_step=1, force=True)

    root = Path(cfg.rl.vllm_export_root)
    assert list(root.glob(".*policy-*.tmp-*")) == []
    assert list(root.glob("hf-policy-u*")) == []


def test_export_retention_prunes_raw_and_hf_as_manifest_pairs(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_keep_sync_exports = 1

    def save_policy(output_dir, update_step, checkpoint_name, extra):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir)

    def fake_export(checkpoint_dir, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True)
        return types.SimpleNamespace(output_dir=Path(output_dir))

    monkeypatch.setattr("MOTE.export.hf_export.export_fitmotn_hf_roundtrip", fake_export)
    monkeypatch.setattr(
        "MOTE.export.validate.validate_export_layout",
        lambda path: types.SimpleNamespace(ok=True, errors=[], warnings=[]),
    )
    monkeypatch.setattr(
        "MOTE.export.roundtrip.validate_hf_roundtrip",
        lambda path, **kwargs: types.SimpleNamespace(ok=True, errors=[], warnings=[]),
    )
    manager = VLLMPolicySyncManager(
        fit_cfg=cfg, rl_dir=tmp_path, save_policy_checkpoint=save_policy
    )
    model = torch.nn.Linear(2, 2)
    manager.sync(model=model, tokenizer=None, update_step=1, force=True)
    with torch.no_grad():
        model.weight.add_(1.0)
    latest = manager.sync(model=model, tokenizer=None, update_step=2, force=True)

    root = Path(cfg.rl.vllm_export_root)
    exports = list(root.glob("hf-policy-u*"))
    raws = list(root.glob("raw-policy-u*"))
    assert exports == [Path(latest.export_dir)]
    assert len(raws) == 1
    manifest = json.loads(
        (exports[0] / "fitmotn_vllm_sync_manifest.json").read_text(encoding="utf-8")
    )
    assert raws[0].name == manifest["raw_checkpoint_name"]


def test_rollout_metadata_has_request_and_input_provenance(tmp_path):
    cfg = _cfg(tmp_path)
    backend = VLLMRolloutBackend(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: output_dir,
    )

    class FakeLLM:
        def generate(self, *args, **kwargs):
            return [FakeRequestOutput([9]), FakeRequestOutput([10])]

    export_dir = str(tmp_path / "hf-policy-u2-test")
    descriptor = {"policy_version": 2, "policy_fingerprint": "abc", "export_dir": export_dir}
    backend.llm = FakeLLM()
    backend._vllm = object
    backend._sampling_params_cls = lambda **kwargs: kwargs
    backend._engine_policy_descriptor = descriptor
    backend.sync_manager.policy_version = 2
    backend.sync_manager.export_dir = Path(export_dir)
    backend._last_sync = RolloutSyncResult(
        synced=True,
        policy_version=2,
        policy_lag_updates=0,
        export_dir=export_dir,
        metadata={"vllm_policy_fingerprint": "abc"},
    )
    kwargs = dict(
        model=None,
        tokenizer=FakeTokenizer(),
        prompts=["a", "b"],
        input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        generation_config=RolloutGenerationConfig(max_new_tokens=4, temperature=0.7, top_p=0.95),
        update_step=2,
    )
    first = backend.generate(**kwargs)
    second = backend.generate(**kwargs)

    assert first.metadata["rollout_request_id"] != second.metadata["rollout_request_id"]
    assert first.metadata["rollout_prompt_fingerprint"] == second.metadata["rollout_prompt_fingerprint"]
    assert first.metadata["rollout_sampling_fingerprint"] == second.metadata["rollout_sampling_fingerprint"]
    assert first.metadata["vllm_policy_fingerprint"] == "abc"
    assert first.metadata["vllm_engine_policy_verified"] is True
