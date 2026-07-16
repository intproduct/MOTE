from __future__ import annotations

import sys
import types
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg

from MOTE.rl.vllm_actor import _engine_topology_snapshot, _handle_actor_request
from MOTE.config.loader import load_config_from_json


class FakeCompletion:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.text = "fake"


class FakeOutput:
    def __init__(self, token_ids):
        self.outputs = [FakeCompletion(token_ids)]


def test_actor_reads_runtime_tensor_parallel_topology():
    parallel = types.SimpleNamespace(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    llm = types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(parallel_config=parallel)
        )
    )
    result = _engine_topology_snapshot(llm, {"tensor_parallel_size": 2, "device": "cuda:0"})
    assert result["observed_tensor_parallel_size"] == 2
    assert result["tensor_parallel_verified"] is True
    assert result["source"] == "llm.llm_engine.vllm_config.parallel_config"


def test_actor_protocol_load_generate_unload(monkeypatch):
    calls = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["llm_kwargs"] = kwargs

        def generate(self, *args, **kwargs):
            calls["generate_kwargs"] = kwargs
            prompts = kwargs.get("prompts") or []
            return [FakeOutput([100 + idx]) for idx, _ in enumerate(prompts)]

        def shutdown(self):
            calls["shutdown"] = calls.get("shutdown", 0) + 1

        def sleep(self, level=1):
            calls["sleep_level"] = level

        def wake_up(self, tags=None):
            calls["wake_tags"] = tags

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    state = {"llm": None, "closed": False}
    ping = _handle_actor_request(state, {"command": "ping"})
    assert ping["engine_loaded"] is False
    assert "cuda_visible_devices" in ping["actor_resources"]
    _handle_actor_request(state, {"command": "load_engine", "llm_kwargs": {"model": "export"}})
    assert calls["llm_kwargs"]["model"] == "export"
    assert state["engine_topology"]["requested_tensor_parallel_size"] == 1
    result = _handle_actor_request(
        state,
        {
            "command": "generate",
            "prompt_token_ids": [[1, 2], [3]],
            "sampling_kwargs": {"max_tokens": 4, "temperature": 0.0},
        },
    )
    assert result["outputs"][0][0]["token_ids"] == [100]
    assert result["outputs"][1][0]["token_ids"] == [101]
    assert calls["generate_kwargs"]["prompts"] == [
        {"prompt_token_ids": [1, 2]},
        {"prompt_token_ids": [3]},
    ]
    assert _handle_actor_request(state, {"command": "sleep", "level": 2})["slept"] is True
    assert _handle_actor_request(state, {"command": "wake_up", "tags": ["weights"]})["woke"] is True
    assert calls["sleep_level"] == 2
    assert calls["wake_tags"] == ["weights"]
    _handle_actor_request(state, {"command": "unload_engine"})
    assert state["llm"] is None
    assert calls["shutdown"] == 1


def test_actor_strips_legacy_device_kwarg_for_vllm_019(monkeypatch):
    calls = {}

    class StrictV019LLM:
        def __init__(self, *, model):
            calls["model"] = model

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = StrictV019LLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    state = {"llm": None, "closed": False, "policy_descriptor": None, "engine_topology": None}

    _handle_actor_request(
        state,
        {
            "command": "load_engine",
            "llm_kwargs": {"model": "export", "device": "cuda:0"},
        },
    )

    assert calls["model"] == "export"
    assert state["engine_topology"]["device"] == "cuda:0"


def test_actor_protocol_rejects_generate_before_load():
    state = {"llm": None, "closed": False}
    try:
        _handle_actor_request(
            state,
            {"command": "generate", "prompt_token_ids": [[1]], "sampling_kwargs": {}},
        )
    except RuntimeError as exc:
        assert "not loaded" in str(exc)
    else:
        raise AssertionError("actor must reject generation before an engine is loaded")


def test_actor_protocol_rejects_policy_provenance_mismatch(monkeypatch):
    class FakeLLM:
        def __init__(self, **kwargs):
            pass

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    state = {"llm": None, "closed": False, "policy_descriptor": None}
    _handle_actor_request(
        state,
        {
            "command": "load_engine",
            "llm_kwargs": {"model": "export"},
            "policy_descriptor": {"policy_version": 1},
        },
    )

    try:
        _handle_actor_request(
            state,
            {
                "command": "generate",
                "prompt_token_ids": [[1]],
                "sampling_kwargs": {},
                "expected_policy_descriptor": {"policy_version": 2},
            },
        )
    except RuntimeError as exc:
        assert "provenance mismatch" in str(exc)
    else:
        raise AssertionError("actor must reject rollout against an unexpected policy")


def test_actor_config_rejects_native_sync(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "runtime": {"dev_mode": True},
                "rl": {
                    "vllm_execution_mode": "subprocess",
                    "vllm_sync_strategy": "weight_transfer_nccl",
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        load_config_from_json(path)
    except ValueError as exc:
        assert "subprocess" in str(exc)
    else:
        raise AssertionError("subprocess actor must reject in-process native NCCL sync")
