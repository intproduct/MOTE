from __future__ import annotations

import sys
import types
import json
import os
import signal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg

from MOTE.rl.vllm_actor import VLLMActorClient, _engine_topology_snapshot, _handle_actor_request
import MOTE.rl.vllm_actor as vllm_actor_module
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
    assert result["diagnostics"]["llm_generate_called"] is True
    assert result["diagnostics"]["prompt_count"] == 2
    assert result["diagnostics"]["output_row_count"] == 2
    assert result["diagnostics"]["completion_count"] == 2
    assert result["diagnostics"]["generate_end_time"] >= result["diagnostics"]["generate_start_time"]
    assert result["diagnostics"]["engine_config"]["model"] == "export"
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


def test_actor_generate_batches_multiple_prompts_with_per_prompt_sampling_params(monkeypatch):
    calls = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class Completion:
        def __init__(self, prompt_token, sample_index):
            self.token_ids = [prompt_token, sample_index]
            self.text = f"{prompt_token}:{sample_index}"
            self.finish_reason = "stop"

    class Output:
        def __init__(self, prompt_token, count):
            self.outputs = [
                Completion(prompt_token, sample_index)
                for sample_index in range(count)
            ]

    class FakeLLM:
        def generate(self, *, prompts, sampling_params):
            calls.append((prompts, sampling_params))
            return [
                Output(prompt["prompt_token_ids"][-1], params.kwargs["n"])
                for prompt, params in zip(prompts, sampling_params)
            ]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    state = {
        "llm": FakeLLM(),
        "status": "READY",
        "policy_descriptor": {"policy_version": 0},
        "engine_config": {"model": "mock"},
        "engine_resources": {},
    }
    result = _handle_actor_request(
        state,
        {
            "command": "generate",
            "prompt_token_ids": [[10], [20]],
            "sampling_kwargs_by_prompt": [
                {"n": 3, "temperature": 1.0, "seed": 101},
                {"n": 3, "temperature": 1.0, "seed": 102},
            ],
            "expected_policy_descriptor": {"policy_version": 0},
        },
    )

    assert len(calls) == 1
    prompts, params = calls[0]
    assert [item["prompt_token_ids"] for item in prompts] == [[10], [20]]
    assert [item.kwargs["seed"] for item in params] == [101, 102]
    assert [len(completions) for completions in result["outputs"]] == [3, 3]
    assert result["diagnostics"]["prompt_count"] == 2
    assert result["diagnostics"]["completion_count"] == 6


def test_actor_close_acknowledges_before_engine_shutdown():
    calls = []

    class BlockingEngine:
        def shutdown(self):
            calls.append("shutdown")
            raise AssertionError("close command must not synchronously shut down the engine")

    engine = BlockingEngine()
    state = {"llm": engine, "closed": False, "status": "READY"}

    result = _handle_actor_request(state, {"command": "close"})

    assert result["closed"] is True
    assert result["actor_resources"]["process_id"] == os.getpid()
    assert state["closed"] is True
    assert state["status"] == "CLOSED"
    assert state["llm"] is engine
    assert calls == []


def test_actor_client_close_uses_shutdown_timeout_and_kills_stuck_group(monkeypatch):
    events = []

    class FakeConnection:
        def send(self, payload):
            events.append(("send", payload["command"]))

        def poll(self, timeout):
            events.append(("poll", timeout))
            return False

        def close(self):
            events.append(("connection_close", None))

    class FakeProcess:
        pid = 123

        def __init__(self):
            self.alive = True
            self.exitcode = None

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            events.append(("join", timeout))

    client = VLLMActorClient(request_timeout_sec=600.0, shutdown_timeout_sec=7.0)
    client._process = FakeProcess()
    client._connection = FakeConnection()
    signals = []

    def terminate(process, sig):
        signals.append(sig.name)
        events.append(("signal", sig.name))
        process.alive = False

    monkeypatch.setattr(client, "_terminate_actor_group", terminate)

    client.close()

    poll_timeouts = [timeout for event, timeout in events if event == "poll"]
    assert len(poll_timeouts) == 1
    assert poll_timeouts[0] == pytest.approx(7.0, abs=0.01)
    assert ("poll", 600.0) not in events
    assert signals == ["SIGTERM"]
    assert client._process is None
    assert client._connection is None


def test_actor_client_close_keeps_handle_and_raises_when_process_survives_kill(monkeypatch):
    class FakeConnection:
        def close(self):
            return None

    class StuckProcess:
        pid = 456
        exitcode = None

        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    process = StuckProcess()
    client = VLLMActorClient(request_timeout_sec=1.0, shutdown_timeout_sec=1.0)
    client._process = process
    client._connection = FakeConnection()
    signals = []
    monkeypatch.setattr(
        client,
        "_terminate_actor_group",
        lambda _process, sig: signals.append(sig.name),
    )

    with pytest.raises(RuntimeError, match="teardown did not finish"):
        client.close(force=True)

    assert signals == ["SIGTERM", "SIGKILL"]
    assert client._process is process
    assert client._connection is None


def test_actor_client_close_releases_stopped_process_handle(monkeypatch):
    events = []

    class FakeConnection:
        def close(self):
            events.append("connection_close")

    class FakeProcess:
        pid = 789
        exitcode = -9

        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            events.append(("join", timeout))

        def close(self):
            events.append("process_close")

    process = FakeProcess()
    client = VLLMActorClient(request_timeout_sec=1.0, shutdown_timeout_sec=1.0)
    client._process = process
    client._connection = FakeConnection()

    def terminate(_process, sig):
        events.append(sig.name)
        if sig == signal.SIGKILL:
            process.alive = False

    monkeypatch.setattr(client, "_terminate_actor_group", terminate)

    report = client.close(force=True)

    assert report["closed"] is True
    assert report["signals_sent"] == ["SIGTERM", "SIGKILL"]
    assert "process_close" in events
    assert client._process is None


def test_actor_client_close_terminates_reported_engine_children(monkeypatch):
    class FakeConnection:
        def close(self):
            return None

    class FakeProcess:
        pid = 900
        exitcode = -15

        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            return None

    process = FakeProcess()
    client = VLLMActorClient(request_timeout_sec=1.0, shutdown_timeout_sec=1.0)
    client._process = process
    client._connection = FakeConnection()
    client.last_engine_info = {
        "actor_resources": {"os_child_process_ids": [901]}
    }
    child_alive = {901: True}
    child_signals = []

    def terminate_actor(_process, _sig):
        process.alive = False

    def signal_children(pids, sig):
        child_signals.append((list(pids), sig.name))
        for pid in pids:
            child_alive[pid] = False

    monkeypatch.setattr(client, "_terminate_actor_group", terminate_actor)
    monkeypatch.setattr(client, "_pid_exists", lambda pid: child_alive.get(pid, False))
    monkeypatch.setattr(client, "_signal_pids", signal_children)
    monkeypatch.setattr(
        client,
        "_engine_child_pid_evidence",
        lambda *_args, **_kwargs: {
            "fresh": [901],
            "historical": [901],
            "verified": [901],
        },
    )

    report = client.close(force=True)

    assert report["closed"] is True
    assert report["known_child_pids"] == [901]
    assert report["surviving_child_pids"] == []
    assert child_signals == [([901], "SIGTERM")]


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


def test_actor_ping_does_not_probe_cuda_runtime(monkeypatch):
    probes = []

    def snapshot(*, probe_cuda_runtime=True):
        probes.append(probe_cuda_runtime)
        return {"cuda_runtime_probe_deferred": not probe_cuda_runtime}

    monkeypatch.setattr(vllm_actor_module, "_cuda_resource_snapshot", snapshot)
    result = _handle_actor_request(
        {"llm": object(), "closed": False, "status": "READY"},
        {"command": "ping"},
    )

    assert probes == [False]
    assert result["actor_resources"]["cuda_runtime_probe_deferred"] is True


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


def test_actor_update_only_state_machine_blocks_generation_until_commit(monkeypatch):
    calls = []
    resource_probes = []

    class Info:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    class Request:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def init_weight_transfer_engine(self, request):
            calls.append("init")

        def update_weights(self, request):
            calls.append("update")

        def reset_prefix_cache(self):
            calls.append("reset_cache")
            return True

        def generate(self, *args, **kwargs):
            return [FakeOutput([2])]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    base = types.ModuleType("vllm.distributed.weight_transfer.base")
    base.WeightTransferInitRequest = Request
    base.WeightTransferUpdateRequest = Request
    nccl = types.ModuleType("vllm.distributed.weight_transfer.nccl_engine")
    nccl.NCCLWeightTransferInitInfo = Info
    nccl.NCCLWeightTransferUpdateInfo = Info
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed.weight_transfer.base", base)
    monkeypatch.setitem(sys.modules, "vllm.distributed.weight_transfer.nccl_engine", nccl)
    monkeypatch.setattr(
        vllm_actor_module,
        "_cuda_resource_snapshot",
        lambda *, probe_cuda_runtime=True: resource_probes.append(probe_cuda_runtime) or {},
    )

    state = {
        "llm": FakeLLM(),
        "closed": False,
        "status": "READY",
        "policy_descriptor": {"policy_version": 0},
        "pending_policy_descriptor": None,
        "weight_transfer_initialized": False,
    }
    _handle_actor_request(
        state,
        {
            "command": "init_weight_transfer",
            "init_info": {"master_address": "127.0.0.1", "master_port": 1, "rank_offset": 1, "world_size": 2},
        },
    )
    result = _handle_actor_request(
        state,
        {
            "command": "receive_weight_update",
            "update_info": {
                "names": ["model.weight"],
                "dtype_names": ["float32"],
                "shapes": [[2, 2]],
                "packed": True,
            },
            "policy_descriptor": {"policy_version": 1},
            "expected_checksums": {},
            "validation_prompt_token_ids": [1],
        },
    )
    assert result["actor_status"] == "UPDATED_PENDING_COMMIT"
    assert calls == ["init", "update", "reset_cache"]
    assert resource_probes == [False]
    with pytest.raises(RuntimeError, match="cannot generate"):
        _handle_actor_request(
            state,
            {"command": "generate", "prompt_token_ids": [[1]], "sampling_kwargs": {}},
        )
    commit = _handle_actor_request(
        state,
        {"command": "commit_weight_update", "policy_descriptor": {"policy_version": 1}},
    )
    assert commit["actor_status"] == "READY"
    assert state["policy_descriptor"] == {"policy_version": 1}


def test_actor_config_accepts_update_only_native_sync(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "runtime": {"dev_mode": True},
                "rl": {
                    "vllm_execution_mode": "subprocess",
                    "vllm_sync_strategy": "weight_transfer_nccl",
                    "vllm_native_transfer_required_level": "update_only",
                    "vllm_actor_cuda_visible_devices": ["0"],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config_from_json(path)
    assert cfg.rl.vllm_sync_strategy == "weight_transfer_nccl"
    assert cfg.rl.vllm_native_transfer_required_level == "update_only"


def test_actor_config_rejects_four_phase_and_tp2_native_sync(tmp_path):
    for suffix, rl_config, expected in [
        (
            "four_phase",
            {
                "vllm_execution_mode": "subprocess",
                "vllm_sync_strategy": "weight_transfer_nccl",
                "vllm_native_transfer_required_level": "four_phase",
                "vllm_actor_cuda_visible_devices": ["0"],
            },
            "update-only",
        ),
        (
            "tp2",
            {
                "vllm_execution_mode": "subprocess",
                "vllm_sync_strategy": "weight_transfer_nccl",
                "vllm_native_transfer_required_level": "update_only",
                "vllm_actor_cuda_visible_devices": ["0", "1"],
                "vllm_tensor_parallel_size": 2,
            },
            "TP1",
        ),
    ]:
        path = tmp_path / f"{suffix}.json"
        path.write_text(json.dumps({"runtime": {"dev_mode": True}, "rl": rl_config}), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            load_config_from_json(path)
