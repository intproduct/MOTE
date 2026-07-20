from __future__ import annotations

import sys
import types
import json
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
from MOTE.rl.vllm_rollout import VLLMRolloutBackend
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
