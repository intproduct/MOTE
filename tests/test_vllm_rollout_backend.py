from __future__ import annotations

import sys
import types
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
    assert calls["export"][0][0].name == "raw-policy-u1"
    assert calls["export"][0][1].name == "hf-policy-u1"
    assert calls["layout"][0].name == "hf-policy-u1"
    assert calls["roundtrip"][0][0].name == "hf-policy-u1"
