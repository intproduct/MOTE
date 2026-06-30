from __future__ import annotations

import importlib
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
setattr(sys.modules["MOTE"], "rl", sys.modules["MOTE.rl"])

from MOTE.config.defaults import make_default_config
from MOTE.rl.vllm_sync import VLLMPolicySyncManager
from MOTE.rl.vllm_weight_mapping import build_weight_mapping_report
from MOTE.rl.vllm_weight_transfer_capabilities import (
    VLLMWeightTransferCapabilityReport,
    probe_vllm_weight_transfer_capabilities,
)


class TinyModel(torch.nn.Module):
    base_model_prefix = "model"

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(2, 2)


class TinyMoTNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.gate = torch.nn.Module()
        self.core.gate.router = torch.nn.Linear(2, 2)
        self.core.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.core.global_block = torch.nn.Linear(2, 2)
        self.not_router_string = torch.nn.Linear(2, 2)


def _cfg(tmp_path):
    cfg = make_default_config()
    cfg.rl.rollout_backend = "vllm"
    cfg.rl.vllm_export_root = str(tmp_path / "exports")
    return cfg


def _manager(cfg, tmp_path):
    return VLLMPolicySyncManager(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: Path(output_dir),
    )


def _fake_importer(level):
    modules = {}
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "mock"

    class LLM:
        def init_weight_transfer_engine(self, request):
            return None

        def update_weights(self, request):
            return None

    if level == "four_phase":
        LLM.start_weight_update = lambda self, is_checkpoint_format=True: None
        LLM.finish_weight_update = lambda self: None
    if level in {"update_only", "four_phase"}:
        vllm.LLM = LLM
    modules["vllm"] = vllm

    config = types.ModuleType("vllm.config")
    if level in {"update_only", "four_phase"}:
        config.WeightTransferConfig = lambda backend="nccl": types.SimpleNamespace(backend=backend)
    modules["vllm.config"] = config

    nccl = types.ModuleType("vllm.distributed.weight_transfer.nccl_engine")
    if level in {"update_only", "four_phase"}:
        class NCCLWeightTransferEngine:
            trainer_init = staticmethod(lambda init_info: object())
            trainer_send_weights = staticmethod(lambda iterator, trainer_args: None)

        nccl.NCCLWeightTransferEngine = NCCLWeightTransferEngine
        nccl.NCCLWeightTransferInitInfo = type("NCCLWeightTransferInitInfo", (), {})
        nccl.NCCLWeightTransferUpdateInfo = type("NCCLWeightTransferUpdateInfo", (), {})
        nccl.NCCLTrainerSendWeightsArgs = type("NCCLTrainerSendWeightsArgs", (), {})
    modules["vllm.distributed.weight_transfer.nccl_engine"] = nccl

    ipc = types.ModuleType("vllm.distributed.weight_transfer.ipc_engine")
    modules["vllm.distributed.weight_transfer.ipc_engine"] = ipc

    def fake_import(name):
        if name in modules:
            return modules[name]
        raise ImportError(name)

    return fake_import


@pytest.mark.parametrize(
    ("mock_level", "expected"),
    [("none", "none"), ("update_only", "update_only"), ("four_phase", "four_phase")],
)
def test_capability_probe_reports_native_transfer_levels(monkeypatch, mock_level, expected):
    import MOTE.rl.vllm_weight_transfer_capabilities as caps

    monkeypatch.setattr(caps.importlib, "import_module", _fake_importer(mock_level))

    report = probe_vllm_weight_transfer_capabilities()

    assert report.native_transfer_level == expected
    if expected != "four_phase":
        assert "vllm.LLM.start_weight_update" in report.missing


def test_dryrun_static_uses_mapping_without_marking_policy_fresh(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_dryrun_static"
    manager = _manager(cfg, tmp_path)
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "mock", "update_only"),
    )

    result = manager.sync(model=TinyModel(), tokenizer=None, update_step=3, force=True)

    assert result.synced is False
    assert result.policy_version == -1
    assert result.policy_lag_updates == 3
    assert result.metadata["vllm_weight_transfer_dryrun_mode"] == "static"
    assert result.metadata["weight_transfer_runtime_inspection_available"] is False


def test_dryrun_runtime_adds_vllm_name_shape_inspection(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_dryrun_runtime"
    manager = _manager(cfg, tmp_path)
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "mock", "update_only"),
    )

    result = manager.sync(
        model=TinyModel(),
        tokenizer=None,
        update_step=1,
        force=True,
        runtime_params={"model.weight": {"shape": [2, 2], "dtype": "float32"}},
    )

    assert result.synced is False
    assert result.metadata["vllm_weight_transfer_dryrun_mode"] == "runtime"
    assert result.metadata["weight_transfer_runtime_inspection_available"] is True
    assert "model.bias" in result.metadata["weight_mapping_report"]["missing_in_vllm"]


def test_module_aware_motn_report_uses_exact_required_keys():
    report = build_weight_mapping_report(TinyMoTNModel())

    assert "core.gate.router.weight" in report.required_motn_keys
    assert "core.blocks.0.weight" in report.required_motn_keys
    assert "core.global_block.weight" in report.required_motn_keys
    assert all("not_router_string" not in key for key in report.required_motn_keys)
    assert report.motn_coverage == 1.0


def test_weight_transfer_nccl_refuses_update_only_without_explicit_fallback(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_nccl"
    cfg.rl.vllm_weight_transfer_fallback_to_export_reload = False
    manager = _manager(cfg, tmp_path)
    manager.policy_version = 0
    manager.export_dir = tmp_path / "hf-policy-u0"
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "mock", "update_only", missing=["vllm.LLM.start_weight_update"]),
    )

    with pytest.raises(RuntimeError, match="requires native_transfer_level='four_phase'"):
        manager.sync(model=TinyModel(), tokenizer=None, update_step=1, force=True, llm=object())


def test_weight_transfer_nccl_fallback_requires_explicit_config(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_nccl"
    cfg.rl.vllm_weight_transfer_fallback_to_export_reload = True
    manager = _manager(cfg, tmp_path)
    manager.policy_version = 0
    manager.export_dir = tmp_path / "hf-policy-u0"
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "mock", "update_only", missing=["vllm.LLM.start_weight_update"]),
    )

    def fake_export(checkpoint_dir, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return types.SimpleNamespace(output_dir=Path(output_dir))

    monkeypatch.setattr("MOTE.export.hf_export.export_fitmotn_hf_roundtrip", fake_export)
    monkeypatch.setattr("MOTE.export.validate.validate_export_layout", lambda path: types.SimpleNamespace(ok=True, errors=[], warnings=[]))
    monkeypatch.setattr("MOTE.export.roundtrip.validate_hf_roundtrip", lambda path, **kwargs: types.SimpleNamespace(ok=True, errors=[], warnings=[]))

    result = manager.sync(model=TinyModel(), tokenizer=None, update_step=1, force=True, llm=object())

    assert result.synced is True
    assert result.policy_version == 1
    assert result.metadata["vllm_weight_transfer_fallback_used"] is True
    assert result.metadata["vllm_sync_strategy_requested"] == "weight_transfer_nccl"


def test_runtime_coverage_below_full_blocks_native_transfer(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_nccl"
    cfg.rl.vllm_weight_transfer_fallback_to_export_reload = False
    manager = _manager(cfg, tmp_path)
    manager.policy_version = 0
    manager.export_dir = tmp_path / "hf-policy-u0"
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "mock", "four_phase"),
    )

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        manager.sync(
            model=TinyModel(),
            tokenizer=None,
            update_step=1,
            force=True,
            llm=object(),
            runtime_params={"model.weight": {"shape": [2, 2], "dtype": "float32"}},
        )
