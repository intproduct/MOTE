from __future__ import annotations

import importlib
import sys
import threading
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
from MOTE.rl.vllm_weight_mapping import (
    build_weight_mapping_report,
    validate_trainable_patch_transfer_selection,
)
from MOTE.rl.vllm_weight_transfer_capabilities import (
    VLLMWeightTransferCapabilityReport,
    probe_vllm_weight_transfer_capabilities,
)
from MOTE.rl.vllm_weight_transfer_adapters import (
    FourPhaseNCCLTransferAdapter,
    NCCLTransferSettings,
    UpdateOnlyNCCLTransferAdapter,
    trainer_send_weights_to_actor,
    select_nccl_transfer_adapter,
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


def _manager(cfg, tmp_path, transfer_training_names=None):
    return VLLMPolicySyncManager(
        fit_cfg=cfg,
        rl_dir=tmp_path,
        save_policy_checkpoint=lambda output_dir, update_step, checkpoint_name, extra: Path(output_dir),
        transfer_training_names=transfer_training_names,
    )


@pytest.fixture(autouse=True)
def _mock_sync_vllm_preflight(monkeypatch):
    monkeypatch.setattr(
        "MOTE.diagnostics.vllm_export_preflight.validate_vllm_export_preflight",
        lambda *args, **kwargs: {"ok": True, "errors": [], "warnings": []},
    )


def _patch_export_reload(monkeypatch):
    def fake_export(checkpoint_dir, output_dir, **kwargs):
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


def test_adapter_selection_honors_required_level():
    update_report = VLLMWeightTransferCapabilityReport(True, "mock", "update_only")
    four_phase_report = VLLMWeightTransferCapabilityReport(True, "mock", "four_phase")

    assert isinstance(
        select_nccl_transfer_adapter(update_report, required_level="update_only"),
        UpdateOnlyNCCLTransferAdapter,
    )
    assert isinstance(
        select_nccl_transfer_adapter(four_phase_report, required_level="four_phase"),
        FourPhaseNCCLTransferAdapter,
    )
    with pytest.raises(RuntimeError, match="below the configured safety gate"):
        select_nccl_transfer_adapter(update_report, required_level="four_phase")


def test_dryrun_static_reports_mapping_and_keeps_policy_fresh_via_export_reload(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_dryrun_static"
    manager = _manager(cfg, tmp_path)
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "mock", "update_only"),
    )
    _patch_export_reload(monkeypatch)

    result = manager.sync(model=TinyModel(), tokenizer=None, update_step=3, force=True)

    assert result.synced is True
    assert result.policy_version == 3
    assert result.policy_lag_updates == 0
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
    _patch_export_reload(monkeypatch)

    result = manager.sync(
        model=TinyModel(),
        tokenizer=None,
        update_step=1,
        force=True,
        runtime_params={"model.weight": {"shape": [2, 2], "dtype": "float32"}},
    )

    assert result.synced is True
    assert result.policy_version == 1
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


def test_trainable_patch_mapping_matches_optimizer_and_reduces_payload():
    model = TinyMoTNModel()
    for param in model.parameters():
        param.requires_grad_(False)
    transfer_names = []
    for name, param in model.named_parameters():
        if name.startswith("core."):
            param.requires_grad_(True)
            transfer_names.append(name)
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=1e-3)

    validation = validate_trainable_patch_transfer_selection(
        model,
        transfer_training_names=transfer_names,
        optimizer=optimizer,
    )
    report = build_weight_mapping_report(
        model,
        selected_training_names=transfer_names,
        transfer_scope="trainable_patch",
    )

    assert validation["transfer_plan_fingerprint"] == report.transfer_plan_fingerprint
    assert report.transfer_scope == "trainable_patch"
    assert report.tensor_count == len(transfer_names)
    assert report.tensor_count < report.full_tensor_count
    assert 0.0 < report.payload_ratio < 1.0
    assert all(entry.training_name.startswith("core.") for entry in report.entries)


def test_trainable_patch_selection_fails_on_optimizer_or_live_drift():
    model = TinyMoTNModel()
    names = [name for name, _param in model.named_parameters()]
    with pytest.raises(RuntimeError, match="requires_grad"):
        validate_trainable_patch_transfer_selection(
            model,
            transfer_training_names=names[:-1],
        )

    for param in model.parameters():
        param.requires_grad_(False)
    patch_names = []
    for name, param in model.named_parameters():
        if name.startswith("core."):
            param.requires_grad_(True)
            patch_names.append(name)
    incomplete_optimizer = torch.optim.AdamW([dict(model.named_parameters())[patch_names[0]]], lr=1e-3)
    with pytest.raises(RuntimeError, match="optimizer parameters"):
        validate_trainable_patch_transfer_selection(
            model,
            transfer_training_names=patch_names,
            optimizer=incomplete_optimizer,
        )


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

    with pytest.raises(RuntimeError, match="configured native transfer gate"):
        manager.sync(model=TinyModel(), tokenizer=None, update_step=1, force=True, llm=object())


def test_update_only_adapter_uses_request_wrappers_and_sends_weights(monkeypatch):
    base = types.ModuleType("vllm.distributed.weight_transfer.base")

    class Request:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    base.WeightTransferInitRequest = Request
    base.WeightTransferUpdateRequest = Request
    nccl = types.ModuleType("vllm.distributed.weight_transfer.nccl_engine")

    class Info:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    class StrictUpdateInfo:
        def __init__(self, names, dtype_names, shapes, packed=False):
            self.names = names
            self.dtype_names = dtype_names
            self.shapes = shapes
            self.packed = packed

    sent = []

    class Engine:
        trainer_init = staticmethod(lambda info: "group")
        trainer_send_weights = staticmethod(lambda iterator, args: sent.extend(list(iterator)))

    nccl.NCCLWeightTransferInitInfo = Info
    nccl.NCCLWeightTransferUpdateInfo = StrictUpdateInfo
    nccl.NCCLTrainerSendWeightsArgs = Info
    nccl.NCCLWeightTransferEngine = Engine
    monkeypatch.setitem(sys.modules, "vllm.distributed.weight_transfer.base", base)
    monkeypatch.setitem(sys.modules, "vllm.distributed.weight_transfer.nccl_engine", nccl)

    class LLM:
        def init_weight_transfer_engine(self, request):
            self.init_request = request

        def update_weights(self, request):
            self.update_request = request

    model = TinyModel()
    mapping = build_weight_mapping_report(model)
    llm = LLM()
    metadata, initialized = UpdateOnlyNCCLTransferAdapter().transfer(
        llm=llm,
        model=model,
        mapping=mapping,
        settings=NCCLTransferSettings(
            master_address="127.0.0.1",
            master_port=12345,
            tensor_parallel_size=1,
            packed=True,
            timeout_sec=1.0,
        ),
        initialized=False,
        update_step=1,
    )

    assert initialized is True
    assert llm.init_request.init_info["master_port"] == 12345
    assert llm.update_request.update_info["names"] == [entry.transfer_name for entry in mapping.entries]
    assert [name for name, _ in sent] == [entry.transfer_name for entry in mapping.entries]
    assert metadata["weight_transfer_adapter"] == "nccl_update_only"


def test_subprocess_trainer_send_timeout_is_fail_closed(monkeypatch):
    nccl = types.ModuleType("vllm.distributed.weight_transfer.nccl_engine")

    class Info:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    release = threading.Event()

    class Engine:
        @staticmethod
        def trainer_send_weights(iterator, args):
            release.wait(timeout=1.0)

    nccl.NCCLTrainerSendWeightsArgs = Info
    nccl.NCCLWeightTransferEngine = Engine
    monkeypatch.setitem(sys.modules, "vllm.distributed.weight_transfer.nccl_engine", nccl)
    model = TinyModel()
    try:
        with pytest.raises(TimeoutError, match="must be discarded"):
            trainer_send_weights_to_actor(
                model=model,
                mapping=build_weight_mapping_report(model),
                settings=NCCLTransferSettings(
                    master_address="127.0.0.1",
                    master_port=12345,
                    tensor_parallel_size=1,
                    packed=True,
                    timeout_sec=0.01,
                ),
                group=object(),
            )
    finally:
        release.set()


def test_manager_allows_explicit_update_only_adapter(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_nccl"
    cfg.rl.vllm_native_transfer_required_level = "update_only"
    cfg.rl.vllm_weight_transfer_validate_after_sync = False
    cfg.rl.vllm_weight_transfer_master_port = 12345
    manager = _manager(cfg, tmp_path)
    manager.policy_version = 0
    manager.export_dir = tmp_path / "hf-policy-u0"
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "mock", "update_only"),
    )

    class Adapter:
        def transfer(self, **kwargs):
            return {"weight_transfer_adapter": "fake_update_only"}, True

    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.select_nccl_transfer_adapter",
        lambda report, required_level: Adapter(),
    )
    model = TinyModel()
    runtime_params = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
    result = manager.sync(
        model=model,
        tokenizer=None,
        update_step=1,
        force=True,
        llm=object(),
        runtime_params=runtime_params,
    )

    assert result.synced is True
    assert result.metadata["vllm_weight_transfer_native_sync"] is True
    assert result.metadata["weight_transfer_adapter"] == "fake_update_only"


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


class _FakeSubprocessActor:
    def __init__(self, name, *, fail_finish=False):
        self.name = name
        self.fail_finish = fail_finish
        self.calls = []
        self.pending_descriptor = None
        self.descriptor = {"policy_version": 0, "policy_fingerprint": "bootstrap", "export_dir": "bootstrap"}

    def weight_transfer_capabilities(self):
        self.calls.append("capabilities")
        return {"native_transfer_level": "update_only", "missing": []}

    def begin_init_weight_transfer(self, *, init_info):
        self.calls.append(("begin_init", dict(init_info)))
        return 40

    def finish_init_weight_transfer(self, request_id):
        self.calls.append(("finish_init", request_id))
        return {"initialized": True}

    def begin_weight_update(self, **kwargs):
        self.calls.append("begin")
        self.pending_descriptor = dict(kwargs["policy_descriptor"])
        return 41

    def finish_weight_update(self, request_id, *, timeout_sec=None):
        self.calls.append(("finish", request_id))
        if self.fail_finish:
            raise RuntimeError(f"{self.name} receive failed")
        return {
            "actor_status": "UPDATED_PENDING_COMMIT",
            "rollout_facing_validation": {"ok": True},
            "checksum_validation": {"ok": None},
        }

    def commit_weight_update(self, *, policy_descriptor):
        self.calls.append("commit")
        assert dict(policy_descriptor) == self.pending_descriptor
        self.descriptor = dict(policy_descriptor)
        return {"committed": True, "actor_status": "READY", "policy_descriptor": self.descriptor}


def test_subprocess_update_only_sync_commits_all_actors_after_receives(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_nccl"
    cfg.rl.vllm_native_transfer_required_level = "update_only"
    cfg.rl.vllm_weight_transfer_master_port = 32000
    manager = _manager(cfg, tmp_path)
    manager.policy_version = 0
    manager.export_dir = tmp_path / "hf-policy-u0"
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "0.19.0", "update_only"),
    )
    sends = []
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.trainer_send_weights_to_actor",
        lambda **kwargs: sends.append(kwargs["settings"].master_port) or {"trainer_send_sec": 0.1},
    )
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.trainer_initialize_nccl_group",
        lambda settings: f"group-{settings.master_port}",
    )
    actors = [_FakeSubprocessActor("actor_0"), _FakeSubprocessActor("actor_1")]
    result = manager.sync(
        model=TinyModel(),
        tokenizer=None,
        update_step=1,
        force=True,
        actors=[
            ("actor_0", actors[0], {"tensor_parallel_size": 1}),
            ("actor_1", actors[1], {"tensor_parallel_size": 1}),
        ],
    )

    assert result.policy_version == 1
    assert result.metadata["vllm_weight_transfer_native_sync"] is True
    assert result.metadata["weight_transfer_commit_barrier"] is True
    assert result.metadata["weight_transfer_commit_atomic"] is False
    assert sends == [32000, 32001]
    assert all(actor.calls.index("commit") > actor.calls.index(("finish", 41)) for actor in actors)
    assert all(actor.descriptor["policy_version"] == 1 for actor in actors)

    second = manager.sync(
        model=TinyModel(),
        tokenizer=None,
        update_step=2,
        force=True,
        actors=[
            ("actor_0", actors[0], {"tensor_parallel_size": 1}),
            ("actor_1", actors[1], {"tensor_parallel_size": 1}),
        ],
    )
    assert second.policy_version == 2
    assert sends == [32000, 32001, 32000, 32001]
    assert all(sum(call[0] == "begin_init" for call in actor.calls if isinstance(call, tuple)) == 1 for actor in actors)


def test_subprocess_trainable_patch_sync_sends_only_allowlisted_parameters(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_nccl"
    cfg.rl.vllm_native_transfer_required_level = "update_only"
    cfg.rl.vllm_weight_transfer_scope = "trainable_patch"
    cfg.rl.vllm_weight_transfer_master_port = 32200
    model = TinyMoTNModel()
    for param in model.parameters():
        param.requires_grad_(False)
    transfer_names = []
    for name, param in model.named_parameters():
        if name.startswith("core."):
            param.requires_grad_(True)
            transfer_names.append(name)
    manager = _manager(cfg, tmp_path, transfer_training_names=transfer_names)
    manager.policy_version = 0
    manager.export_dir = tmp_path / "hf-policy-u0"
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "0.19.0", "update_only"),
    )
    sent_mappings = []
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.trainer_send_weights_to_actor",
        lambda **kwargs: sent_mappings.append(kwargs["mapping"]) or {"trainer_send_sec": 0.1},
    )
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.trainer_initialize_nccl_group",
        lambda settings: f"group-{settings.master_port}",
    )
    actors = [_FakeSubprocessActor("actor_0"), _FakeSubprocessActor("actor_1")]

    result = manager.sync(
        model=model,
        tokenizer=None,
        update_step=1,
        force=True,
        actors=[
            ("actor_0", actors[0], {"tensor_parallel_size": 1}),
            ("actor_1", actors[1], {"tensor_parallel_size": 1}),
        ],
    )

    assert result.metadata["vllm_weight_transfer_scope"] == "trainable_patch"
    assert result.metadata["weight_transfer_payload_ratio"] < 1.0
    assert len(sent_mappings) == 2
    assert all(
        {entry.training_name for entry in mapping.entries} == set(transfer_names)
        for mapping in sent_mappings
    )
    assert all(actor.descriptor["weight_transfer_scope"] == "trainable_patch" for actor in actors)


def test_trainable_patch_sync_rejects_frozen_parameter_drift(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_nccl"
    cfg.rl.vllm_weight_transfer_scope = "trainable_patch"
    model = TinyMoTNModel()
    for param in model.parameters():
        param.requires_grad_(False)
    transfer_names = []
    for name, param in model.named_parameters():
        if name.startswith("core."):
            param.requires_grad_(True)
            transfer_names.append(name)
    manager = _manager(cfg, tmp_path, transfer_training_names=transfer_names)
    manager._frozen_drift_metadata(model, initialize=True)
    with torch.no_grad():
        model.not_router_string.weight.add_(1.0)

    with pytest.raises(RuntimeError, match="frozen policy parameters changed"):
        manager._frozen_drift_metadata(model)


def test_subprocess_partial_receive_failure_never_commits_or_advances_policy(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.rl.vllm_sync_strategy = "weight_transfer_nccl"
    cfg.rl.vllm_native_transfer_required_level = "update_only"
    cfg.rl.vllm_weight_transfer_master_port = 32100
    manager = _manager(cfg, tmp_path)
    manager.policy_version = 4
    manager.export_dir = tmp_path / "hf-policy-u4"
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.probe_vllm_weight_transfer_capabilities",
        lambda: VLLMWeightTransferCapabilityReport(True, "0.19.0", "update_only"),
    )
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.trainer_send_weights_to_actor",
        lambda **kwargs: {"trainer_send_sec": 0.1},
    )
    monkeypatch.setattr(
        "MOTE.rl.vllm_sync.trainer_initialize_nccl_group",
        lambda settings: f"group-{settings.master_port}",
    )
    actor_0 = _FakeSubprocessActor("actor_0")
    actor_1 = _FakeSubprocessActor("actor_1", fail_finish=True)

    with pytest.raises(RuntimeError, match="actor_1 receive failed"):
        manager.sync(
            model=TinyModel(),
            tokenizer=None,
            update_step=5,
            force=True,
            actors=[
                ("actor_0", actor_0, {"tensor_parallel_size": 1}),
                ("actor_1", actor_1, {"tensor_parallel_size": 1}),
            ],
        )

    assert manager.policy_version == 4
    assert "commit" not in actor_0.calls
    assert "commit" not in actor_1.calls
