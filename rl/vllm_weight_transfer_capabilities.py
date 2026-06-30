from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

NativeTransferLevel = Literal["none", "update_only", "four_phase"]


@dataclass
class VLLMWeightTransferCapabilityReport:
    vllm_available: bool
    vllm_version: Optional[str]
    native_transfer_level: NativeTransferLevel
    weight_transfer_config: bool = False
    llm_methods: Dict[str, bool] = field(default_factory=dict)
    llm_method_signatures: Dict[str, Optional[str]] = field(default_factory=dict)
    nccl_available: bool = False
    nccl_trainer_apis: Dict[str, bool] = field(default_factory=dict)
    nccl_dataclasses: Dict[str, bool] = field(default_factory=dict)
    ipc_available: bool = False
    ipc_trainer_apis: Dict[str, bool] = field(default_factory=dict)
    ipc_dataclasses: Dict[str, bool] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _signature(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def _load_attr(module_name: str, attr: str) -> tuple[Any, Optional[str]]:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr, None), None
    except Exception as exc:
        return None, f"{module_name}.{attr}: {type(exc).__name__}: {exc}"


def probe_vllm_weight_transfer_capabilities() -> VLLMWeightTransferCapabilityReport:
    try:
        vllm = importlib.import_module("vllm")
    except Exception as exc:
        return VLLMWeightTransferCapabilityReport(
            vllm_available=False,
            vllm_version=None,
            native_transfer_level="none",
            missing=["vllm"],
            errors=[f"import vllm failed: {type(exc).__name__}: {exc}"],
        )

    missing: List[str] = []
    errors: List[str] = []
    version = getattr(vllm, "__version__", None)

    weight_transfer_config, err = _load_attr("vllm.config", "WeightTransferConfig")
    if err:
        errors.append(err)
    has_wtc = weight_transfer_config is not None
    if not has_wtc:
        missing.append("vllm.config.WeightTransferConfig")

    llm_cls = getattr(vllm, "LLM", None)
    llm_methods: Dict[str, bool] = {}
    llm_method_signatures: Dict[str, Optional[str]] = {}
    for name in [
        "init_weight_transfer_engine",
        "start_weight_update",
        "update_weights",
        "finish_weight_update",
    ]:
        method = getattr(llm_cls, name, None) if llm_cls is not None else None
        llm_methods[name] = callable(method)
        llm_method_signatures[name] = _signature(method)
        if not callable(method):
            missing.append(f"vllm.LLM.{name}")

    nccl_engine, err = _load_attr("vllm.distributed.weight_transfer.nccl_engine", "NCCLWeightTransferEngine")
    if err:
        errors.append(err)
    nccl_trainer_apis = {
        "trainer_init": callable(getattr(nccl_engine, "trainer_init", None)),
        "trainer_send_weights": callable(getattr(nccl_engine, "trainer_send_weights", None)),
    }
    nccl_dataclasses: Dict[str, bool] = {}
    for name in [
        "NCCLWeightTransferInitInfo",
        "NCCLWeightTransferUpdateInfo",
        "NCCLTrainerSendWeightsArgs",
    ]:
        obj, err = _load_attr("vllm.distributed.weight_transfer.nccl_engine", name)
        if err:
            errors.append(err)
        nccl_dataclasses[name] = obj is not None
        if obj is None:
            missing.append(f"vllm.distributed.weight_transfer.nccl_engine.{name}")
    for name, ok in nccl_trainer_apis.items():
        if not ok:
            missing.append(f"NCCLWeightTransferEngine.{name}")
    nccl_available = bool(nccl_engine is not None and all(nccl_trainer_apis.values()) and all(nccl_dataclasses.values()))

    ipc_engine, err = _load_attr("vllm.distributed.weight_transfer.ipc_engine", "IPCWeightTransferEngine")
    if err:
        errors.append(err)
    ipc_trainer_apis = {
        "trainer_send_weights": callable(getattr(ipc_engine, "trainer_send_weights", None)),
    }
    ipc_dataclasses = {}
    for name in ["IPCWeightTransferInitInfo", "IPCWeightTransferUpdateInfo", "IPCTrainerSendWeightsArgs"]:
        obj, err = _load_attr("vllm.distributed.weight_transfer.ipc_engine", name)
        if err:
            errors.append(err)
        ipc_dataclasses[name] = obj is not None
    ipc_available = bool(ipc_engine is not None and all(ipc_trainer_apis.values()) and all(ipc_dataclasses.values()))

    update_only_ready = bool(
        has_wtc
        and llm_methods.get("init_weight_transfer_engine")
        and llm_methods.get("update_weights")
        and nccl_available
    )
    four_phase_ready = bool(
        update_only_ready
        and llm_methods.get("start_weight_update")
        and llm_methods.get("finish_weight_update")
    )
    if four_phase_ready:
        level: NativeTransferLevel = "four_phase"
    elif update_only_ready:
        level = "update_only"
    else:
        level = "none"

    return VLLMWeightTransferCapabilityReport(
        vllm_available=True,
        vllm_version=None if version is None else str(version),
        native_transfer_level=level,
        weight_transfer_config=has_wtc,
        llm_methods=llm_methods,
        llm_method_signatures=llm_method_signatures,
        nccl_available=nccl_available,
        nccl_trainer_apis=nccl_trainer_apis,
        nccl_dataclasses=nccl_dataclasses,
        ipc_available=ipc_available,
        ipc_trainer_apis=ipc_trainer_apis,
        ipc_dataclasses=ipc_dataclasses,
        missing=sorted(set(missing)),
        errors=errors,
    )
