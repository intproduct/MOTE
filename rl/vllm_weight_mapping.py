from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


MOTN_REQUIRED_CHILDREN = (
    "core.gate",
    "core.gate.router",
    "core.blocks",
    "core.global_block",
)


@dataclass
class WeightMappingEntry:
    training_name: str
    transfer_name: str
    shape: List[int]
    dtype: str
    numel: int
    num_bytes: int
    is_motn: bool = False
    motn_role: Optional[str] = None
    tied_alias_of: Optional[str] = None


@dataclass
class VLLMWeightMappingReport:
    entries: List[WeightMappingEntry] = field(default_factory=list)
    missing_in_training_model: List[str] = field(default_factory=list)
    missing_in_vllm: List[str] = field(default_factory=list)
    shape_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    dtype_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    tied_weight_aliases: Dict[str, str] = field(default_factory=dict)
    required_motn_keys: List[str] = field(default_factory=list)
    transferred_motn_keys: List[str] = field(default_factory=list)
    coverage: float = 1.0
    motn_coverage: float = 1.0
    tensor_count: int = 0
    num_bytes: int = 0
    runtime_inspection_available: bool = False
    transfer_scope: str = "full_policy"
    transfer_plan_fingerprint: str = ""
    full_tensor_count: int = 0
    full_num_bytes: int = 0
    payload_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["entries"] = [asdict(entry) for entry in self.entries]
        return data


def canonical_transfer_name(name: str, *, base_model_prefix: str = "model") -> str:
    candidates = [str(base_model_prefix or "model")]
    for candidate in ("model", "transformer"):
        if candidate not in candidates:
            candidates.append(candidate)
    for prefix in candidates:
        marker = prefix + "."
        if name.startswith(marker):
            return "model." + name[len(marker) :]
    return name


def _tensor_num_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _module_param_names(model: torch.nn.Module, module_prefix: str, module: torch.nn.Module) -> List[str]:
    keys = []
    for rel_name, _param in module.named_parameters(recurse=True):
        keys.append(f"{module_prefix}.{rel_name}" if rel_name else module_prefix)
    return keys


def collect_module_aware_motn_keys(model: torch.nn.Module) -> Dict[str, str]:
    required: Dict[str, str] = {}
    for module_name, module in model.named_modules():
        for child_path in MOTN_REQUIRED_CHILDREN:
            target = module
            found = True
            for part in child_path.split("."):
                target = getattr(target, part, None)
                if target is None:
                    found = False
                    break
            if not found or not isinstance(target, torch.nn.Module):
                continue
            prefix = f"{module_name}.{child_path}" if module_name else child_path
            for key in _module_param_names(model, prefix, target):
                if key in dict(model.named_parameters()):
                    required[key] = child_path
    return required


def _runtime_param_map(runtime_params: Optional[Mapping[str, Any]]) -> Dict[str, Tuple[List[int], Optional[str]]]:
    if runtime_params is None:
        return {}
    out: Dict[str, Tuple[List[int], Optional[str]]] = {}
    for name, value in runtime_params.items():
        if isinstance(value, torch.Tensor):
            out[str(name)] = (list(value.shape), str(value.dtype).replace("torch.", ""))
        elif isinstance(value, Mapping):
            shape = value.get("shape")
            dtype = value.get("dtype")
            out[str(name)] = (list(shape or []), None if dtype is None else str(dtype).replace("torch.", ""))
        elif isinstance(value, (tuple, list)) and len(value) >= 1:
            shape = list(value[0])
            dtype = None if len(value) < 2 else str(value[1]).replace("torch.", "")
            out[str(name)] = (shape, dtype)
    return out


def build_weight_mapping_report(
    model: torch.nn.Module,
    *,
    runtime_params: Optional[Mapping[str, Any]] = None,
    selected_training_names: Optional[Sequence[str]] = None,
    transfer_scope: str = "full_policy",
) -> VLLMWeightMappingReport:
    base_prefix = str(getattr(model, "base_model_prefix", "model") or "model")
    all_training_params = dict(model.named_parameters())
    selected_set = None if selected_training_names is None else {str(name) for name in selected_training_names}
    missing_in_training = sorted(selected_set - set(all_training_params)) if selected_set is not None else []
    training_params = {
        name: tensor
        for name, tensor in all_training_params.items()
        if selected_set is None or name in selected_set
    }
    motn_roles = collect_module_aware_motn_keys(model)
    runtime = _runtime_param_map(runtime_params)
    entries: List[WeightMappingEntry] = []
    transferred_motn: List[str] = []
    shape_mismatches: List[Dict[str, Any]] = []
    dtype_mismatches: List[Dict[str, Any]] = []
    missing_in_vllm: List[str] = []
    tied_aliases: Dict[str, str] = {}

    seen_param_ids: Dict[int, str] = {}
    for training_name, tensor in training_params.items():
        transfer_name = canonical_transfer_name(training_name, base_model_prefix=base_prefix)
        tied_alias_of = seen_param_ids.get(id(tensor))
        if tied_alias_of is None:
            seen_param_ids[id(tensor)] = transfer_name
        else:
            tied_aliases[transfer_name] = tied_alias_of
        is_motn = training_name in motn_roles
        if is_motn:
            transferred_motn.append(transfer_name)
        dtype = str(tensor.dtype).replace("torch.", "")
        shape = list(tensor.shape)
        if runtime:
            runtime_shape_dtype = runtime.get(transfer_name)
            if runtime_shape_dtype is None:
                missing_in_vllm.append(transfer_name)
            else:
                runtime_shape, runtime_dtype = runtime_shape_dtype
                if runtime_shape and list(runtime_shape) != shape:
                    shape_mismatches.append(
                        {
                            "name": transfer_name,
                            "training_shape": shape,
                            "vllm_shape": list(runtime_shape),
                        }
                    )
                if runtime_dtype is not None and str(runtime_dtype) != dtype:
                    dtype_mismatches.append(
                        {
                            "name": transfer_name,
                            "training_dtype": dtype,
                            "vllm_dtype": str(runtime_dtype),
                        }
                    )
        entries.append(
            WeightMappingEntry(
                training_name=training_name,
                transfer_name=transfer_name,
                shape=shape,
                dtype=dtype,
                numel=int(tensor.numel()),
                num_bytes=_tensor_num_bytes(tensor),
                is_motn=is_motn,
                motn_role=motn_roles.get(training_name),
                tied_alias_of=tied_alias_of,
            )
        )

    required_transfer_names = [entry.transfer_name for entry in entries]
    missing_count = len(missing_in_vllm) if runtime else 0
    coverage = 1.0 if not runtime else float((len(required_transfer_names) - missing_count) / max(1, len(required_transfer_names)))
    required_motn_transfer = [
        canonical_transfer_name(key, base_model_prefix=base_prefix)
        for key in sorted(motn_roles)
    ]
    transferred_motn_set = set(transferred_motn)
    motn_covered_count = sum(1 for key in required_motn_transfer if key in transferred_motn_set and key not in missing_in_vllm)
    motn_coverage = 1.0 if not required_motn_transfer else float(motn_covered_count / len(required_motn_transfer))
    full_num_bytes = sum(_tensor_num_bytes(tensor) for tensor in all_training_params.values())
    plan_payload = [
        {
            "training_name": entry.training_name,
            "transfer_name": entry.transfer_name,
            "shape": entry.shape,
            "dtype": entry.dtype,
        }
        for entry in entries
    ]
    plan_fingerprint = hashlib.sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    selected_num_bytes = sum(entry.num_bytes for entry in entries)
    return VLLMWeightMappingReport(
        entries=entries,
        missing_in_training_model=missing_in_training,
        missing_in_vllm=sorted(set(missing_in_vllm)),
        shape_mismatches=shape_mismatches,
        dtype_mismatches=dtype_mismatches,
        tied_weight_aliases=tied_aliases,
        required_motn_keys=required_motn_transfer,
        transferred_motn_keys=sorted(transferred_motn_set),
        coverage=coverage,
        motn_coverage=motn_coverage,
        tensor_count=len(entries),
        num_bytes=selected_num_bytes,
        runtime_inspection_available=bool(runtime),
        transfer_scope=str(transfer_scope),
        transfer_plan_fingerprint=plan_fingerprint,
        full_tensor_count=len(all_training_params),
        full_num_bytes=full_num_bytes,
        payload_ratio=float(selected_num_bytes / max(1, full_num_bytes)),
    )


def validate_trainable_patch_transfer_selection(
    model: torch.nn.Module,
    *,
    transfer_training_names: Sequence[str],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Fail closed unless transfer, requires-grad, and optimizer sets match."""
    params = dict(model.named_parameters())
    configured = {str(name) for name in transfer_training_names}
    live = {name for name, param in params.items() if bool(param.requires_grad)}
    missing = sorted(configured - set(params))
    if missing:
        raise RuntimeError(f"patch transfer selection contains unknown model parameters: {missing}")
    if configured != live:
        raise RuntimeError(
            "patch transfer selection does not match requires_grad parameters: "
            f"missing_from_transfer={sorted(live - configured)}, "
            f"unexpected_in_transfer={sorted(configured - live)}"
        )

    optimizer_names: Optional[set[str]] = None
    if optimizer is not None:
        names_by_id = {id(param): name for name, param in params.items()}
        unnamed: List[int] = []
        optimizer_names = set()
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                name = names_by_id.get(id(param))
                if name is None:
                    unnamed.append(id(param))
                else:
                    optimizer_names.add(name)
        if unnamed:
            raise RuntimeError(f"optimizer contains parameters absent from model.named_parameters: {unnamed}")
        if optimizer_names != configured:
            raise RuntimeError(
                "patch transfer selection does not match optimizer parameters: "
                f"missing_from_transfer={sorted(optimizer_names - configured)}, "
                f"not_in_optimizer={sorted(configured - optimizer_names)}"
            )

    report = build_weight_mapping_report(
        model,
        selected_training_names=sorted(configured),
        transfer_scope="trainable_patch",
    )
    if not report.entries:
        raise RuntimeError("patch transfer selection is empty")
    motn_training_names = set(collect_module_aware_motn_keys(model))
    outside_patch = sorted(configured - motn_training_names)
    if outside_patch:
        raise RuntimeError(
            "trainable_patch transfer contains parameters outside module-aware MoTN patch modules: "
            f"{outside_patch}"
        )
    return {
        "transfer_scope": "trainable_patch",
        "transfer_training_name_count": len(configured),
        "optimizer_training_name_count": None if optimizer_names is None else len(optimizer_names),
        "transfer_plan_fingerprint": report.transfer_plan_fingerprint,
        "weight_transfer_tensor_count": report.tensor_count,
        "weight_transfer_bytes": report.num_bytes,
        "full_policy_tensor_count": report.full_tensor_count,
        "full_policy_bytes": report.full_num_bytes,
        "weight_transfer_payload_ratio": report.payload_ratio,
    }


def iter_transfer_tensors(report: VLLMWeightMappingReport, model: torch.nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    params = dict(model.named_parameters())
    for entry in report.entries:
        yield entry.transfer_name, params[entry.training_name]


def selected_tensor_checksums(
    report: VLLMWeightMappingReport,
    model: torch.nn.Module,
    *,
    max_tensors: int = 8,
) -> Dict[str, float]:
    params = dict(model.named_parameters())
    candidates = [entry for entry in report.entries if entry.is_motn] or list(report.entries)
    take = min(len(candidates), max(1, int(max_tensors)))
    if take <= 1:
        selected = candidates[:take]
    else:
        indices = sorted({round(index * (len(candidates) - 1) / (take - 1)) for index in range(take)})
        selected = [candidates[index] for index in indices]
    checksums: Dict[str, float] = {}
    with torch.no_grad():
        for entry in selected:
            tensor = params[entry.training_name].detach().float().cpu()
            checksums[entry.transfer_name] = float(tensor.sum().item())
    return checksums
