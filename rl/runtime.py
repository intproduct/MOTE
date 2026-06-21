from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..audit import json_dump, parameter_snapshot, to_jsonable
from ..eval.restore import restore_fitmotn_model
from ..gate import RouterLogits, SoftGate, TopKGate, _SoftGate, _TopKGate
from ..patching import (
    build_patch_model_config,
    patch_qwen_ffn_layers,
    resolve_layer_idxs,
    set_motn_usage_tracking,
    set_trainable_motn_only,
    set_trainable_patch_only,
)
from ..runtime import dtype_to_name, load_causal_lm_and_tokenizer


ROUTER_GATE_TYPES = (TopKGate, SoftGate, _TopKGate, _SoftGate)


@dataclass
class RLPolicyLoadInfo:
    is_fitmotn: bool
    metadata: Optional[Dict[str, Any]]
    layer_idxs: List[int]
    patch_cfg: Dict[str, Any]
    base_model_path: str
    tokenizer_path: str
    resolved_dtype: torch.dtype
    loaded_from: str


def load_policy_for_rl(fit_cfg, resume_from: str | Path | None = None, logger: logging.Logger | None = None):
    target_value = resume_from or getattr(fit_cfg.rl, "resume_from", None) or fit_cfg.model.model_path
    target = Path(str(target_value)).expanduser().resolve()
    device = torch.device(fit_cfg.model.device)
    metadata = None
    layer_idxs: List[int] = []
    patch_cfg: Dict[str, Any] = {}

    if (target / "fitmotn_state.pt").exists():
        model, tokenizer, metadata = restore_fitmotn_model(target, device=str(device))
        layer_idxs = [int(idx) for idx in metadata.get("layers_to_patch", [])]
        patch_cfg = dict(metadata.get("patch_cfg") or metadata.get("motn_cfg") or {})
        resolved_dtype = next(model.parameters()).dtype
        base_model_path = str(metadata.get("base_model_path", target))
        tokenizer_path = str(metadata.get("tokenizer_path", base_model_path))
        if logger is not None:
            logger.info("[RLLoad] restored FitMoTN checkpoint=%s base_model_path=%s layers=%s", target, base_model_path, layer_idxs)
    else:
        model, tokenizer, resolved_dtype = load_causal_lm_and_tokenizer(
            target,
            device=device,
            trust_remote_code=fit_cfg.model.trust_remote_code,
            torch_dtype=fit_cfg.model.torch_dtype,
            use_cache=False,
        )
        layer_idxs = resolve_layer_idxs(int(model.config.num_hidden_layers), fit_cfg.model.layers_to_patch)
        patch_cfg = build_patch_model_config(fit_cfg)
        model = patch_qwen_ffn_layers(model, layer_idxs, patch_cfg, device=device, dtype=torch.float32, log=logger)
        base_model_path = str(target)
        tokenizer_path = str(target)
        if logger is not None:
            logger.info("[RLLoad] loaded dense model and applied FitMoTN patch model=%s layers=%s", target, layer_idxs)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    try:
        set_motn_usage_tracking(model, False)
    except Exception as exc:
        if logger is not None:
            logger.warning("[RLUsage] failed to disable MOTE usage tracking: %s", exc)
    return model, tokenizer, RLPolicyLoadInfo(
        is_fitmotn=bool(layer_idxs),
        metadata=metadata,
        layer_idxs=list(layer_idxs),
        patch_cfg=dict(patch_cfg),
        base_model_path=str(base_model_path),
        tokenizer_path=str(tokenizer_path),
        resolved_dtype=resolved_dtype,
        loaded_from=str(target),
    )


def load_reference_for_rl(fit_cfg, logger: logging.Logger | None = None):
    if float(getattr(fit_cfg.rl, "beta", 0.0)) <= 0.0 or bool(getattr(fit_cfg.rl, "no_ref_model", True)):
        if logger is not None:
            logger.info("[RLLoad] reference model disabled because beta<=0 or no_ref_model=true")
        return None
    ref_model, _, _ = load_policy_for_rl(fit_cfg, resume_from=getattr(fit_cfg.rl, "resume_from", None), logger=logger)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    return ref_model


def _selected_names_for_ids(model: torch.nn.Module, selected_ids: set[int]) -> List[str]:
    selected_names = []
    for name, param in model.named_parameters():
        if id(param) in selected_ids:
            param.requires_grad_(True)
            selected_names.append(name)
    return selected_names


def set_trainable_mode_for_rl(model: torch.nn.Module, mode: str, logger: logging.Logger | None = None) -> Dict[str, Any]:
    requested_mode = str(mode or "patch_only").strip().lower()
    effective_mode = requested_mode
    alias_note = None
    if requested_mode == "motn_only":
        effective_mode = "patch_only"
        alias_note = "motn_only is treated as patch_only in this version"

    if effective_mode == "patch_only":
        if requested_mode == "motn_only":
            set_trainable_motn_only(model, log=logger)
        else:
            set_trainable_patch_only(model, log=logger)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        selected_ids: set[int] = set()
        if effective_mode == "all":
            if logger is not None:
                logger.warning("[RLTrainable] trainable_mode=all will train the base LLM; this is not recommended for MOTE attribution")
            for param in model.parameters():
                param.requires_grad_(True)
        elif effective_mode == "gate_only":
            for module in model.modules():
                if isinstance(module, ROUTER_GATE_TYPES):
                    selected_ids.update(id(param) for param in module.parameters(recurse=True))
            _selected_names_for_ids(model, selected_ids)
        elif effective_mode == "router_only":
            for module in model.modules():
                if isinstance(module, ROUTER_GATE_TYPES):
                    router = getattr(module, "router", None)
                    if isinstance(router, RouterLogits):
                        selected_ids.update(id(param) for param in router.parameters(recurse=True))
            _selected_names_for_ids(model, selected_ids)
        elif effective_mode == "global_only":
            for module in model.modules():
                global_block = getattr(module, "global_block", None)
                if global_block is not None:
                    selected_ids.update(id(param) for param in global_block.parameters(recurse=True))
            _selected_names_for_ids(model, selected_ids)
        else:
            raise ValueError(f"Unsupported rl.trainable_mode={requested_mode!r}")

    names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable_count == 0:
        raise RuntimeError(
            f"rl.trainable_mode={requested_mode} selected no trainable parameters. "
            "Refusing to fall back to parameter-name substring matching."
        )
    if alias_note and logger is not None:
        logger.info("[RLTrainable] %s", alias_note)
    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "alias_note": alias_note,
        "trainable_names": names,
    }


def build_trainable_summary(model: torch.nn.Module, names: Sequence[str], optimizer=None) -> Dict[str, Any]:
    snap = parameter_snapshot(model)
    trainable_tensor_count = int(sum(1 for _, param in model.named_parameters() if param.requires_grad))
    optimizer_groups = []
    if optimizer is not None:
        name_by_id = {id(param): name for name, param in model.named_parameters()}
        for idx, group in enumerate(optimizer.param_groups):
            params = list(group.get("params", []))
            optimizer_groups.append(
                {
                    "index": int(idx),
                    "lr": float(group.get("lr", 0.0)),
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                    "param_count": int(len(params)),
                    "param_numel": int(sum(param.numel() for param in params)),
                    "param_name_sample": [name_by_id.get(id(param), "<unnamed>") for param in params[:10]],
                }
            )
    return {
        **snap,
        "trainable_param_tensor_count": trainable_tensor_count,
        "trainable_name_sample": list(names[:20]),
        "trainable_name_count": int(len(names)),
        "optimizer_param_groups": optimizer_groups,
    }


def log_trainable_summary(summary: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> None:
    logger.info(
        "[RLTrainable] trainable=%s total=%s ratio=%s tensor_count=%s",
        summary.get("trainable_params"),
        summary.get("total_params"),
        summary.get("trainable_ratio"),
        summary.get("trainable_param_tensor_count"),
    )
    for name in list(summary.get("trainable_name_sample") or []):
        logger.info("[RLTrainable] name=%s", name)
    logger.info("[RLTrainable] optimizer_groups=%s", to_jsonable(summary.get("optimizer_param_groups")))
    json_dump(output_dir / "trainable_summary.json", summary)


def disable_runtime_usage_tracking(model: torch.nn.Module) -> None:
    set_motn_usage_tracking(model, False)


def enable_runtime_usage_tracking(model: torch.nn.Module) -> None:
    set_motn_usage_tracking(model, True)


def _call_usage_reset_methods(obj: Any) -> None:
    if obj is None:
        return
    for method_name in (
        "reset_runtime_usage",
        "clear_runtime_usage",
        "reset_usage",
        "clear_usage",
        "clear_runtime_usage_buffers",
        "reset_runtime_usage_buffers",
        "reset_runtime_usage_cache",
        "clear_runtime_usage_cache",
    ):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                continue


def reset_runtime_usage_buffers(model: torch.nn.Module) -> None:
    for module in model.modules():
        _call_usage_reset_methods(module)
        core = getattr(module, "core", None)
        _call_usage_reset_methods(core)
        gate = getattr(core, "gate", None) if core is not None else None
        _call_usage_reset_methods(gate)


def collect_train_forward_router_usage(model: torch.nn.Module) -> Dict[str, Any]:
    usage: Dict[str, Any] = {}
    for name, module in model.named_modules():
        core = getattr(module, "core", None)
        if core is None or not hasattr(core, "collect_runtime_usage_tensors"):
            continue
        stats = core.collect_runtime_usage_tensors()
        item = {}
        for key in ("expert_counts", "top1_counts", "topk_counts", "importance", "load", "drop_rate", "capacity", "entropy_soft", "entropy_hard"):
            value = stats.get(key)
            if isinstance(value, torch.Tensor):
                item[key] = value.detach().cpu()
            elif value is not None:
                item[key] = value
        if item:
            usage[name] = item
    return to_jsonable(usage)


def dtype_name_from_load_info(load_info: RLPolicyLoadInfo) -> str:
    return dtype_to_name(load_info.resolved_dtype)
