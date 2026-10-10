"""Experiment-local native-vLLM loader for text-only FitMoTN Qwen3.5 exports.

Version 2 normalizes the JSON-serialized patch dtype before constructing MoTN
operators.  It keeps vLLM's native Qwen3.5 attention and hybrid cache, and
replaces only the FFN layers listed in the immutable HF export.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from fitmotn.model import MOTNFFNLayer
from fitmotn.patching import PROJ_NAMES, resolve_operator_block_layout
from fitmotn.rl.vllm_qwen35_native import (
    Qwen35TextForCausalLM,
    Qwen35TextHybridConfig,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig


def _fitmotn_selected_parameter_checksums(
    worker: Any, names: list[str]
) -> dict[str, Any]:
    """Read selected parameters inside the vLLM worker process."""
    parameters = dict(worker.get_model().named_parameters())
    observed: dict[str, float] = {}
    missing: list[str] = []
    for name in names:
        key = str(name)
        tensor = parameters.get(key)
        if tensor is None:
            missing.append(key)
            continue
        observed[key] = float(tensor.detach().float().cpu().sum().item())
    return {
        "checksums": observed,
        "missing": missing,
    }


def _install_worker_checksum_rpc() -> None:
    """Install a string-addressable RPC without enabling pickle transport."""
    try:
        from vllm.v1.worker.worker_base import WorkerBase
    except ImportError:
        return
    if not hasattr(WorkerBase, "_fitmotn_selected_parameter_checksums"):
        setattr(
            WorkerBase,
            "_fitmotn_selected_parameter_checksums",
            _fitmotn_selected_parameter_checksums,
        )


_install_worker_checksum_rpc()


class _ProjectionShape:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)


def _freeze_inference_module(module: nn.Module) -> nn.Module:
    """Make custom vLLM modules safe for direct runtime weight copies."""
    module.requires_grad_(False)
    return module


def shape_only_qwen_mlp(hidden_size: int, intermediate_size: int) -> Any:
    return SimpleNamespace(
        gate_proj=_ProjectionShape(hidden_size, intermediate_size),
        up_proj=_ProjectionShape(hidden_size, intermediate_size),
        down_proj=_ProjectionShape(intermediate_size, hidden_size),
        act_fn=nn.SiLU(),
    )


def normalize_patch_config(patch_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(patch_config)
    dtype = normalized.get("dtype")
    if isinstance(dtype, str):
        dtype_name = dtype.removeprefix("torch.").lower()
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "half": torch.float16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported FitMoTN patch dtype {dtype!r}")
        normalized["dtype"] = dtype_map[dtype_name]
    return normalized


class FitMoTNNativeQwen35ForCausalLMV2(Qwen35TextForCausalLM):
    """Native Qwen3.5 attention/cache plus exact exported FitMoTN FFNs."""

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        outer_config = vllm_config.model_config.hf_config
        patch_config = normalize_patch_config(
            dict(getattr(outer_config, "fitmotn_patch_config", {}) or {})
        )
        layer_indices = tuple(
            sorted({int(i) for i in getattr(outer_config, "layers_to_patch", [])})
        )
        base_config = dict(getattr(outer_config, "base_model_config", {}) or {})

        if not base_config:
            raise ValueError("FitMoTN export is missing base_model_config")
        if base_config.get("model_type") != "qwen3_5_text":
            raise ValueError(
                "native FitMoTN adapter only supports qwen3_5_text exports; "
                f"got {base_config.get('model_type')!r}"
            )
        if patch_config.get("patch_backend", "motn") != "motn":
            raise ValueError(
                "native FitMoTN adapter only supports patch_backend='motn'"
            )

        native_config = Qwen3_5TextConfig(**base_config)
        vllm_config.model_config.hf_text_config = native_config
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        hidden_size = int(native_config.hidden_size)
        intermediate_size = int(native_config.intermediate_size)
        device = next(self.parameters()).device
        model_dtype = vllm_config.model_config.dtype
        layers = self.model.layers
        if any(i < 0 or i >= len(layers) for i in layer_indices):
            raise ValueError(
                f"layers_to_patch={layer_indices!r} is invalid for {len(layers)} layers"
            )

        for layer_index in layer_indices:
            replacement = MOTNFFNLayer(
                shape_only_qwen_mlp(hidden_size, intermediate_size),
                patch_config,
                device=device,
                dtype=model_dtype,
                layer_idx=layer_index,
            ).to(device)
            replacement.layer_idx = layer_index
            replacement.patch_backend = "motn"
            replacement.fitmotn_block_layout = {}
            for projection_name in PROJ_NAMES:
                projection = getattr(replacement, projection_name)
                layout = resolve_operator_block_layout(projection, patch_config)
                replacement.fitmotn_block_layout[projection_name] = layout
                projection.fitmotn_block_layout = layout
            replacement.fitmotn_expert_warmup_state = {"enabled": False}
            # vLLM actors are inference-only. Custom torch modules retain
            # requires_grad=True by default, unlike vLLM's native parameters.
            # vLLM 0.19's runtime-parameter update path performs a direct
            # ``param.copy_(weight)``; keeping a patch parameter as a trainable
            # leaf makes that correct in-place copy fail under autograd.
            _freeze_inference_module(replacement)
            layers[layer_index].mlp = replacement

        self._fitmotn_weight_prefixes = tuple(
            f"model.layers.{i}.mlp." for i in layer_indices
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        parameters = dict(self.named_parameters())
        custom_loaded: set[str] = set()

        def native_weights():
            for name, loaded_weight in weights:
                if name.startswith(self._fitmotn_weight_prefixes):
                    if name not in parameters:
                        raise ValueError(
                            f"exported FitMoTN parameter {name!r} has no native target"
                        )
                    default_weight_loader(parameters[name], loaded_weight)
                    custom_loaded.add(name)
                else:
                    yield name, loaded_weight

        native_loaded = super().load_weights(native_weights())
        expected_custom = {
            name
            for name in parameters
            if name.startswith(self._fitmotn_weight_prefixes)
        }
        missing_custom = expected_custom - custom_loaded
        if missing_custom:
            preview = sorted(missing_custom)[:8]
            raise ValueError(
                f"FitMoTN export omitted {len(missing_custom)} native parameters; "
                f"first_missing={preview}"
            )
        return set(native_loaded) | custom_loaded


def register() -> None:
    ModelRegistry.register_model(
        "FitMoTNForCausalLM",
        "fitmotn.rl.vllm_qwen35_motn_native:FitMoTNNativeQwen35ForCausalLMV2",
    )
    MODELS_CONFIG_MAP["FitMoTNForCausalLM"] = Qwen35TextHybridConfig
