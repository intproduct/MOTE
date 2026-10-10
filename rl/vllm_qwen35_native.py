from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.models.config import (
    HybridAttentionMambaModelConfig,
    MODELS_CONFIG_MAP,
    Qwen3_5ForConditionalGenerationConfig,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.transformers import TransformersForCausalLM


_REGISTERED = False


class Qwen35TextForCausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    """Complete the pure-text Qwen3.5 native interface missing in vLLM 0.19."""

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: Any):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: Any):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()


    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError("pure-text Qwen3.5 does not accept multimodal features")
        positions = torch.arange(len(input_tokens), dtype=torch.long).repeat(3, 1)
        return positions, 0


class FitMoTNQwen35ForCausalLM(
    TransformersForCausalLM, HasInnerState, IsHybrid, SupportsMRoPE
):
    """Add the pure-text M-RoPE interface to exported FitMoTN policies."""

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: Any):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: Any):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError(
                "pure-text FitMoTN Qwen3.5 does not accept multimodal features"
            )
        positions = torch.arange(len(input_tokens), dtype=torch.long).repeat(3, 1)
        return positions, 0


class Qwen35TextHybridConfig:
    """Apply the dtype and hybrid-page updates used by native Qwen3.5."""

    @classmethod
    def verify_and_update_model_config(cls, model_config: Any) -> None:
        return None

    @classmethod
    def verify_and_update_config(cls, vllm_config: Any) -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)


def register_qwen35_text_native() -> None:
    """Register a spawn-safe native implementation for Qwen3_5ForCausalLM."""

    global _REGISTERED
    if _REGISTERED:
        return
    ModelRegistry.register_model(
        "Qwen3_5ForCausalLM",
        "fitmotn.rl.vllm_qwen35_native:Qwen35TextForCausalLM",
    )
    ModelRegistry.register_model(
        "FitMoTNForCausalLM",
        "fitmotn.rl.vllm_qwen35_motn_native:FitMoTNNativeQwen35ForCausalLMV2",
    )
    MODELS_CONFIG_MAP["Qwen3_5ForCausalLM"] = Qwen35TextHybridConfig
    _REGISTERED = True
