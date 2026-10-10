from __future__ import annotations

import torch

from MOTE.rl import vllm_qwen35_native as native
from MOTE.rl.vllm_qwen35_motn_native import _freeze_inference_module
from vllm.model_executor.models.interfaces import (
    has_inner_state,
    is_hybrid,
    supports_mrope,
)


def test_qwen35_native_registration_is_idempotent():
    native.register_qwen35_text_native()
    registered = native.ModelRegistry.models["Qwen3_5ForCausalLM"]
    registered_fitmotn = native.ModelRegistry.models["FitMoTNForCausalLM"]
    native.register_qwen35_text_native()
    assert native.ModelRegistry.models["Qwen3_5ForCausalLM"] is registered
    assert native.ModelRegistry.models["FitMoTNForCausalLM"] is registered_fitmotn
    assert native.MODELS_CONFIG_MAP["Qwen3_5ForCausalLM"] is native.Qwen35TextHybridConfig


def test_native_motn_patch_is_frozen_for_runtime_weight_copies():
    module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    assert any(parameter.requires_grad for parameter in module.parameters())
    assert _freeze_inference_module(module) is module
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_qwen35_native_exposes_hybrid_state_and_mrope_interfaces():
    cls = native.Qwen35TextForCausalLM
    assert has_inner_state(cls)
    assert is_hybrid(cls)
    assert supports_mrope(cls)
    assert hasattr(cls, "get_mamba_state_dtype_from_config")
    assert hasattr(cls, "get_mamba_state_shape_from_config")
    assert hasattr(cls, "get_mamba_state_copy_func")


def test_fitmotn_transformers_backend_exposes_text_mrope_interface():
    cls = native.FitMoTNQwen35ForCausalLM
    assert supports_mrope(cls)
    assert is_hybrid(cls)
    assert has_inner_state(cls)
    assert hasattr(cls, "get_mamba_state_dtype_from_config")
    assert hasattr(cls, "get_mamba_state_shape_from_config")
    assert hasattr(cls, "get_mamba_state_copy_func")
    model = cls.__new__(cls)
    positions, delta = model.get_mrope_input_positions([21, 22, 23], [])
    expected = torch.arange(3, dtype=torch.long)
    assert positions.shape == (3, 3)
    assert torch.equal(positions[0], expected)
    assert torch.equal(positions[0], positions[1])
    assert torch.equal(positions[1], positions[2])
    assert delta == 0


def test_qwen35_text_mrope_positions_are_three_identical_text_axes():
    model = native.Qwen35TextForCausalLM.__new__(native.Qwen35TextForCausalLM)
    positions, delta = model.get_mrope_input_positions([11, 12, 13, 14], [])
    expected = torch.arange(4, dtype=torch.long)
    assert positions.shape == (3, 4)
    assert torch.equal(positions[0], expected)
    assert torch.equal(positions[0], positions[1])
    assert torch.equal(positions[1], positions[2])
    assert delta == 0


def test_qwen35_text_mrope_rejects_multimodal_features():
    model = native.Qwen35TextForCausalLM.__new__(native.Qwen35TextForCausalLM)
    try:
        model.get_mrope_input_positions([11], [object()])
    except ValueError as exc:
        assert "pure-text" in str(exc)
    else:
        raise AssertionError("pure-text compatibility model accepted multimodal input")
