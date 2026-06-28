from __future__ import annotations

import importlib.resources as resources
import builtins

import pytest

from fitmotn.export import hf_export
from fitmotn.export.remote_code.configuration_fitmotn import FitMoTNConfig
from fitmotn.export.roundtrip import validate_hf_roundtrip


def test_fitmotn_config_mirrors_generation_fields():
    config = FitMoTNConfig(
        base_model_name_or_path="tiny",
        base_model_config={
            "model_type": "gpt2",
            "vocab_size": 17,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "is_encoder_decoder": False,
            "tie_word_embeddings": True,
            "torch_dtype": "torch.float32",
            "use_cache": True,
            "n_embd": 8,
        },
        fitmotn_patch_config={"patch_backend": "motn", "dtype": "torch.float32"},
        layers_to_patch=[],
    )

    data = config.to_dict()

    assert config.model_type == "fitmotn"
    assert data["vocab_size"] == 17
    assert data["bos_token_id"] == 1
    assert data["eos_token_id"] == 2
    assert data["pad_token_id"] == 0
    assert data["tie_word_embeddings"] is True
    assert data["torch_dtype"] == "float32"
    assert data["fitmotn_patch_config"]["dtype"] == "float32"
    assert data["auto_map"]["AutoModelForCausalLM"] == "modeling_fitmotn.FitMoTNForCausalLM"


def test_tiny_noop_hf_roundtrip(tmp_path):
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    pytest.importorskip("torch")

    from transformers import GPT2Config

    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM

    base = GPT2Config(vocab_size=16, n_positions=16, n_ctx=16, n_embd=8, n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2)
    base.use_cache = True
    config = FitMoTNConfig(
        base_model_name_or_path="tiny-gpt2-local",
        base_model_config=base.to_dict(),
        base_model_architectures=["GPT2LMHeadModel"],
        tokenizer_source="tiny-gpt2-local",
        fitmotn_patch_config={},
        layers_to_patch=[],
        patch_backend="motn",
        use_cache=True,
    )
    model = FitMoTNForCausalLM(config)
    assert model.base_model is model.wrapped_model
    assert "wrapped_model.lm_head.weight" in model.all_tied_weights_keys
    model.save_pretrained(tmp_path, safe_serialization=False)
    package = "fitmotn.export.remote_code"
    for filename in ("configuration_fitmotn.py", "modeling_fitmotn.py"):
        source = resources.files(package).joinpath(filename)
        (tmp_path / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    result = validate_hf_roundtrip(tmp_path, device="cpu", torch_dtype="float32")

    assert result.ok, result.errors


def test_nonempty_layers_to_patch_calls_patching_helper(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    pytest.importorskip("torch")

    from transformers import GPT2Config

    import fitmotn.patching as patching
    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM

    calls = {}

    def fake_patch_qwen_ffn_layers(*, model, layer_idxs, motn_cfg, device, dtype, log=None):
        calls["layer_idxs"] = list(layer_idxs)
        calls["motn_cfg"] = dict(motn_cfg)
        calls["device"] = str(device)
        calls["dtype"] = str(dtype)
        return model

    monkeypatch.setattr(patching, "patch_qwen_ffn_layers", fake_patch_qwen_ffn_layers)

    base = GPT2Config(vocab_size=16, n_positions=16, n_ctx=16, n_embd=8, n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2)
    config = FitMoTNConfig(
        base_model_name_or_path="tiny-gpt2-local",
        base_model_config=base.to_dict(),
        fitmotn_patch_config={"patch_backend": "motn", "E": 2, "dtype": "float32"},
        layers_to_patch=[0],
        patch_backend="motn",
    )

    FitMoTNForCausalLM(config)

    assert calls["layer_idxs"] == [0]
    assert calls["motn_cfg"]["patch_backend"] == "motn"
    assert calls["motn_cfg"]["E"] == 2


def test_safe_serialization_requires_safetensors(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "safetensors":
            raise ImportError("missing safetensors")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="safe_serialization true requires safetensors"):
        hf_export._require_safetensors_if_needed(True)

    hf_export._require_safetensors_if_needed(False)
