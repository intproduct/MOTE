from __future__ import annotations

import importlib.resources as resources
import json

import pytest

from fitmotn.diagnostics.vllm_export_preflight import (
    apply_vllm_transformers_weight_mapping,
    collect_safetensor_shapes,
    validate_vllm_export_preflight,
)
from fitmotn.export.remote_code.configuration_fitmotn import FitMoTNConfig


def _build_tiny_export(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable")
    from transformers import GPT2Config
    import transformers.dynamic_module_utils as dynamic_module_utils

    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM

    base = GPT2Config(
        vocab_size=16,
        n_positions=16,
        n_ctx=16,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )
    config = FitMoTNConfig(
        base_model_name_or_path="tiny-local",
        base_model_config=base.to_dict(),
        tokenizer_source="tiny-local",
        fitmotn_patch_config={},
        layers_to_patch=[],
        patch_backend="motn",
        architectures=["FitMoTNForCausalLM"],
        use_cache=True,
    )
    model = FitMoTNForCausalLM(config)
    model.save_pretrained(tmp_path, safe_serialization=True)
    package = "fitmotn.export.remote_code"
    for filename in ("configuration_fitmotn.py", "modeling_fitmotn.py"):
        source = resources.files(package).joinpath(filename)
        (tmp_path / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "fitmotn_export_manifest.json").write_text(
        json.dumps(
            {
                "export_stage": "hf_roundtrip",
                "hf_roundtrip_ready": True,
                "vllm_ready": True,
                "vllm_model_impl": "transformers",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "fitmotn_export_config.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / "hf_modules_cache"
    monkeypatch.setenv("HF_MODULES_CACHE", str(cache))
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(cache))
    return resources.files(package).joinpath("modeling_fitmotn.py")


def test_preflight_accepts_complete_tiny_export(tmp_path, monkeypatch):
    expected_code = _build_tiny_export(tmp_path, monkeypatch)
    report = validate_vllm_export_preflight(
        tmp_path,
        expected_remote_code=expected_code,
        require_vllm=False,
    )
    assert report["ok"], report
    assert report["checks"]["weights"]["has_lm_head"] is True
    assert report["checks"]["vllm_weight_contract_tp1"]["missing_target_count"] == 0
    assert report["checks"]["vllm_weight_contract_tp1"]["unexpected_mapped_count"] == 0


def test_preflight_rejects_stale_remote_code(tmp_path, monkeypatch):
    expected_code = _build_tiny_export(tmp_path, monkeypatch)
    (tmp_path / "modeling_fitmotn.py").write_text("# stale\n", encoding="utf-8")
    report = validate_vllm_export_preflight(
        tmp_path,
        expected_remote_code=expected_code,
        require_vllm=False,
    )
    assert report["ok"] is False
    assert any("stale" in error for error in report["errors"])


def test_collect_safetensors_includes_explicit_tied_head(tmp_path, monkeypatch):
    _build_tiny_export(tmp_path, monkeypatch)
    shapes = collect_safetensor_shapes(tmp_path)
    assert shapes["model.wte.weight"] == (16, 8)
    assert shapes["lm_head.weight"] == (16, 8)


def test_preflight_rejects_checkpoint_without_explicit_lm_head(tmp_path, monkeypatch):
    expected_code = _build_tiny_export(tmp_path, monkeypatch)
    from safetensors.torch import load_file, save_file

    weight_path = tmp_path / "model.safetensors"
    weights = dict(load_file(weight_path))
    weights.pop("lm_head.weight")
    save_file(weights, weight_path)

    report = validate_vllm_export_preflight(
        tmp_path,
        expected_remote_code=expected_code,
        require_vllm=False,
    )
    assert report["ok"] is False
    assert any("lm_head.weight" in error for error in report["errors"])
