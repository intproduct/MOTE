from __future__ import annotations

import importlib.resources as resources
import builtins

import pytest

from fitmotn.export import hf_export
from fitmotn.export.remote_code.configuration_fitmotn import FitMoTNConfig
from fitmotn.export.roundtrip import validate_hf_roundtrip


def _tiny_gpt2_config(vocab_size: int = 23, hidden_size: int = 8):
    from transformers import GPT2Config

    base = GPT2Config(
        vocab_size=vocab_size,
        n_positions=16,
        n_ctx=16,
        n_embd=hidden_size,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=2,
    )
    base.use_cache = True
    return base


def _fitmotn_config_from_base(base, **kwargs):
    data = dict(
        base_model_name_or_path="tiny-local",
        base_model_config=base.to_dict(),
        base_model_architectures=list(base.to_dict().get("architectures") or []),
        tokenizer_source="tiny-local",
        fitmotn_patch_config={},
        layers_to_patch=[],
        patch_backend="motn",
        use_cache=True,
    )
    data.update(kwargs)
    return FitMoTNConfig(**data)


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
    assert data["auto_map"]["AutoModel"] == "modeling_fitmotn.FitMoTNModel"
    assert data["auto_map"]["AutoModelForCausalLM"] == "modeling_fitmotn.FitMoTNForCausalLM"


def test_fitmotn_model_returns_hidden_states_not_logits(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    torch = pytest.importorskip("torch")

    from transformers import AutoConfig, AutoModel
    import transformers.dynamic_module_utils as dynamic_module_utils

    base = _tiny_gpt2_config(vocab_size=23, hidden_size=8)
    config = _fitmotn_config_from_base(base)
    config.save_pretrained(tmp_path)
    package = "fitmotn.export.remote_code"
    for filename in ("configuration_fitmotn.py", "modeling_fitmotn.py"):
        source = resources.files(package).joinpath(filename)
        (tmp_path / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "hf_modules_cache"))
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "hf_modules_cache"))

    loaded_config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)
    model = AutoModel.from_config(loaded_config, trust_remote_code=True)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    out = model(input_ids=input_ids, return_dict=False)

    assert isinstance(out, tuple)
    assert out[0].shape[:2] == input_ids.shape
    assert out[0].shape[-1] == 8
    assert out[0].shape[-1] != 23


def test_causal_lm_state_dict_uses_canonical_model_prefix():
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    pytest.importorskip("torch")

    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM

    config = _fitmotn_config_from_base(_tiny_gpt2_config())
    model = FitMoTNForCausalLM(config)
    keys = set(model.state_dict().keys())

    assert any(k.startswith("model.") for k in keys)
    assert any(k.startswith("lm_head.") for k in keys)
    assert not any(k.startswith("model.model.") for k in keys)


def test_tiny_noop_hf_roundtrip(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    pytest.importorskip("torch")

    from transformers import AutoModel
    import transformers.dynamic_module_utils as dynamic_module_utils

    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM

    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "hf_modules_cache"))
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "hf_modules_cache"))
    config = _fitmotn_config_from_base(_tiny_gpt2_config(vocab_size=16, hidden_size=8))
    model = FitMoTNForCausalLM(config)
    assert model.base_model is model.model
    assert "lm_head.weight" in model.all_tied_weights_keys
    model.save_pretrained(tmp_path, safe_serialization=False)
    package = "fitmotn.export.remote_code"
    for filename in ("configuration_fitmotn.py", "modeling_fitmotn.py"):
        source = resources.files(package).joinpath(filename)
        (tmp_path / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    result = validate_hf_roundtrip(tmp_path, device="cpu", torch_dtype="float32")

    assert result.ok, result.errors
    loaded_decoder = AutoModel.from_pretrained(tmp_path, trust_remote_code=True)
    assert loaded_decoder.__class__.__name__ == "FitMoTNModel"


def test_fitmotn_model_loads_legacy_nested_layout_and_ignores_lm_head():
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    pytest.importorskip("torch")

    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM, FitMoTNModel

    config = _fitmotn_config_from_base(_tiny_gpt2_config())
    lm = FitMoTNForCausalLM(config)
    legacy_state = {}
    for key, value in lm.state_dict().items():
        if key.startswith("model."):
            legacy_state["model.model." + key[len("model.") :]] = value
        else:
            legacy_state[key] = value

    decoder = FitMoTNModel(config)
    result = decoder.load_state_dict(legacy_state, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_meta_device_construction_unpatched():
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    torch = pytest.importorskip("torch")

    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNModel

    config = _fitmotn_config_from_base(_tiny_gpt2_config())
    with torch.device("meta"):
        model = FitMoTNModel(config)
    assert model is not None


def test_meta_device_construction_patched_qwen2():
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    torch = pytest.importorskip("torch")

    try:
        from transformers import Qwen2Config
    except Exception:
        pytest.skip("transformers does not provide Qwen2Config")
    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNModel

    base = Qwen2Config(
        vocab_size=23,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    config = _fitmotn_config_from_base(
        base,
        fitmotn_patch_config={
            "patch_backend": "motn",
            "gate_type": "topk",
            "E": 2,
            "topk": 1,
            "d": 2,
            "k_in": 2,
            "block_init_mode": "base_stats_normal",
            "dtype": "float32",
        },
        layers_to_patch=[0],
    )

    with torch.device("meta"):
        model = FitMoTNModel(config)

    keys = set(model.state_dict().keys())
    assert any("blocks" in key for key in keys)
    assert any("router" in key or "gate" in key for key in keys)


def test_optional_vllm_smoke_loads_tiny_patched_export(tmp_path, monkeypatch):
    vllm = pytest.importorskip("vllm")
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("vLLM smoke test requires CUDA in this test environment")
    try:
        from transformers import PreTrainedTokenizerFast, Qwen2Config
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
    except Exception as exc:
        pytest.skip(f"tiny tokenizer/Qwen2 dependencies unavailable: {exc}")
    import transformers.dynamic_module_utils as dynamic_module_utils

    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM

    base = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config = _fitmotn_config_from_base(
        base,
        fitmotn_patch_config={
            "patch_backend": "motn",
            "gate_type": "topk",
            "E": 2,
            "topk": 1,
            "d": 2,
            "k_in": 2,
            "dtype": "float32",
        },
        layers_to_patch=[0],
    )
    model = FitMoTNForCausalLM(config)
    model.save_pretrained(tmp_path, safe_serialization=False)
    package = "fitmotn.export.remote_code"
    for filename in ("configuration_fitmotn.py", "modeling_fitmotn.py"):
        source = resources.files(package).joinpath(filename)
        (tmp_path / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "1": 4, "+": 5, "=": 6}
    tokenizer_impl = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(tmp_path)

    keys = set(model.state_dict().keys())
    assert any(any(marker in key for marker in ("router", "blocks", "gate", "global_block")) for key in keys)

    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "hf_modules_cache"))
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "hf_modules_cache"))

    try:
        llm = vllm.LLM(
            model=str(tmp_path),
            tokenizer=str(tmp_path),
            trust_remote_code=True,
            model_impl="transformers",
            enforce_eager=True,
        )
    except Exception as exc:
        if "NVMLError" in type(exc).__name__ or "NVML" in str(exc) or "blocked the request" in str(exc):
            pytest.skip(f"vLLM could not initialize NVML in this environment: {exc}")
        raise
    outputs = llm.generate(["1+1="], vllm.SamplingParams(temperature=0.0, max_tokens=8))
    assert outputs


def test_nonempty_layers_to_patch_calls_patching_helper(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not transformers.utils.is_torch_available():
        pytest.skip("transformers reports torch unavailable in this environment")
    pytest.importorskip("torch")

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

    base = _tiny_gpt2_config(vocab_size=16, hidden_size=8)
    config = _fitmotn_config_from_base(
        base,
        fitmotn_patch_config={"patch_backend": "motn", "E": 2, "dtype": "float32"},
        layers_to_patch=[0],
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
