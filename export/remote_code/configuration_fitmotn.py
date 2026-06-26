from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


GENERATION_CRITICAL_FIELDS = (
    "vocab_size",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "decoder_start_token_id",
    "is_encoder_decoder",
    "tie_word_embeddings",
    "torch_dtype",
    "dtype",
    "use_cache",
    "max_position_embeddings",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "sliding_window",
    "rope_theta",
    "rope_scaling",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    text = str(value)
    if text.startswith("torch."):
        return text.replace("torch.", "")
    return value


class FitMoTNConfig(PretrainedConfig):
    model_type = "fitmotn"

    def __init__(
        self,
        base_model_name_or_path: str | None = None,
        base_model_config: dict[str, Any] | None = None,
        base_model_architectures: list[str] | None = None,
        tokenizer_source: str | None = None,
        fitmotn_patch_config: dict[str, Any] | None = None,
        layers_to_patch: list[int] | None = None,
        patch_backend: str = "motn",
        fitmotn_export_version: int = 1,
        format_version: int = 1,
        original_model_type: str | None = None,
        exported_code_ready: bool = True,
        auto_map_ready: bool = True,
        **kwargs: Any,
    ) -> None:
        base_cfg = _json_safe(dict(base_model_config or {}))
        patch_cfg = _json_safe(dict(fitmotn_patch_config or {}))
        for field in GENERATION_CRITICAL_FIELDS:
            if field not in kwargs and field in base_cfg:
                kwargs[field] = _json_safe(base_cfg[field])
        kwargs.setdefault(
            "auto_map",
            {
                "AutoConfig": "configuration_fitmotn.FitMoTNConfig",
                "AutoModelForCausalLM": "modeling_fitmotn.FitMoTNForCausalLM",
            },
        )
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_config = base_cfg
        self.base_model_architectures = list(base_model_architectures or base_cfg.get("architectures") or [])
        self.tokenizer_source = tokenizer_source
        self.fitmotn_patch_config = patch_cfg
        self.layers_to_patch = [int(i) for i in (layers_to_patch or [])]
        self.patch_backend = str(patch_backend or patch_cfg.get("patch_backend") or "motn").lower()
        self.fitmotn_export_version = int(fitmotn_export_version)
        self.format_version = int(format_version)
        self.original_model_type = original_model_type or base_cfg.get("model_type")
        self.exported_code_ready = bool(exported_code_ready)
        self.auto_map_ready = bool(auto_map_ready)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if "torch_dtype" not in data and "dtype" in data:
            data["torch_dtype"] = _json_safe(data["dtype"])
        return _json_safe(data)
