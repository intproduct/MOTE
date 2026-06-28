from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

try:
    from transformers.generation import GenerationMixin
except Exception:  # pragma: no cover - older transformers keeps generation on PreTrainedModel
    GenerationMixin = object

try:
    from .configuration_fitmotn import FitMoTNConfig
except Exception:  # pragma: no cover - supports direct local imports in tests
    from configuration_fitmotn import FitMoTNConfig


FITMOTN_IMPORT_ERROR = "Loading exported FitMoTN models requires the fitmotn package to be installed, e.g. pip install -e ."


def _torch_dtype_from_string(value: Any, default: torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return default
    name = str(value).replace("torch.", "").strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    return mapping.get(name, default)


def _base_config_from_dict(base_model_config: dict[str, Any]):
    model_type = base_model_config.get("model_type")
    if not model_type:
        raise ValueError("FitMoTN base_model_config must include model_type for local reconstruction")
    try:
        config_cls = CONFIG_MAPPING[str(model_type)]
    except KeyError as exc:
        raise ValueError(
            f"Installed transformers does not support base model_type={model_type!r}; "
            "install a transformers version that includes this architecture."
        ) from exc
    return config_cls.from_dict(dict(base_model_config))


class FitMoTNForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = FitMoTNConfig
    base_model_prefix = "wrapped_model"
    _tied_weights_keys = None
    supports_gradient_checkpointing = True

    def __init__(self, config: FitMoTNConfig) -> None:
        super().__init__(config)
        base_config = _base_config_from_dict(config.base_model_config)
        try:
            self.wrapped_model = AutoModelForCausalLM.from_config(base_config)
        except Exception as exc:
            model_type = getattr(base_config, "model_type", config.base_model_config.get("model_type"))
            raise ValueError(
                f"Installed transformers cannot construct AutoModelForCausalLM for base model_type={model_type!r}; "
                "install a transformers version that includes this causal LM architecture."
            ) from exc
        layers_to_patch = [int(i) for i in getattr(config, "layers_to_patch", [])]
        if layers_to_patch:
            try:
                from fitmotn.patching import patch_qwen_ffn_layers
            except Exception as exc:
                raise ImportError(FITMOTN_IMPORT_ERROR) from exc
            patch_cfg = dict(getattr(config, "fitmotn_patch_config", {}) or {})
            patch_backend = str(getattr(config, "patch_backend", None) or patch_cfg.get("patch_backend") or "motn").lower()
            patch_cfg["patch_backend"] = patch_backend
            ref = next(self.wrapped_model.parameters(), None)
            device = ref.device if ref is not None else torch.device("cpu")
            dtype = ref.dtype if ref is not None else torch.float32
            patch_cfg["dtype"] = _torch_dtype_from_string(patch_cfg.get("dtype"), dtype)
            self.wrapped_model = patch_qwen_ffn_layers(
                model=self.wrapped_model,
                layer_idxs=layers_to_patch,
                motn_cfg=patch_cfg,
                device=device,
                dtype=dtype,
            )
        self._sync_wrapped_tied_weight_keys()

    def _sync_wrapped_tied_weight_keys(self) -> None:
        keys = []
        for attr in ("_tied_weights_keys", "_dynamic_tied_weights_keys"):
            for key in getattr(self.wrapped_model, attr, None) or []:
                keys.append(f"{self.base_model_prefix}.{key}")
        self._tied_weights_keys = sorted(set(keys)) or None

    @property
    def all_tied_weights_keys(self):
        return list(self._tied_weights_keys or [])

    def forward(self, *args: Any, **kwargs: Any):
        return self.base_model(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any):
        if hasattr(self.base_model, "prepare_inputs_for_generation"):
            return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
        return super().prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        if hasattr(self.base_model, "get_output_embeddings"):
            return self.base_model.get_output_embeddings()
        return None

    def set_output_embeddings(self, new_embeddings):
        if hasattr(self.base_model, "set_output_embeddings"):
            return self.base_model.set_output_embeddings(new_embeddings)
        raise NotImplementedError("base_model does not implement set_output_embeddings")

    def tie_weights(self):
        if hasattr(self.base_model, "tie_weights"):
            return self.base_model.tie_weights()
        return None

    def resize_token_embeddings(self, new_num_tokens: int | None = None, pad_to_multiple_of: int | None = None, mean_resizing: bool = True):
        if hasattr(self.base_model, "resize_token_embeddings"):
            resized = self.base_model.resize_token_embeddings(
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                mean_resizing=mean_resizing,
            )
            if new_num_tokens is not None:
                self.config.vocab_size = int(new_num_tokens)
            return resized
        return super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)

    def save_pretrained(self, *args: Any, **kwargs: Any):
        with _skip_broken_deepspeed_probe():
            return super().save_pretrained(*args, **kwargs)


@contextmanager
def _skip_broken_deepspeed_probe():
    try:
        import accelerate.utils.imports as accelerate_imports
        import accelerate.utils.other as accelerate_other
    except Exception:
        yield
        return

    original_imports = getattr(accelerate_imports, "is_deepspeed_available", None)
    original_other = getattr(accelerate_other, "is_deepspeed_available", None)

    try:
        if original_imports is not None:
            accelerate_imports.is_deepspeed_available = lambda: False
        if original_other is not None:
            accelerate_other.is_deepspeed_available = lambda: False
        yield
    finally:
        if original_imports is not None:
            accelerate_imports.is_deepspeed_available = original_imports
        if original_other is not None:
            accelerate_other.is_deepspeed_available = original_other
