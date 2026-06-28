from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
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


def _propagate_runtime_config(base_config: Any, config: FitMoTNConfig) -> Any:
    for attr in ("_attn_implementation", "attn_implementation"):
        value = getattr(config, attr, None)
        if value is not None:
            try:
                setattr(base_config, attr, value)
            except Exception:
                pass
    if hasattr(config, "use_cache"):
        try:
            base_config.use_cache = bool(getattr(config, "use_cache"))
        except Exception:
            pass
    dtype_value = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if dtype_value is not None:
        try:
            base_config.torch_dtype = dtype_value
        except Exception:
            pass
    return base_config


def _build_patched_decoder(config: FitMoTNConfig) -> nn.Module:
    base_config = _propagate_runtime_config(_base_config_from_dict(config.base_model_config), config)
    try:
        decoder = AutoModel.from_config(base_config)
    except Exception as exc:
        model_type = getattr(base_config, "model_type", config.base_model_config.get("model_type"))
        raise ValueError(
            f"Installed transformers cannot construct AutoModel for base model_type={model_type!r}; "
            "install a transformers version that includes this decoder architecture."
        ) from exc

    layers_to_patch = [int(i) for i in getattr(config, "layers_to_patch", [])]
    if layers_to_patch:
        try:
            from fitmotn.patching import patch_qwen_ffn_layers, set_motn_usage_tracking
        except Exception as exc:
            raise ImportError(FITMOTN_IMPORT_ERROR) from exc
        patch_cfg = dict(getattr(config, "fitmotn_patch_config", {}) or {})
        patch_backend = str(getattr(config, "patch_backend", None) or patch_cfg.get("patch_backend") or "motn").lower()
        patch_cfg["patch_backend"] = patch_backend
        ref = next(decoder.parameters(), None)
        device = ref.device if ref is not None else torch.device("cpu")
        dtype = ref.dtype if ref is not None else torch.float32
        patch_cfg["dtype"] = _torch_dtype_from_string(patch_cfg.get("dtype"), dtype)
        decoder = patch_qwen_ffn_layers(
            model=decoder,
            layer_idxs=layers_to_patch,
            motn_cfg=patch_cfg,
            device=device,
            dtype=dtype,
        )
        if bool(getattr(config, "fitmotn_disable_usage_tracking", True)):
            set_motn_usage_tracking(decoder, False)
    return decoder


def _build_lm_head(config: FitMoTNConfig, decoder: nn.Module) -> nn.Module:
    base_config = _propagate_runtime_config(_base_config_from_dict(config.base_model_config), config)
    try:
        causal_lm = AutoModelForCausalLM.from_config(base_config)
    except Exception as exc:
        model_type = getattr(base_config, "model_type", config.base_model_config.get("model_type"))
        raise ValueError(
            f"Installed transformers cannot construct AutoModelForCausalLM for base model_type={model_type!r}; "
            "install a transformers version that includes this causal LM architecture."
        ) from exc
    head = causal_lm.get_output_embeddings()
    if head is None:
        hidden_size = int(getattr(config, "hidden_size", getattr(base_config, "hidden_size", 0)) or 0)
        vocab_size = int(getattr(config, "vocab_size", getattr(base_config, "vocab_size", 0)) or 0)
        if hidden_size <= 0 or vocab_size <= 0:
            raise ValueError("Could not infer hidden_size/vocab_size for FitMoTN lm_head")
        head = nn.Linear(hidden_size, vocab_size, bias=False)
    return head


def _remap_legacy_nested_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    remapped = dict(state_dict)
    for key, value in list(state_dict.items()):
        if key.startswith("model.model."):
            new_key = "model." + key[len("model.model.") :]
            if new_key not in remapped:
                remapped[new_key] = value
            remapped.pop(key, None)
    return remapped


class FitMoTNModel(PreTrainedModel):
    """vLLM-facing AutoModel wrapper.

    This class returns decoder hidden states. It intentionally has no lm_head;
    vLLM's Transformers backend loads this AutoModel class and owns the runtime
    language-model head itself.
    """

    config_class = FitMoTNConfig
    base_model_prefix = "model"
    _supports_attention_backend = True
    _keys_to_ignore_on_load_unexpected = [r"lm_head\..*"]
    supports_gradient_checkpointing = True

    def __init__(self, config: FitMoTNConfig) -> None:
        super().__init__(config)
        self.model = _build_patched_decoder(config)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = _remap_legacy_nested_keys(dict(state_dict))
        if strict:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lm_head.")}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def get_decoder(self):
        if hasattr(self.model, "get_decoder"):
            return self.model.get_decoder()
        return self.model


class FitMoTNForCausalLM(PreTrainedModel, GenerationMixin):
    """HF generation wrapper around the same decoder used by FitMoTNModel."""

    config_class = FitMoTNConfig
    base_model_prefix = "model"
    _tied_weights_keys = None
    supports_gradient_checkpointing = True

    def __init__(self, config: FitMoTNConfig) -> None:
        super().__init__(config)
        self.model = _build_patched_decoder(config)
        self.lm_head = _build_lm_head(config, self.model)
        self._sync_tied_weight_keys()
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.tie_weights()

    def _sync_tied_weight_keys(self) -> None:
        keys = []
        for attr in ("_tied_weights_keys", "_dynamic_tied_weights_keys"):
            for key in getattr(self.model, attr, None) or []:
                keys.append(f"model.{key}")
        if bool(getattr(self.config, "tie_word_embeddings", False)):
            keys.append("lm_head.weight")
        self._tied_weights_keys = sorted(set(keys)) or None

    @property
    def all_tied_weights_keys(self):
        return list(self._tied_weights_keys or [])

    def forward(self, *args: Any, labels: torch.Tensor | None = None, return_dict: bool | None = None, **kwargs: Any):
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)
        decoder_outputs = self.model(*args, return_dict=return_dict, **kwargs)
        hidden_states = decoder_outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (logits,) + tuple(decoder_outputs[1:])
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(decoder_outputs, "past_key_values", None),
            hidden_states=getattr(decoder_outputs, "hidden_states", None),
            attentions=getattr(decoder_outputs, "attentions", None),
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids}
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        if "position_ids" in kwargs:
            model_inputs["position_ids"] = kwargs["position_ids"]
        return model_inputs

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        if hasattr(self.model, "get_decoder"):
            return self.model.get_decoder()
        return self.model

    def tie_weights(self):
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()
        if output_embeddings is not None and input_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, input_embeddings)
        return None

    def resize_token_embeddings(self, new_num_tokens: int | None = None, pad_to_multiple_of: int | None = None, mean_resizing: bool = True):
        resized = super().resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )
        if new_num_tokens is not None:
            self.config.vocab_size = int(new_num_tokens)
        return resized

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
