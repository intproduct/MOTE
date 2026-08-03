from __future__ import annotations

from pathlib import Path

import torch as tc

from ..checkpointing import get_restore_state_dict, load_fitmotn_metadata
from ..gate import normalize_legacy_gate_state_dict_for_model
from ..patching import patch_qwen_ffn_layers
from ..runtime import load_causal_lm_and_tokenizer


def restore_fitmotn_model(
    ckpt_dir: str | Path,
    device: str = "cuda:0",
    *,
    trust_remote_code: bool | None = None,
    torch_dtype: str | None = None,
    use_cache: bool = True,
):
    ckpt_path = Path(ckpt_dir).resolve()
    metadata = load_fitmotn_metadata(ckpt_path)
    base_model_path = metadata["base_model_path"]
    layer_idxs = list(metadata["layers_to_patch"])
    patch_cfg = dict(metadata.get("patch_cfg") or metadata.get("motn_cfg") or {})
    patch_backend = str(metadata.get("patch_backend", patch_cfg.get("patch_backend", "motn")) or "motn").lower()
    patch_cfg["patch_backend"] = patch_backend
    state_dict = get_restore_state_dict(metadata)
    fit_cfg = metadata.get("fit_cfg") or {}
    model_cfg = fit_cfg.get("model") if isinstance(fit_cfg, dict) else getattr(fit_cfg, "model", {})
    resolved_torch_dtype = torch_dtype if torch_dtype is not None else metadata.get("resolved_model_dtype", None)
    if resolved_torch_dtype is None:
        resolved_torch_dtype = getattr(model_cfg, "torch_dtype", None) if not isinstance(model_cfg, dict) else model_cfg.get("torch_dtype", "auto")
    model, tokenizer, ref_dtype = load_causal_lm_and_tokenizer(
        base_model_path,
        device=tc.device(device),
        trust_remote_code=bool(metadata.get("trust_remote_code", True) if trust_remote_code is None else trust_remote_code),
        torch_dtype=resolved_torch_dtype,
        use_cache=use_cache,
    )
    if "dtype" not in patch_cfg:
        patch_cfg["dtype"] = ref_dtype
    model = patch_qwen_ffn_layers(model=model, layer_idxs=layer_idxs, motn_cfg=patch_cfg, device=tc.device(device), dtype=ref_dtype)
    state_dict = normalize_legacy_gate_state_dict_for_model(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if patch_backend in {"sparse_mixt", "mixed_mixt"}:
        prefixes = tuple(
            prefix
            for idx in layer_idxs
            for prefix in (
                f"model.layers.{int(idx)}.mlp.",
                f"model.language_model.layers.{int(idx)}.mlp.",
                f"language_model.layers.{int(idx)}.mlp.",
                f"layers.{int(idx)}.mlp.",
            )
        )
        missing_patch = [key for key in incompatible.missing_keys if key.startswith(prefixes)]
        if missing_patch or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{patch_backend} checkpoint does not match the declared backend layout: "
                f"missing_patch={missing_patch[:20]}, unexpected={list(incompatible.unexpected_keys)[:20]}"
            )
    model.eval()
    return model, tokenizer, metadata
