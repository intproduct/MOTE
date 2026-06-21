from __future__ import annotations

from pathlib import Path

import torch as tc

from ..checkpointing import get_restore_state_dict, load_fitmotn_metadata
from ..gate import normalize_legacy_gate_state_dict_for_model
from ..patching import patch_qwen_ffn_layers
from ..runtime import load_causal_lm_and_tokenizer


def restore_fitmotn_model(ckpt_dir: str | Path, device: str = "cuda:0"):
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
    torch_dtype = metadata.get("resolved_model_dtype", None)
    if torch_dtype is None:
        torch_dtype = getattr(model_cfg, "torch_dtype", None) if not isinstance(model_cfg, dict) else model_cfg.get("torch_dtype", "auto")
    model, tokenizer, ref_dtype = load_causal_lm_and_tokenizer(
        base_model_path,
        device=tc.device(device),
        trust_remote_code=bool(metadata.get("trust_remote_code", True)),
        torch_dtype=torch_dtype,
        use_cache=True,
    )
    if "dtype" not in patch_cfg:
        patch_cfg["dtype"] = ref_dtype
    model = patch_qwen_ffn_layers(model=model, layer_idxs=layer_idxs, motn_cfg=patch_cfg, device=tc.device(device), dtype=ref_dtype)
    state_dict = normalize_legacy_gate_state_dict_for_model(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer, metadata
