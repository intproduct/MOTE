from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch as tc
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def normalize_hf_config(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def resolve_torch_dtype(dtype_name: str | None, device: tc.device) -> tc.dtype:
    if dtype_name is None:
        dtype_name = "auto"
    name = str(dtype_name).strip().lower()
    if name in {"", "auto"}:
        return tc.float16 if device.type == "cuda" else tc.float32
    mapping = {
        "float16": tc.float16,
        "fp16": tc.float16,
        "half": tc.float16,
        "bfloat16": tc.bfloat16,
        "bf16": tc.bfloat16,
        "float32": tc.float32,
        "fp32": tc.float32,
        "float": tc.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype={dtype_name}")
    resolved = mapping[name]
    return resolved


def dtype_to_name(dtype: tc.dtype) -> str:
    return str(dtype).replace("torch.", "")


def align_mixt_generation_config(model, tokenizer) -> None:
    """Keep existing stop IDs and include the native chat terminator in memory."""
    cfg = model.config
    base = getattr(cfg, "base_model_config", {}) or {}
    if not (getattr(cfg, "mixt_compress_layers", None) or base.get("mixt_compress_layers")):
        return
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Native MixT tokenizer has no eos_token_id")
    for target in (cfg, getattr(model, "generation_config", None)):
        if target is None:
            continue
        old = getattr(target, "eos_token_id", None)
        ids = list(old) if isinstance(old, (list, tuple)) else ([] if old is None else [old])
        target.eos_token_id = list(dict.fromkeys([int(v) for v in ids] + [int(eos)]))
        target.pad_token_id = tokenizer.pad_token_id
    if base:
        base["eos_token_id"] = cfg.eos_token_id
        base["pad_token_id"] = cfg.pad_token_id


def load_hf_tokenizer(
    model_path: str | Path,
    *,
    trust_remote_code: bool = True,
    padding_side: str | None = None,
):
    """Load only the tokenizer artifacts for immutable static evaluation."""
    resolved_path = str(Path(model_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        fix_mistral_regex=False,
    )
    if padding_side is not None:
        tokenizer.padding_side = str(padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm_and_tokenizer(
    model_path: str | Path,
    *,
    device: tc.device,
    trust_remote_code: bool = True,
    torch_dtype: str | None = "auto",
    use_cache: Optional[bool] = None,
    attn_implementation: str | None = None,
):
    model_path = str(Path(model_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        fix_mistral_regex=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_dtype = resolve_torch_dtype(torch_dtype, device)
    load_kwargs: Dict[str, Any] = {
        "device_map": None,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation is None:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if getattr(config, "mixt_compress_layers", None):
            # Match the validated partner import arithmetic across training and
            # raw checkpoint restoration. Faster kernels remain an explicit choice.
            attn_implementation = "eager"
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=resolved_dtype, **load_kwargs).to(device=device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=resolved_dtype, **load_kwargs).to(device=device)
    if use_cache is not None:
        model.config.use_cache = bool(use_cache)
    align_mixt_generation_config(model, tokenizer)
    return model, tokenizer, resolved_dtype
