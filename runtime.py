from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch as tc
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    if resolved is tc.bfloat16 and device.type != "cuda":
        return tc.float32
    return resolved


def dtype_to_name(dtype: tc.dtype) -> str:
    return str(dtype).replace("torch.", "")


def load_causal_lm_and_tokenizer(
    model_path: str | Path,
    *,
    device: tc.device,
    trust_remote_code: bool = True,
    torch_dtype: str | None = "auto",
    use_cache: Optional[bool] = None,
):
    model_path = str(Path(model_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_dtype = resolve_torch_dtype(torch_dtype, device)
    load_kwargs: Dict[str, Any] = {
        "device_map": None,
        "trust_remote_code": trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=resolved_dtype, **load_kwargs).to(device=device, dtype=resolved_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=resolved_dtype, **load_kwargs).to(device=device, dtype=resolved_dtype)
    if use_cache is not None:
        model.config.use_cache = bool(use_cache)
    return model, tokenizer, resolved_dtype
