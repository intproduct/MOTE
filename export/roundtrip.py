from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .validate import ExportValidationResult


TOKENIZER_MARKERS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "sentencepiece.bpe.model",
)


def _err(code: str, message: str, path: Path | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"code": code, "message": message}
    if path is not None:
        out["path"] = str(path)
    return out


def _resolve_torch_dtype(torch_dtype: str):
    if str(torch_dtype or "auto").lower() == "auto":
        return "auto"
    import torch

    name = str(torch_dtype).replace("torch.", "").strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")
    return mapping[name]


def _has_tokenizer_files(path: Path) -> bool:
    return any((path / marker).exists() for marker in TOKENIZER_MARKERS)


def _has_patched_module(model: Any) -> bool:
    try:
        from fitmotn.patching import PATCHED_FFN_TYPES

        return any(isinstance(module, PATCHED_FFN_TYPES) for module in model.modules())
    except Exception:
        names = {"MOTNFFNLayer", "SparseMiXTFFNLayer", "ADTNBaselineFFNLayer"}
        return any(type(module).__name__ in names for module in model.modules())


def validate_hf_roundtrip(export_dir: str | Path, device: str = "cpu", torch_dtype: str = "auto", smoke_prompt: str | None = None) -> ExportValidationResult:
    target = Path(export_dir).expanduser()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    config_path = target / "config.json"
    if not config_path.exists():
        errors.append(_err("missing_file", "config.json is required", config_path))
        return ExportValidationResult(ok=False, errors=errors, warnings=warnings)
    for filename in ("configuration_fitmotn.py", "modeling_fitmotn.py"):
        if not (target / filename).exists():
            errors.append(_err("missing_file", f"{filename} is required", target / filename))
    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(_err("invalid_json", f"config.json is not valid JSON: {exc}", config_path))
        return ExportValidationResult(ok=False, errors=errors, warnings=warnings)
    auto_map = config_data.get("auto_map") or {}
    if auto_map.get("AutoConfig") != "configuration_fitmotn.FitMoTNConfig":
        errors.append(_err("invalid_auto_map", "AutoConfig auto_map entry is missing or invalid", config_path))
    if auto_map.get("AutoModel") != "modeling_fitmotn.FitMoTNModel":
        errors.append(_err("invalid_auto_map", "AutoModel auto_map entry is missing or invalid", config_path))
    if auto_map.get("AutoModelForCausalLM") != "modeling_fitmotn.FitMoTNForCausalLM":
        errors.append(_err("invalid_auto_map", "AutoModelForCausalLM auto_map entry is missing or invalid", config_path))
    if errors:
        return ExportValidationResult(ok=False, errors=errors, warnings=warnings)

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as exc:
        errors.append(_err("transformers_unavailable", f"transformers is required for HF roundtrip validation: {exc}"))
        return ExportValidationResult(ok=False, errors=errors, warnings=warnings)

    try:
        hf_config = AutoConfig.from_pretrained(target, trust_remote_code=True)
    except Exception as exc:
        errors.append(_err("autoconfig_failed", f"AutoConfig roundtrip failed: {exc}", target))
        return ExportValidationResult(ok=False, errors=errors, warnings=warnings)
    if getattr(hf_config, "model_type", None) != "fitmotn":
        errors.append(_err("invalid_config_type", "AutoConfig did not load model_type='fitmotn'", config_path))

    try:
        dtype = _resolve_torch_dtype(torch_dtype)
        load_kwargs = {"trust_remote_code": True}
        if dtype != "auto":
            load_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(target, **load_kwargs)
        if device:
            model.to(device)
        model.eval()
    except Exception as exc:
        errors.append(_err("automodel_failed", f"AutoModelForCausalLM roundtrip failed: {exc}", target))
        return ExportValidationResult(ok=False, errors=errors, warnings=warnings)

    if not hasattr(model, "base_model"):
        errors.append(_err("missing_base_model", "Loaded FitMoTN wrapper does not expose base_model"))
    layers_to_patch = getattr(getattr(model, "config", None), "layers_to_patch", []) or []
    if layers_to_patch and not _has_patched_module(model):
        errors.append(_err("missing_patched_modules", "layers_to_patch is non-empty but no patched MOTE/ADTN module was found"))

    tokenizer = None
    if _has_tokenizer_files(target):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                target,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
        except Exception as exc:
            errors.append(_err("tokenizer_failed", f"Tokenizer files are present but AutoTokenizer failed: {exc}", target))

    if smoke_prompt and tokenizer is not None:
        try:
            import torch

            inputs = tokenizer(smoke_prompt, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                errors.append(_err("smoke_no_logits", "Smoke forward did not return logits"))
            elif torch.isnan(logits).any().item():
                errors.append(_err("smoke_nan_logits", "Smoke forward produced NaN logits"))
        except Exception as exc:
            errors.append(_err("smoke_failed", f"Smoke forward failed: {exc}", target))

    return ExportValidationResult(ok=not errors, errors=errors, warnings=warnings)
