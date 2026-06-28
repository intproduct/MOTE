from __future__ import annotations

import importlib.resources as resources
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .format import FITMOTN_AUTO_MAP
from .manifest import build_hf_roundtrip_payloads, json_safe, write_json


REMOTE_CODE_FILES = ("configuration_fitmotn.py", "modeling_fitmotn.py")


@dataclass
class HFExportResult:
    output_dir: Path
    manifest: dict[str, Any]
    export_config: dict[str, Any]
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _require_safetensors_if_needed(safe_serialization: bool) -> None:
    if not safe_serialization:
        return
    try:
        import safetensors  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "--safe_serialization true requires safetensors to be installed; "
            "install safetensors or pass --safe_serialization false."
        ) from exc


def _copy_remote_code(output_dir: Path) -> None:
    package = "fitmotn.export.remote_code"
    for filename in REMOTE_CODE_FILES:
        source = resources.files(package).joinpath(filename)
        (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _base_config_dict(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        raise ValueError("restored model does not expose a serializable Hugging Face config")
    data = json_safe(config.to_dict())
    if not isinstance(data, dict):
        raise ValueError("restored model config did not serialize to a JSON object")
    data["use_cache"] = True
    if "torch_dtype" in data:
        data["torch_dtype"] = json_safe(data["torch_dtype"])
    return data


def _save_generation_config(model: Any, output_dir: Path) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None or not hasattr(generation_config, "save_pretrained"):
        return
    if hasattr(generation_config, "use_cache"):
        generation_config.use_cache = True
    generation_config.save_pretrained(output_dir)


def _save_tokenizer(tokenizer: Any, output_dir: Path) -> None:
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# FitMoTN HF roundtrip export

This directory is a Stage 4B FitMoTN export. It is loadable through Hugging Face custom code:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True)
```

The wrapped patched model is exposed through `base_model`. vLLM support is intentionally disabled in this stage; Stage 4C will add vLLM offline runner support.

Do not commit private exported weights, tokenizer files, checkpoints, or manifests containing private local paths.
""",
        encoding="utf-8",
    )


def _assert_saved_config(output_dir: Path) -> None:
    path = output_dir / "config.json"
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "fitmotn":
        raise ValueError("saved config.json does not contain model_type='fitmotn'")
    auto_map = config.get("auto_map") or {}
    for key, expected in FITMOTN_AUTO_MAP.items():
        if auto_map.get(key) != expected:
            raise ValueError(f"saved config.json missing auto_map[{key!r}]={expected!r}")


def _load_state_into_wrapper(wrapper: Any, restored_model: Any) -> tuple[list[str], list[str]]:
    result = wrapper.base_model.load_state_dict(restored_model.state_dict(), strict=False)
    if hasattr(result, "missing_keys"):
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
    elif isinstance(result, tuple):
        missing = list(result[0])
        unexpected = list(result[1] if len(result) > 1 else [])
    else:
        missing = []
        unexpected = []
    if missing or unexpected:
        raise ValueError(f"wrapper.base_model state load was not exact; missing={missing}, unexpected={unexpected}")
    return missing, unexpected


def export_fitmotn_hf_roundtrip(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
    tokenizer_source: str | None = None,
    copy_tokenizer: bool = True,
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
    torch_dtype: str = "auto",
    roundtrip_device: str = "cpu",
    base_trust_remote_code: bool = False,
) -> HFExportResult:
    _require_safetensors_if_needed(bool(safe_serialization))

    from ..eval.restore import restore_fitmotn_model
    from .format import EXPORT_CONFIG_FILENAME, EXPORT_FORMAT_VERSION, EXPORT_MANIFEST_FILENAME
    from .remote_code.configuration_fitmotn import FitMoTNConfig
    from .remote_code.modeling_fitmotn import FitMoTNForCausalLM

    checkpoint_path = Path(checkpoint_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    restored_model, tokenizer, metadata = restore_fitmotn_model(
        checkpoint_path,
        device=roundtrip_device,
        trust_remote_code=base_trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=True,
    )
    base_model_name_or_path = str(base_model or metadata.get("base_model_path") or "")
    if not base_model_name_or_path:
        raise ValueError("base_model_name_or_path could not be inferred; pass --base_model")
    resolved_tokenizer = str(tokenizer_source or metadata.get("tokenizer_path") or base_model_name_or_path)
    base_cfg = _base_config_dict(restored_model)
    patch_cfg = dict(metadata.get("patch_cfg") or metadata.get("motn_cfg") or {})
    if not patch_cfg and metadata.get("layers_to_patch"):
        raise ValueError("raw checkpoint metadata does not contain full patch_cfg/motn_cfg needed for HF roundtrip export")
    patch_backend = str(metadata.get("patch_backend", patch_cfg.get("patch_backend", "motn")) or "motn").lower()
    patch_cfg["patch_backend"] = patch_backend
    layers_to_patch = [int(i) for i in (metadata.get("layers_to_patch") or [])]

    config = FitMoTNConfig(
        base_model_name_or_path=base_model_name_or_path,
        base_model_config=base_cfg,
        base_model_architectures=list(base_cfg.get("architectures") or []),
        tokenizer_source=resolved_tokenizer,
        fitmotn_patch_config=json_safe(patch_cfg),
        layers_to_patch=layers_to_patch,
        patch_backend=patch_backend,
        fitmotn_export_version=EXPORT_FORMAT_VERSION,
        format_version=EXPORT_FORMAT_VERSION,
        original_model_type=base_cfg.get("model_type"),
        exported_code_ready=True,
        auto_map_ready=True,
        architectures=["FitMoTNForCausalLM"],
        use_cache=True,
    )
    wrapper = FitMoTNForCausalLM(config)
    missing, unexpected = _load_state_into_wrapper(wrapper, restored_model)
    wrapper.eval()

    output_path.mkdir(parents=True, exist_ok=True)
    wrapper.save_pretrained(output_path, safe_serialization=bool(safe_serialization), max_shard_size=str(max_shard_size))
    _assert_saved_config(output_path)
    _copy_remote_code(output_path)
    if copy_tokenizer:
        _save_tokenizer(tokenizer, output_path)
    _save_generation_config(restored_model, output_path)
    manifest, export_config = build_hf_roundtrip_payloads(
        checkpoint_path,
        base_model=base_model_name_or_path,
        tokenizer=resolved_tokenizer,
        metadata=metadata,
        base_model_config=base_cfg,
        missing_keys=missing,
        unexpected_keys=unexpected,
        safe_serialization=bool(safe_serialization),
        max_shard_size=str(max_shard_size),
    )
    write_json(output_path / EXPORT_MANIFEST_FILENAME, manifest)
    write_json(output_path / EXPORT_CONFIG_FILENAME, export_config)
    _write_readme(output_path)
    return HFExportResult(
        output_dir=output_path,
        manifest=manifest,
        export_config=export_config,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )
