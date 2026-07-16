from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def collect_safetensor_shapes(model_dir: str | Path) -> Dict[str, tuple[int, ...]]:
    root = Path(model_dir).expanduser().resolve()
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError("safetensors is required for vLLM export preflight") from exc

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = dict(_load_json(index_path).get("weight_map") or {})
        shard_names = sorted(set(str(name) for name in weight_map.values()))
    else:
        shard_names = sorted(path.name for path in root.glob("*.safetensors"))
    if not shard_names:
        raise FileNotFoundError(f"no safetensors weights found in {root}")

    shapes: Dict[str, tuple[int, ...]] = {}
    for shard_name in shard_names:
        shard = root / shard_name
        if not shard.exists():
            raise FileNotFoundError(f"safetensors index references missing shard: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in shapes:
                    raise ValueError(f"duplicate safetensors key across shards: {key}")
                shapes[str(key)] = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
    return shapes


def apply_vllm_transformers_weight_mapping(model: Any, name: str) -> str | None:
    """Mirror the vLLM 0.18/0.19 generic Transformers mapper for one key."""

    mapped = str(name)
    for source, target in dict(getattr(model, "_checkpoint_conversion_mapping", {}) or {}).items():
        mapped = re.sub(source, target, mapped)

    ignored = list(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])
    if any(re.search(pattern, mapped) for pattern in ignored):
        return None

    base_model_prefix = str(getattr(model, "base_model_prefix", "") or "")
    if base_model_prefix and base_model_prefix != "model":
        mapped = re.sub(rf"^{re.escape(base_model_prefix)}\.(.+)", r"model.\1", mapped)

    direct_names = [name for name, _ in model.named_children()]
    direct_names.extend(name for name, _ in model.named_parameters(recurse=False))
    direct_names.extend(name for name, _ in model.named_buffers(recurse=False))
    alternatives = "|".join(re.escape(item) for item in direct_names)
    if alternatives:
        mapped = re.sub(rf"^(?!model\.)(({alternatives}).*)", r"model.\1", mapped)
        mapped = re.sub(rf"^(model\.)((?!{alternatives}).+)", r"\2", mapped)
    mapped = re.sub(r"^model\.(.+\.)*(lm_head.+)", r"\2", mapped)
    return mapped


def _model_state_shapes(model: Any) -> Dict[str, tuple[int, ...]]:
    return {str(name): tuple(int(dim) for dim in tensor.shape) for name, tensor in model.state_dict().items()}


def validate_vllm_export_preflight(
    model_dir: str | Path,
    *,
    expected_remote_code: str | Path | None = None,
    require_vllm: bool = True,
) -> Dict[str, Any]:
    root = Path(model_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checks: Dict[str, Any] = {}

    required_files = [
        "config.json",
        "configuration_fitmotn.py",
        "modeling_fitmotn.py",
        "fitmotn_export_manifest.json",
        "fitmotn_export_config.json",
    ]
    missing_files = [name for name in required_files if not (root / name).exists()]
    checks["required_files"] = {"ok": not missing_files, "missing": missing_files}
    if missing_files:
        errors.append(f"missing required export files: {missing_files}")
        return {"ok": False, "model_dir": str(root), "checks": checks, "errors": errors, "warnings": warnings}

    manifest = _load_json(root / "fitmotn_export_manifest.json")
    config_json = _load_json(root / "config.json")
    checks["manifest"] = {
        "export_stage": manifest.get("export_stage"),
        "hf_roundtrip_ready": manifest.get("hf_roundtrip_ready"),
        "vllm_ready": manifest.get("vllm_ready"),
        "vllm_model_impl": manifest.get("vllm_model_impl"),
    }
    if manifest.get("export_stage") != "hf_roundtrip" or manifest.get("hf_roundtrip_ready") is not True:
        errors.append("export manifest is not HF roundtrip-ready")
    if manifest.get("vllm_model_impl") != "transformers":
        errors.append("export manifest must declare vllm_model_impl='transformers'")
    auto_map = dict(config_json.get("auto_map") or {})
    if auto_map.get("AutoModel") != "modeling_fitmotn.FitMoTNModel":
        errors.append("config auto_map.AutoModel is not FitMoTNModel")

    exported_remote_code = root / "modeling_fitmotn.py"
    remote_check: Dict[str, Any] = {"export_sha256": _sha256(exported_remote_code)}
    if expected_remote_code is not None:
        expected_path = Path(expected_remote_code).expanduser().resolve()
        remote_check["expected_path"] = str(expected_path)
        remote_check["expected_sha256"] = _sha256(expected_path) if expected_path.exists() else None
        remote_check["matches_repository"] = bool(
            expected_path.exists() and remote_check["export_sha256"] == remote_check["expected_sha256"]
        )
        if not remote_check["matches_repository"]:
            errors.append("exported modeling_fitmotn.py is stale relative to the repository remote code")
    checks["remote_code"] = remote_check

    try:
        weight_shapes = collect_safetensor_shapes(root)
    except Exception as exc:
        errors.append(f"could not inspect safetensors: {exc}")
        weight_shapes = {}
    weight_keys = set(weight_shapes)
    embed_keys = sorted(
        key for key in weight_keys if key.endswith(("embed_tokens.weight", "wte.weight", "word_embeddings.weight"))
    )
    checks["weights"] = {
        "tensor_count": len(weight_keys),
        "embedding_keys": embed_keys,
        "has_lm_head": "lm_head.weight" in weight_keys,
    }
    if not embed_keys:
        warnings.append("checkpoint embedding key was not recognized by the common-name heuristic")
    if "lm_head.weight" not in weight_keys:
        errors.append("checkpoint has no explicit lm_head.weight required by strict vLLM loading")

    versions: Dict[str, Any] = {}
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModel

        versions["torch"] = torch.__version__
        versions["transformers"] = transformers.__version__
        versions["cuda_available"] = bool(torch.cuda.is_available())
        hf_config = AutoConfig.from_pretrained(root, trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModel.from_config(hf_config, trust_remote_code=True)
        tp_plan = getattr(model, "tp_plan", None)
        pp_plan = getattr(model, "pp_plan", None)
        checks["parallel_plans"] = {
            "tp_plan_type": type(tp_plan).__name__,
            "tp_plan_size": len(tp_plan) if isinstance(tp_plan, dict) else None,
            "pp_plan_type": type(pp_plan).__name__,
            "pp_plan_size": len(pp_plan) if isinstance(pp_plan, dict) else None,
        }
        if not isinstance(tp_plan, dict):
            errors.append("AutoModel.tp_plan is not a dict")
        if not isinstance(pp_plan, dict):
            errors.append("AutoModel.pp_plan is not a dict")

        target_shapes = {f"model.{key}": shape for key, shape in _model_state_shapes(model).items()}
        input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        input_weight = getattr(input_embeddings, "weight", None)
        if input_weight is not None:
            target_shapes["lm_head.weight"] = tuple(int(dim) for dim in input_weight.shape)
        else:
            errors.append("AutoModel does not expose input embedding weights needed for vLLM lm_head")

        mapped_shapes: Dict[str, tuple[int, ...]] = {}
        collisions: Dict[str, list[str]] = {}
        for source_name, shape in weight_shapes.items():
            mapped_name = apply_vllm_transformers_weight_mapping(model, source_name)
            if mapped_name is None:
                continue
            if mapped_name in mapped_shapes:
                collisions.setdefault(mapped_name, []).append(source_name)
            mapped_shapes[mapped_name] = shape
        missing_targets = sorted(set(target_shapes) - set(mapped_shapes))
        unexpected_mapped = sorted(set(mapped_shapes) - set(target_shapes))
        shape_mismatches = {
            key: {"checkpoint": mapped_shapes[key], "target": target_shapes[key]}
            for key in sorted(set(mapped_shapes) & set(target_shapes))
            if tuple(mapped_shapes[key]) != tuple(target_shapes[key])
        }
        checks["vllm_weight_contract_tp1"] = {
            "target_count": len(target_shapes),
            "mapped_count": len(mapped_shapes),
            "missing_targets": missing_targets[:50],
            "missing_target_count": len(missing_targets),
            "unexpected_mapped": unexpected_mapped[:50],
            "unexpected_mapped_count": len(unexpected_mapped),
            "shape_mismatches": shape_mismatches,
            "collisions": collisions,
        }
        if missing_targets:
            errors.append(f"vLLM TP1 mapped checkpoint misses {len(missing_targets)} target weights")
        if unexpected_mapped:
            errors.append(f"vLLM TP1 mapping produces {len(unexpected_mapped)} unexpected weights")
        if shape_mismatches:
            errors.append(f"vLLM TP1 mapping has {len(shape_mismatches)} shape mismatches")
        if collisions:
            errors.append(f"vLLM TP1 mapping has {len(collisions)} key collisions")
    except Exception as exc:
        errors.append(f"could not construct/meta-validate AutoModel: {type(exc).__name__}: {exc}")

    if require_vllm:
        try:
            import vllm

            versions["vllm"] = getattr(vllm, "__version__", "unknown")
            if not str(versions["vllm"]).startswith("0.19."):
                warnings.append(
                    f"preflight contract is targeted at vLLM 0.19.x; installed={versions['vllm']} requires separate GPU acceptance"
                )
        except Exception as exc:
            errors.append(f"vLLM import failed: {exc}")
    checks["versions"] = versions

    warnings.append("TP2+ custom ADTN tensor sharding is not validated by this TP1 preflight")
    warnings.append("CUDA attention/KV-cache profiling and token generation still require the GPU smoke gate")
    return {
        "ok": not errors,
        "model_dir": str(root),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
