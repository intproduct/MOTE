from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional


EVAL_MODES = {"baseline_small", "mid", "final"}
SUPPORTED_BACKENDS = {"lm_eval", "evalscope"}
SUPPORTED_GENERATION_KEYS = {"temperature", "top_p", "top_k", "do_sample", "max_gen_toks"}
SUPPORTED_RUNTIME_KEYS = {
    "apply_chat_template",
    "enable_thinking",
    "think_end_token",
    "fewshot_as_multiturn",
    "system_instruction",
    "chat_template_args",
}


def normalize_backend_name(name: str) -> str:
    backend = str(name or "lm_eval").strip().lower()
    if backend not in {"lm_eval", "evalscope", "both"}:
        raise ValueError(f"Unsupported eval backend: {name!r}")
    return backend


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping-like value, got {type(value).__name__}")


def _merge_dict(base: Mapping[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dict(dict(merged.get(key, {})), value)
        else:
            merged[key] = value
    return merged


def get_enabled_backends(eval_cfg) -> list[str]:
    backend = normalize_backend_name(getattr(eval_cfg, "eval_backend", "lm_eval"))
    if backend == "both":
        return ["lm_eval", "evalscope"]
    return [backend]


def get_primary_backend(eval_cfg) -> str:
    backend = normalize_backend_name(getattr(eval_cfg, "eval_backend", "lm_eval"))
    primary = str(getattr(eval_cfg, "primary_eval_backend", "lm_eval") or "lm_eval").strip().lower()
    if backend == "both":
        if primary not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported primary eval backend: {primary!r}")
        return primary
    return backend


def _resolve_protocol(eval_cfg, task_name: str, task_override: Mapping[str, Any], mode_override: Mapping[str, Any]) -> str:
    if "protocol" in mode_override:
        return str(mode_override["protocol"])
    if "protocol" in task_override:
        return str(task_override["protocol"])
    protocols = getattr(eval_cfg, "protocols")
    return str(getattr(protocols, "task_protocols", {}).get(task_name, getattr(protocols, "default", "legacy")))


def _resolve_fewshot(eval_cfg, task_name: str, eval_mode: str, task_override: Mapping[str, Any], mode_override: Mapping[str, Any]) -> int:
    if "fewshot" in mode_override:
        return int(mode_override["fewshot"])
    fewshot_by_mode = task_override.get("fewshot_by_mode", {}) or {}
    if eval_mode in fewshot_by_mode:
        return int(fewshot_by_mode[eval_mode])
    if "fewshot" in task_override:
        return int(task_override["fewshot"])
    fewshot_cfg = getattr(eval_cfg, "fewshot")
    mode_map = dict(getattr(fewshot_cfg, "mode_overrides", {}).get(eval_mode, {}) or {})
    if task_name in mode_map:
        return int(mode_map[task_name])
    task_map = dict(getattr(fewshot_cfg, "task_overrides", {}) or {})
    if task_name in task_map:
        return int(task_map[task_name])
    return int(getattr(fewshot_cfg, "default", 0))


def _resolve_limit(eval_cfg, task_name: str, eval_mode: str, task_override: Mapping[str, Any], mode_override: Mapping[str, Any]) -> Optional[int]:
    if "limit" in mode_override:
        value = int(mode_override["limit"])
        return None if value <= 0 else value
    if eval_mode in (task_override.get("limits") or {}):
        value = int(task_override["limits"][eval_mode])
        return None if value <= 0 else value
    if "limit" in task_override:
        value = int(task_override["limit"])
        return None if value <= 0 else value
    mapping = dict(getattr(getattr(eval_cfg, "limits"), eval_mode, {}) or {})
    value = int(mapping.get(task_name, 0))
    return None if value <= 0 else value


def _resolve_max_gen_toks(eval_cfg, task_name: str, eval_mode: str, task_override: Mapping[str, Any], mode_override: Mapping[str, Any]) -> Optional[int]:
    if "max_gen_toks" in mode_override:
        value = int(mode_override["max_gen_toks"])
        return None if value <= 0 else value
    if eval_mode in (task_override.get("max_gen_toks_by_mode") or {}):
        value = int(task_override["max_gen_toks_by_mode"][eval_mode])
        return None if value <= 0 else value
    if "max_gen_toks" in task_override:
        value = int(task_override["max_gen_toks"])
        return None if value <= 0 else value
    mapping = dict(getattr(getattr(eval_cfg, "max_gen_toks"), eval_mode, {}) or {})
    value = int(mapping.get(task_name, 0))
    return None if value <= 0 else value


def _resolve_runtime(eval_cfg, task_override: Mapping[str, Any], mode_override: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = _to_plain_dict(getattr(eval_cfg, "runtime"))
    runtime = _merge_dict(runtime, dict(task_override.get("runtime", {}) or {}))
    runtime = _merge_dict(runtime, dict(mode_override.get("runtime", {}) or {}))
    return runtime


def _resolve_generation(eval_cfg, task_name: str, eval_mode: str, task_override: Mapping[str, Any], mode_override: Mapping[str, Any]) -> Dict[str, Any]:
    generation = _to_plain_dict(getattr(eval_cfg, "generation"))
    generation = _merge_dict(generation, dict(task_override.get("generation", {}) or {}))
    generation = _merge_dict(generation, dict(mode_override.get("generation", {}) or {}))
    max_gen_toks = _resolve_max_gen_toks(eval_cfg, task_name, eval_mode, task_override, mode_override)
    if max_gen_toks is not None:
        generation["max_gen_toks"] = int(max_gen_toks)
    return generation


def _resolve_backend_config(eval_cfg, backend: str, task_override: Mapping[str, Any], mode_override: Mapping[str, Any]) -> Dict[str, Any]:
    backend_defaults = _to_plain_dict(getattr(getattr(eval_cfg, "backend_defaults"), backend))
    backend_extra = dict((getattr(eval_cfg, "backend_extra", {}) or {}).get(backend, {}) or {})
    task_backend = dict((task_override.get("backend") or {}).get(backend, {}) or {})
    mode_backend = dict((mode_override.get("backend") or {}).get(backend, {}) or {})
    merged = _merge_dict(backend_defaults, backend_extra)
    merged = _merge_dict(merged, task_backend)
    merged = _merge_dict(merged, mode_backend)
    return merged


def resolve_task_eval_settings(fit_cfg, task_name: str, eval_mode: str, backend: str) -> Dict[str, Any]:
    if eval_mode not in EVAL_MODES:
        raise ValueError(f"Unknown eval mode: {eval_mode!r}")
    backend = normalize_backend_name(backend)
    if backend == "both":
        raise ValueError("resolve_task_eval_settings expects a single backend")
    eval_cfg = fit_cfg.eval
    task_override = dict(getattr(eval_cfg, "task_overrides", {}).get(task_name, {}) or {})
    mode_override = dict((task_override.get("modes") or {}).get(eval_mode, {}) or {})

    runtime = _resolve_runtime(eval_cfg, task_override, mode_override)
    generation = _resolve_generation(eval_cfg, task_name, eval_mode, task_override, mode_override)
    gen_kwargs = {k: v for k, v in generation.items() if k in SUPPORTED_GENERATION_KEYS and v is not None}
    unsupported_generation_keys = sorted(k for k, v in generation.items() if k not in SUPPORTED_GENERATION_KEYS and v is not None)
    unsupported_runtime_keys = sorted(k for k, v in runtime.items() if k not in SUPPORTED_RUNTIME_KEYS and v not in (None, {}, []))

    return {
        "task_name": task_name,
        "eval_mode": eval_mode,
        "backend": backend,
        "protocol": _resolve_protocol(eval_cfg, task_name, task_override, mode_override),
        "fewshot": _resolve_fewshot(eval_cfg, task_name, eval_mode, task_override, mode_override),
        "limit": _resolve_limit(eval_cfg, task_name, eval_mode, task_override, mode_override),
        "runtime": runtime,
        "generation": generation,
        "gen_kwargs": gen_kwargs or None,
        "backend_config": _resolve_backend_config(eval_cfg, backend, task_override, mode_override),
        "unsupported_generation_keys": unsupported_generation_keys,
        "unsupported_runtime_keys": unsupported_runtime_keys,
    }
