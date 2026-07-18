from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from ..utils.paths import resolve_path
from .defaults import make_default_config
from .schema import FitMoTNConfig
from ..rl.device_topology import (
    normalize_rollout_actor_configs,
    normalize_visible_devices,
    rollout_actor_specs,
    validate_actor_topology,
)


VALID_EVAL_BACKENDS = {"lm_eval", "evalscope", "both"}
SINGLE_BACKENDS = {"lm_eval", "evalscope"}
VALID_LR_SCHEDULER_TYPES = {"linear", "constant", "constant_with_warmup", "cosine"}
VALID_RESUME_STAGES = {"auto", "stage_b", "full"}
VALID_TEMPERATURE_SCHEDULE_TYPES = {"cosine", "constant"}
VALID_RL_MODES = {"gsm8k_grpo"}
VALID_RL_ROLLOUT_BACKENDS = {"hf", "vllm"}
VALID_VLLM_SYNC_STRATEGIES = {
    "export_reload",
    "weight_transfer_dryrun_static",
    "weight_transfer_dryrun_runtime",
    "weight_transfer_nccl",
    "weight_transfer_ipc",
}
VALID_VLLM_NATIVE_TRANSFER_LEVELS = {"none", "update_only", "four_phase"}
VALID_VLLM_WEIGHT_TRANSFER_BACKENDS = {"nccl", "ipc"}
VALID_VLLM_WEIGHT_TRANSFER_SCOPES = {"full_policy", "trainable_patch"}
VALID_VLLM_EXECUTION_MODES = {"in_process", "subprocess"}
VALID_RL_TRAINABLE_MODES = {"all", "patch_only", "motn_only", "gate_only", "router_only", "global_only"}
VALID_FORMAT_MODES = {"raw", "chat"}
VALID_ZERO_ADVANTAGE_RETRY_ACTIONS = {"warn_continue", "raise"}
VALID_BLOCK_INIT_MODES = {"gamma_normal", "base_stats_normal", "base_stats_trunc_normal"}
VALID_EXTRA_DATASET_FORMATS = {"text", "chat_messages", "prompt_response", "reasoning_qa"}
VALID_EXTRA_DATASET_SOURCES = {"hf", "local_jsonl", "jsonl", "jsonl_gz", "load_from_disk", "auto"}

MODEL_ALIASES = {
    "qwen3_8b": "${MODEL_ROOT}/Qwen3-8B",
    "qwen3_0_6b": "${MODEL_ROOT}/Qwen3-0.6B",
}

DATASET_ALIASES = {
    "gsm8k": {
        "gsm8k_cache_path": "${CACHE_ROOT}/gsm8k_main",
        "datas_dir": "${DATA_ROOT}",
        "use_wiki_local": False,
        "use_fineweb": False,
        "use_code": False,
        "use_gsm8k_train": True,
        "use_gsm8k_socratic_train": False,
        "use_svamp_train": False,
        "use_metamath_train": False,
        "use_math_train": False,
        "use_mmlu_train": False,
        "use_openr1_math": False,
        "use_numinamath_cot": False,
        "use_openthoughts_math": False,
        "use_bespoke_stratos": False,
    },
}

DATA_PATH_FIELDS = [
    "tok_shard_dir",
    "datas_dir",
    "fineweb_cache_path",
    "code_cache_path",
    "gsm8k_cache_path",
    "gsm8k_socratic_cache_path",
    "svamp_cache_path",
    "metamath_cache_path",
    "mmlu_cache_path",
    "math_cache_root",
    "openr1_math_cache_path",
    "numinamath_cot_cache_path",
    "openthoughts_math_cache_path",
    "bespoke_stratos_cache_path",
    "custom_reasoning_jsonl_path",
]
PATH_FIELDS = {
    "model": {"model_path"},
    "data": set(DATA_PATH_FIELDS),
    "output": {"root_dir"},
    "train": {"resume_fitmotn_from", "resume_weights_from", "resume_checkpoint_from"},
    "rl": {"resume_from", "resume_weights_from", "resume_checkpoint_from", "train_json", "vllm_export_root"},
    "diagnostics": {"prompts_file"},
}

ACTIVE_DATA_PATHS = {
    "tok_shard_dir": "use_wiki_local",
    "fineweb_cache_path": "use_fineweb",
    "code_cache_path": "use_code",
    "gsm8k_cache_path": "use_gsm8k_train",
    "gsm8k_socratic_cache_path": "use_gsm8k_socratic_train",
    "svamp_cache_path": "use_svamp_train",
    "metamath_cache_path": "use_metamath_train",
    "mmlu_cache_path": "use_mmlu_train",
    "math_cache_root": "use_math_train",
    "openr1_math_cache_path": "use_openr1_math",
    "numinamath_cot_cache_path": "use_numinamath_cot",
    "openthoughts_math_cache_path": "use_openthoughts_math",
    "bespoke_stratos_cache_path": "use_bespoke_stratos",
    "custom_reasoning_jsonl_path": "use_custom_reasoning_jsonl",
}


def _ensure_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object/dict, got {type(value).__name__}")
    return value


def _apply_section_values(section_name: str, section_obj: Any, values: Mapping[str, Any]) -> None:
    if not is_dataclass(section_obj):
        raise TypeError(f"config section {section_name} is not a dataclass")
    valid_fields = {field.name: field for field in fields(section_obj)}
    unknown = sorted(set(values.keys()) - set(valid_fields.keys()))
    if unknown:
        raise ValueError(f"Unknown keys in config section '{section_name}': {', '.join(unknown)}")
    for key, value in values.items():
        current_value = getattr(section_obj, key)
        if is_dataclass(current_value):
            nested_values = _ensure_mapping(f"{section_name}.{key}", value)
            _apply_section_values(f"{section_name}.{key}", current_value, nested_values)
        else:
            setattr(section_obj, key, value)


def _normalize_alias(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    alias = str(value).strip().lower()
    if alias == "":
        return None
    return alias


def _path_sources(cfg: FitMoTNConfig) -> dict[str, str]:
    sources = getattr(cfg, "_path_sources", None)
    if sources is None:
        sources = {}
        setattr(cfg, "_path_sources", sources)
    return sources


def _mark_path_source(cfg: FitMoTNConfig, section_name: str, field_name: str, source: str) -> None:
    if field_name in PATH_FIELDS.get(section_name, set()):
        _path_sources(cfg)[f"{section_name}.{field_name}"] = source


def _mark_payload_path_sources(cfg: FitMoTNConfig, payload: Mapping[str, Any], source: str) -> None:
    for section_name, field_names in PATH_FIELDS.items():
        section_values = payload.get(section_name)
        if not isinstance(section_values, Mapping):
            continue
        for field_name in field_names:
            if field_name in section_values:
                _mark_path_source(cfg, section_name, field_name, source)


def _source_for(cfg: FitMoTNConfig, section_name: str, field_name: str) -> str:
    return _path_sources(cfg).get(f"{section_name}.{field_name}", "default")


def _apply_aliases(cfg: FitMoTNConfig) -> None:
    model_alias = _normalize_alias(getattr(cfg, "model_alias", None), "model_alias")
    if model_alias is not None:
        if model_alias not in MODEL_ALIASES:
            raise ValueError(f"model_alias must be one of {sorted(MODEL_ALIASES)}, got {model_alias!r}")
        if not getattr(cfg.model, "model_path", None) or _source_for(cfg, "model", "model_path") == "default":
            cfg.model.model_path = MODEL_ALIASES[model_alias]
            _mark_path_source(cfg, "model", "model_path", "alias")
        cfg.model_alias = model_alias

    dataset_alias = _normalize_alias(getattr(cfg, "dataset_alias", None), "dataset_alias")
    if dataset_alias is not None:
        if dataset_alias not in DATASET_ALIASES:
            raise ValueError(f"dataset_alias must be one of {sorted(DATASET_ALIASES)}, got {dataset_alias!r}")
        alias_values = DATASET_ALIASES[dataset_alias]
        for key, value in alias_values.items():
            if key.endswith("_path") or key.endswith("_dir") or key.endswith("_root"):
                if not getattr(cfg.data, key, None) or _source_for(cfg, "data", key) == "default":
                    setattr(cfg.data, key, value)
                    _mark_path_source(cfg, "data", key, "alias")
            else:
                setattr(cfg.data, key, value)
        cfg.dataset_alias = dataset_alias


def _normalize_optional_path_value(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _resolve_optional_path(section_name: str, obj: Any, field_name: str, cfg: FitMoTNConfig, *, allow_none: bool = True) -> None:
    value = _normalize_optional_path_value(getattr(obj, field_name, None))
    if value is None:
        if not allow_none:
            resolve_path(None, key=f"{section_name}.{field_name}", source=_source_for(cfg, section_name, field_name), cfg=cfg, allow_none=False)
        setattr(obj, field_name, None)
        return
    setattr(
        obj,
        field_name,
        resolve_path(
            value,
            key=f"{section_name}.{field_name}",
            source=_source_for(cfg, section_name, field_name),
            cfg=cfg,
            allow_none=allow_none,
        ),
    )


def _resolve_and_validate_paths(cfg: FitMoTNConfig) -> None:
    _apply_aliases(cfg)

    _resolve_optional_path("model", cfg.model, "model_path", cfg, allow_none=False)
    _resolve_optional_path("output", cfg.output, "root_dir", cfg, allow_none=False)
    _resolve_optional_path("train", cfg.train, "resume_fitmotn_from", cfg)
    _resolve_optional_path("train", cfg.train, "resume_weights_from", cfg)
    _resolve_optional_path("train", cfg.train, "resume_checkpoint_from", cfg)
    _resolve_optional_path("rl", cfg.rl, "resume_from", cfg)
    _resolve_optional_path("rl", cfg.rl, "resume_weights_from", cfg)
    _resolve_optional_path("rl", cfg.rl, "resume_checkpoint_from", cfg)
    _resolve_optional_path("rl", cfg.rl, "train_json", cfg)
    _resolve_optional_path("rl", cfg.rl, "vllm_export_root", cfg)
    _resolve_optional_path("diagnostics", cfg.diagnostics, "prompts_file", cfg)

    require_gsm8k = bool(getattr(cfg.rl, "enabled", False) and getattr(cfg.rl, "use_config_data", True))

    for field_name, flag_name in ACTIVE_DATA_PATHS.items():
        require_path = bool(getattr(cfg.data, flag_name, False)) or (field_name == "gsm8k_cache_path" and require_gsm8k)
        if require_path or _source_for(cfg, "data", field_name) != "default":
            _resolve_optional_path("data", cfg.data, field_name, cfg, allow_none=not require_path)
    _resolve_optional_path("data", cfg.data, "datas_dir", cfg)
    for field_name in DATA_PATH_FIELDS:
        if field_name == "datas_dir" or field_name in ACTIVE_DATA_PATHS:
            continue
        if _source_for(cfg, "data", field_name) != "default":
            _resolve_optional_path("data", cfg.data, field_name, cfg)


def _normalize_extra_datasets(cfg: FitMoTNConfig) -> None:
    value = getattr(cfg.data, "extra_datasets", [])
    if value is None:
        cfg.data.extra_datasets = []
        return
    if not isinstance(value, list):
        raise TypeError("data.extra_datasets must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    role_defaults = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "model": "assistant", "system": "system"}
    path_fields = {"path", "cache_path"}
    for idx, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"data.extra_datasets[{idx}] must be an object/dict")
        ds = dict(item)
        name = str(ds.get("name", "")).strip()
        if not name:
            raise ValueError(f"data.extra_datasets[{idx}].name is required")
        if name in seen:
            raise ValueError(f"data.extra_datasets contains duplicate name {name!r}")
        seen.add(name)
        dataset_format = str(ds.get("format", "")).strip().lower()
        if dataset_format not in VALID_EXTRA_DATASET_FORMATS:
            raise ValueError(
                f"data.extra_datasets[{idx}].format must be one of {sorted(VALID_EXTRA_DATASET_FORMATS)}, got {dataset_format!r}"
            )
        source = str(ds.get("source", "")).strip().lower()
        if source not in VALID_EXTRA_DATASET_SOURCES:
            raise ValueError(
                f"data.extra_datasets[{idx}].source must be one of {sorted(VALID_EXTRA_DATASET_SOURCES)}, got {source!r}"
            )
        ds["name"] = name
        ds["format"] = dataset_format
        ds["source"] = source
        ds["enabled"] = bool(ds.get("enabled", True))
        ds["split"] = str(ds.get("split", "train") or "train")
        weight = float(ds.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"data.extra_datasets[{idx}].weight must be a finite non-negative number, got {weight}")
        ds["weight"] = weight
        if ds.get("max_samples") is not None:
            max_samples = int(ds["max_samples"])
            if max_samples <= 0:
                raise ValueError(f"data.extra_datasets[{idx}].max_samples must be > 0 when set, got {max_samples}")
            ds["max_samples"] = max_samples
        default_group = "pretrain" if dataset_format == "text" else "task"
        ds["group"] = str(ds.get("group", default_group) or default_group)
        ds["bucket"] = str(ds.get("bucket", "pretrain_general" if ds["group"] == "pretrain" else "extra_task") or "extra_task")
        ds["source_family"] = str(
            ds.get("source_family", "text" if dataset_format == "text" else ("reasoning" if dataset_format == "reasoning_qa" else "chat"))
        )
        role_map = dict(role_defaults)
        role_map.update({str(k).strip().lower(): str(v).strip().lower() for k, v in dict(ds.get("role_map") or {}).items()})
        ds["role_map"] = role_map
        ds["role_key"] = str(ds.get("role_key", "role") or "role")
        ds["content_key"] = str(ds.get("content_key", "content") or "content")
        ds["skip_if_no_assistant"] = bool(ds.get("skip_if_no_assistant", True))
        ds["skip_empty"] = bool(ds.get("skip_empty", True))
        for field_name in path_fields:
            if ds.get(field_name) is not None:
                ds[field_name] = resolve_path(
                    str(ds[field_name]),
                    key=f"data.extra_datasets[{idx}].{field_name}",
                    source="config",
                    cfg=cfg,
                    allow_none=True,
                )
        if source == "hf" and not str(ds.get("hf_name", "")).strip():
            raise ValueError(f"data.extra_datasets[{idx}].hf_name is required when source='hf'")
        if source in {"local_jsonl", "jsonl", "jsonl_gz", "load_from_disk"} and not str(ds.get("path", "")).strip():
            raise ValueError(f"data.extra_datasets[{idx}].path is required when source={source!r}")
        if source == "auto" and not (str(ds.get("path", "")).strip() or str(ds.get("cache_path", "")).strip() or str(ds.get("hf_name", "")).strip()):
            raise ValueError(f"data.extra_datasets[{idx}] requires path, cache_path, or hf_name when source='auto'")
        normalized.append(ds)
    cfg.data.extra_datasets = normalized


def _setdefault_task_value(mapping: dict[str, int], task_name: str, value: Any) -> None:
    if task_name not in mapping:
        mapping[task_name] = int(value)


def _ensure_backend_extra(eval_cfg) -> None:
    backend_extra = dict(getattr(eval_cfg, "backend_extra", {}) or {})
    backend_extra.setdefault("lm_eval", {})
    backend_extra.setdefault("evalscope", {})
    eval_cfg.backend_extra = backend_extra


def _finalize_eval_config(cfg: FitMoTNConfig) -> None:
    eval_cfg = cfg.eval
    eval_backend = str(getattr(eval_cfg, "eval_backend", "lm_eval")).strip().lower()
    if eval_backend not in VALID_EVAL_BACKENDS:
        raise ValueError(f"eval.eval_backend must be one of {sorted(VALID_EVAL_BACKENDS)}, got {eval_backend!r}")
    eval_cfg.eval_backend = eval_backend

    primary_backend = str(getattr(eval_cfg, "primary_eval_backend", "lm_eval")).strip().lower()
    if primary_backend not in SINGLE_BACKENDS:
        raise ValueError(f"eval.primary_eval_backend must be one of {sorted(SINGLE_BACKENDS)}, got {primary_backend!r}")
    if eval_backend != "both" and primary_backend != eval_backend:
        primary_backend = eval_backend
    eval_cfg.primary_eval_backend = primary_backend

    eval_cfg.backend_defaults.lm_eval.device = str(getattr(eval_cfg, "lm_eval_device", eval_cfg.backend_defaults.lm_eval.device))
    eval_cfg.backend_defaults.lm_eval.batch_size = int(getattr(eval_cfg, "lm_eval_batch_size", eval_cfg.backend_defaults.lm_eval.batch_size))
    if eval_cfg.backend_defaults.evalscope.device == "cuda:0":
        eval_cfg.backend_defaults.evalscope.device = str(eval_cfg.backend_defaults.lm_eval.device)
    if int(eval_cfg.backend_defaults.evalscope.batch_size) == 1:
        eval_cfg.backend_defaults.evalscope.batch_size = int(eval_cfg.backend_defaults.lm_eval.batch_size)

    _ensure_backend_extra(eval_cfg)

    fewshot_cfg = eval_cfg.fewshot
    _setdefault_task_value(fewshot_cfg.task_overrides, "gsm8k", getattr(eval_cfg, "lm_eval_num_fewshot_gsm8k", 8))
    _setdefault_task_value(fewshot_cfg.task_overrides, "mmlu", getattr(eval_cfg, "lm_eval_num_fewshot_mmlu", 5))
    _setdefault_task_value(fewshot_cfg.task_overrides, "hendrycks_math", getattr(eval_cfg, "lm_eval_num_fewshot_math", 4))

    limits_cfg = eval_cfg.limits
    _setdefault_task_value(limits_cfg.baseline_small, "gsm8k", getattr(eval_cfg, "baseline_small_limit_gsm8k", 64))
    _setdefault_task_value(limits_cfg.baseline_small, "mmlu", getattr(eval_cfg, "baseline_small_limit_mmlu", 128))
    _setdefault_task_value(limits_cfg.mid, "gsm8k", getattr(eval_cfg, "early_limit_gsm8k", 32))
    _setdefault_task_value(limits_cfg.mid, "mmlu", getattr(eval_cfg, "early_limit_mmlu", 64))
    _setdefault_task_value(limits_cfg.final, "gsm8k", getattr(eval_cfg, "final_limit_gsm8k", 0))
    _setdefault_task_value(limits_cfg.final, "mmlu", getattr(eval_cfg, "final_limit_mmlu", 512))
    _setdefault_task_value(limits_cfg.final, "hendrycks_math", getattr(eval_cfg, "final_limit_math", 256))

    max_gen_toks_cfg = eval_cfg.max_gen_toks
    _setdefault_task_value(max_gen_toks_cfg.baseline_small, "gsm8k", getattr(eval_cfg, "early_max_gen_toks_gsm8k", 256))
    _setdefault_task_value(max_gen_toks_cfg.mid, "gsm8k", getattr(eval_cfg, "early_max_gen_toks_gsm8k", 256))
    _setdefault_task_value(max_gen_toks_cfg.final, "gsm8k", getattr(eval_cfg, "final_max_gen_toks_gsm8k", 256))
    _setdefault_task_value(max_gen_toks_cfg.final, "hendrycks_math", getattr(eval_cfg, "final_max_gen_toks_math", 256))

    eval_cfg.protocols.default = str(getattr(eval_cfg.protocols, "default", "legacy") or "legacy")
    eval_cfg.runtime.chat_template_args = dict(getattr(eval_cfg.runtime, "chat_template_args", {}) or {})


def _finalize_train_config(cfg: FitMoTNConfig, explicit_train_keys: set[str], *, validate_paths: bool = False) -> FitMoTNConfig:
    train_cfg = cfg.train
    data_cfg = cfg.data
    model_cfg = cfg.model
    default_log_every = int(getattr(train_cfg, "log_every", 1))

    block_init_mode = str(getattr(model_cfg, "block_init_mode", "gamma_normal") or "gamma_normal").strip().lower()
    if block_init_mode not in VALID_BLOCK_INIT_MODES:
        raise ValueError(f"model.block_init_mode must be one of {sorted(VALID_BLOCK_INIT_MODES)}, got {block_init_mode!r}")
    model_cfg.block_init_mode = block_init_mode
    model_cfg.block_init_std_scale = float(getattr(model_cfg, "block_init_std_scale", 1.0))
    if (not math.isfinite(float(model_cfg.block_init_std_scale))) or float(model_cfg.block_init_std_scale) <= 0.0:
        raise ValueError(f"model.block_init_std_scale must be > 0, got {model_cfg.block_init_std_scale}")
    model_cfg.block_init_trunc_std = float(getattr(model_cfg, "block_init_trunc_std", 2.0))
    if (not math.isfinite(float(model_cfg.block_init_trunc_std))) or float(model_cfg.block_init_trunc_std) <= 0.0:
        raise ValueError(f"model.block_init_trunc_std must be > 0, got {model_cfg.block_init_trunc_std}")

    reasoning_format = str(getattr(data_cfg, "reasoning_format", "raw") or "raw").strip().lower()
    if reasoning_format not in VALID_FORMAT_MODES:
        raise ValueError(f"data.reasoning_format must be one of {sorted(VALID_FORMAT_MODES)}, got {reasoning_format!r}")
    data_cfg.reasoning_format = reasoning_format
    data_cfg.reasoning_chat_enable_thinking = bool(getattr(data_cfg, "reasoning_chat_enable_thinking", False))
    system_prompt = getattr(data_cfg, "reasoning_chat_system_prompt", None)
    data_cfg.reasoning_chat_system_prompt = None if system_prompt is None or str(system_prompt).strip() == "" else str(system_prompt)
    data_cfg.reasoning_chat_use_generation_prompt_for_labels = bool(
        getattr(data_cfg, "reasoning_chat_use_generation_prompt_for_labels", True)
    )
    data_cfg.use_custom_reasoning_jsonl = bool(getattr(data_cfg, "use_custom_reasoning_jsonl", False))
    data_cfg.custom_reasoning_jsonl_path = _normalize_optional_path_value(getattr(data_cfg, "custom_reasoning_jsonl_path", None))
    data_cfg.wt_custom_reasoning = float(getattr(data_cfg, "wt_custom_reasoning", 1.0))
    data_cfg.custom_reasoning_dataset_name = str(
        getattr(data_cfg, "custom_reasoning_dataset_name", "custom_verified_math") or "custom_verified_math"
    )
    data_cfg.custom_reasoning_bucket = str(getattr(data_cfg, "custom_reasoning_bucket", "gsm8k_core") or "gsm8k_core")
    if data_cfg.use_custom_reasoning_jsonl and not data_cfg.custom_reasoning_jsonl_path:
        raise ValueError("data.custom_reasoning_jsonl_path is required when data.use_custom_reasoning_jsonl=true")
    _normalize_extra_datasets(cfg)

    legacy_resume = _normalize_optional_path_value(getattr(train_cfg, "resume_fitmotn_from", None))
    weights_resume = _normalize_optional_path_value(getattr(train_cfg, "resume_weights_from", None))
    exact_resume = _normalize_optional_path_value(getattr(train_cfg, "resume_checkpoint_from", None))
    if legacy_resume and weights_resume and legacy_resume != weights_resume:
        raise ValueError("train.resume_fitmotn_from and train.resume_weights_from point to different checkpoints")
    weights_resume = weights_resume or legacy_resume
    if weights_resume and exact_resume:
        raise ValueError("train.resume_weights_from and train.resume_checkpoint_from are mutually exclusive")
    train_cfg.resume_fitmotn_from = weights_resume
    train_cfg.resume_weights_from = weights_resume
    train_cfg.resume_checkpoint_from = exact_resume

    resume_stage = str(getattr(train_cfg, "resume_stage", "auto") or "auto").strip().lower()
    if resume_stage not in VALID_RESUME_STAGES:
        raise ValueError(f"train.resume_stage must be one of {sorted(VALID_RESUME_STAGES)}, got {resume_stage!r}")
    if getattr(train_cfg, "resume_fitmotn_from", None) and resume_stage == "full":
        raise ValueError("train.resume_stage='full' is not implemented for resume_fitmotn_from; use 'auto' or 'stage_b'")
    train_cfg.resume_stage = resume_stage

    extra_updates = getattr(train_cfg, "extra_updates", None)
    if extra_updates is None:
        train_cfg.extra_updates = None
    else:
        train_cfg.extra_updates = int(extra_updates)
        if int(train_cfg.extra_updates) <= 0:
            raise ValueError(f"train.extra_updates must be > 0 when set, got {train_cfg.extra_updates}")

    if "block_lr" not in explicit_train_keys or getattr(train_cfg, "block_lr", None) is None:
        train_cfg.block_lr = float(getattr(train_cfg, "lr"))
    else:
        train_cfg.block_lr = float(getattr(train_cfg, "block_lr"))
    if "router_lr" not in explicit_train_keys or getattr(train_cfg, "router_lr", None) is None:
        train_cfg.router_lr = float(getattr(train_cfg, "block_lr"))
    else:
        train_cfg.router_lr = float(getattr(train_cfg, "router_lr"))

    stage_lr_fields = [
        "stage_a_block_lr",
        "stage_a_router_lr",
        "stage_b_block_lr",
        "stage_b_router_lr",
    ]
    stage_specific_lr_enabled = False
    for field_name in stage_lr_fields:
        value = getattr(train_cfg, field_name, None)
        if value is None:
            continue
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"train.{field_name} must be > 0 when set, got {value}")
        setattr(train_cfg, field_name, value)
        stage_specific_lr_enabled = True

    if getattr(train_cfg, "usage_report_every", None) is None:
        train_cfg.usage_report_every = int(getattr(train_cfg, "usage_dump_every", 0))
    if getattr(train_cfg, "usage_light_every", None) is None:
        train_cfg.usage_light_every = default_log_every
    if getattr(train_cfg, "usage_light_jsonl_every", None) is None:
        train_cfg.usage_light_jsonl_every = default_log_every
    if getattr(train_cfg, "train_jsonl_every", None) is None:
        train_cfg.train_jsonl_every = default_log_every

    if bool(getattr(train_cfg, "benchmark_train_only", False)):
        benchmark_defaults = {
            "enable_usage_runtime_tracking": False,
            "enable_usage_report": False,
            "enable_heavy_runtime_stats": False,
            "enable_grad_param_norm": False,
            "enable_cuda_snapshot": False,
            "usage_light_every": 0,
            "usage_light_jsonl_every": 0,
            "usage_report_every": 0,
        }
        for key, value in benchmark_defaults.items():
            if key not in explicit_train_keys:
                setattr(train_cfg, key, value)

    if getattr(train_cfg, "enable_usage_runtime_tracking", None) is None:
        train_cfg.enable_usage_runtime_tracking = True
    if getattr(train_cfg, "enable_usage_report", None) is None:
        train_cfg.enable_usage_report = True
    if getattr(train_cfg, "enable_heavy_runtime_stats", None) is None:
        train_cfg.enable_heavy_runtime_stats = True
    if getattr(train_cfg, "enable_grad_param_norm", None) is None:
        train_cfg.enable_grad_param_norm = False
    if getattr(train_cfg, "enable_cuda_snapshot", None) is None:
        train_cfg.enable_cuda_snapshot = False

    scheduler_type = str(getattr(train_cfg, "lr_scheduler_type", "linear") or "linear").strip().lower()
    if scheduler_type not in VALID_LR_SCHEDULER_TYPES:
        raise ValueError(
            f"train.lr_scheduler_type must be one of {sorted(VALID_LR_SCHEDULER_TYPES)}, got {scheduler_type!r}"
        )
    train_cfg.lr_scheduler_type = scheduler_type
    if stage_specific_lr_enabled and scheduler_type != "constant":
        raise ValueError("stage-specific learning rates are currently supported only with lr_scheduler_type='constant'.")

    lr_decay_steps = getattr(train_cfg, "lr_decay_steps", None)
    if lr_decay_steps is None:
        train_cfg.lr_decay_steps = None
    else:
        train_cfg.lr_decay_steps = int(lr_decay_steps)
        if int(train_cfg.lr_decay_steps) <= 0:
            raise ValueError(f"train.lr_decay_steps must be > 0 when set, got {train_cfg.lr_decay_steps}")

    temperature_schedule_type = str(getattr(train_cfg, "temperature_schedule_type", "cosine") or "cosine").strip().lower()
    if temperature_schedule_type not in VALID_TEMPERATURE_SCHEDULE_TYPES:
        raise ValueError(
            f"train.temperature_schedule_type must be one of {sorted(VALID_TEMPERATURE_SCHEDULE_TYPES)}, got {temperature_schedule_type!r}"
        )
    train_cfg.temperature_schedule_type = temperature_schedule_type
    for field_name in ["begin_t", "end_t", "stage_a_begin_t", "stage_a_end_t", "stage_b_begin_t", "stage_b_end_t"]:
        value = getattr(train_cfg, field_name, None)
        if value is not None:
            setattr(train_cfg, field_name, float(value))

    train_cfg.final_answer_weight = float(getattr(train_cfg, "final_answer_weight", 1.0))
    if float(train_cfg.final_answer_weight) <= 0.0:
        raise ValueError(f"train.final_answer_weight must be > 0, got {train_cfg.final_answer_weight}")
    train_cfg.final_answer_marker = str(getattr(train_cfg, "final_answer_marker", "####"))
    train_cfg.final_answer_weight_enabled = bool(getattr(train_cfg, "final_answer_weight_enabled", False))
    train_cfg.save_every_updates = int(getattr(train_cfg, "save_every_updates", 1000))
    train_cfg.checkpoint_keep_last_n = int(getattr(train_cfg, "checkpoint_keep_last_n", 3))
    train_cfg.checkpoint_keep_every_n = int(getattr(train_cfg, "checkpoint_keep_every_n", 0))
    train_cfg.save_on_stage_transition = bool(getattr(train_cfg, "save_on_stage_transition", True))
    train_cfg.checkpoint_fail_on_save_error = bool(getattr(train_cfg, "checkpoint_fail_on_save_error", True))
    train_cfg.checkpoint_temp_max_age_sec = float(getattr(train_cfg, "checkpoint_temp_max_age_sec", 3600.0))
    if train_cfg.save_every_updates < 1:
        raise ValueError("train.save_every_updates must be >= 1")
    if train_cfg.checkpoint_keep_last_n < 0 or train_cfg.checkpoint_keep_every_n < 0:
        raise ValueError("train checkpoint retention values must be >= 0")
    if train_cfg.checkpoint_temp_max_age_sec < 0.0:
        raise ValueError("train.checkpoint_temp_max_age_sec must be >= 0")

    train_cfg.usage_dump_every = int(getattr(train_cfg, "usage_report_every", 0))
    train_cfg.usage_light_every = int(getattr(train_cfg, "usage_light_every", 0))
    train_cfg.usage_light_jsonl_every = int(getattr(train_cfg, "usage_light_jsonl_every", 0))
    train_cfg.usage_report_every = int(getattr(train_cfg, "usage_report_every", 0))
    train_cfg.train_jsonl_every = int(getattr(train_cfg, "train_jsonl_every", default_log_every))
    train_cfg.heavy_log_every = int(getattr(train_cfg, "heavy_log_every", 500))
    _validate_bucket_config(cfg)
    _finalize_eval_config(cfg)
    _finalize_rl_config(cfg)
    if validate_paths:
        _resolve_and_validate_paths(cfg)
    return cfg


def _normalize_optional_path(value: Any) -> str | None:
    return _normalize_optional_path_value(value)


def _finalize_rl_config(cfg: FitMoTNConfig) -> None:
    rl_cfg = cfg.rl
    rl_cfg.enabled = bool(getattr(rl_cfg, "enabled", False))
    rl_cfg.run_after_sft = bool(getattr(rl_cfg, "run_after_sft", False))
    if rl_cfg.run_after_sft and not rl_cfg.enabled:
        raise ValueError("rl.run_after_sft=true requires rl.enabled=true")

    mode = str(getattr(rl_cfg, "mode", "gsm8k_grpo") or "gsm8k_grpo").strip().lower()
    if mode not in VALID_RL_MODES:
        raise ValueError(f"rl.mode must be one of {sorted(VALID_RL_MODES)}, got {mode!r}")
    rl_cfg.mode = mode

    trainable_mode = str(getattr(rl_cfg, "trainable_mode", "patch_only") or "patch_only").strip().lower()
    if trainable_mode not in VALID_RL_TRAINABLE_MODES:
        raise ValueError(f"rl.trainable_mode must be one of {sorted(VALID_RL_TRAINABLE_MODES)}, got {trainable_mode!r}")
    rl_cfg.trainable_mode = trainable_mode

    legacy_rl_resume = _normalize_optional_path(getattr(rl_cfg, "resume_from", None))
    rl_weights_resume = _normalize_optional_path(getattr(rl_cfg, "resume_weights_from", None))
    rl_exact_resume = _normalize_optional_path(getattr(rl_cfg, "resume_checkpoint_from", None))
    if legacy_rl_resume and rl_weights_resume and legacy_rl_resume != rl_weights_resume:
        raise ValueError("rl.resume_from and rl.resume_weights_from point to different checkpoints")
    rl_weights_resume = rl_weights_resume or legacy_rl_resume
    if rl_weights_resume and rl_exact_resume:
        raise ValueError("rl.resume_weights_from and rl.resume_checkpoint_from are mutually exclusive")
    rl_cfg.resume_from = rl_weights_resume
    rl_cfg.resume_weights_from = rl_weights_resume
    rl_cfg.resume_checkpoint_from = rl_exact_resume
    if rl_cfg.run_after_sft and (rl_weights_resume or rl_exact_resume):
        raise ValueError("rl.run_after_sft=true cannot be combined with an RL resume checkpoint")
    rl_cfg.train_json = _normalize_optional_path(getattr(rl_cfg, "train_json", None))
    rl_cfg.vllm_export_root = _normalize_optional_path(getattr(rl_cfg, "vllm_export_root", None))
    rl_cfg.train_source = str(getattr(rl_cfg, "train_source", "gsm8k_train") or "gsm8k_train")
    rl_cfg.output_subdir = str(getattr(rl_cfg, "output_subdir", "rl_grpo") or "rl_grpo").strip() or "rl_grpo"
    rl_cfg.prompt_template = str(getattr(rl_cfg, "prompt_template", "") or "")
    prompt_format = str(getattr(rl_cfg, "prompt_format", "raw") or "raw").strip().lower()
    if prompt_format not in VALID_FORMAT_MODES:
        raise ValueError(f"rl.prompt_format must be one of {sorted(VALID_FORMAT_MODES)}, got {prompt_format!r}")
    rl_cfg.prompt_format = prompt_format
    rl_cfg.chat_enable_thinking = bool(getattr(rl_cfg, "chat_enable_thinking", False))
    chat_system_prompt = getattr(rl_cfg, "chat_system_prompt", None)
    rl_cfg.chat_system_prompt = None if chat_system_prompt is None or str(chat_system_prompt).strip() == "" else str(chat_system_prompt)
    rl_cfg.use_config_data = bool(getattr(rl_cfg, "use_config_data", True))
    rl_cfg.no_ref_model = bool(getattr(rl_cfg, "no_ref_model", True))
    rl_cfg.enable_usage_tracking = bool(getattr(rl_cfg, "enable_usage_tracking", False))
    rl_cfg.gradient_checkpointing = bool(getattr(rl_cfg, "gradient_checkpointing", False))
    rl_cfg.rollout_use_cache = bool(getattr(rl_cfg, "rollout_use_cache", True))
    rl_cfg.checkpoint_fail_on_save_error = bool(getattr(rl_cfg, "checkpoint_fail_on_save_error", True))
    rollout_backend = str(getattr(rl_cfg, "rollout_backend", "hf") or "hf").strip().lower()
    if rollout_backend not in VALID_RL_ROLLOUT_BACKENDS:
        raise ValueError(f"rl.rollout_backend must be one of {sorted(VALID_RL_ROLLOUT_BACKENDS)}, got {rollout_backend!r}")
    rl_cfg.rollout_backend = rollout_backend
    vllm_sync_strategy = str(getattr(rl_cfg, "vllm_sync_strategy", "export_reload") or "export_reload").strip().lower()
    if vllm_sync_strategy not in VALID_VLLM_SYNC_STRATEGIES:
        raise ValueError(
            f"rl.vllm_sync_strategy must be one of {sorted(VALID_VLLM_SYNC_STRATEGIES)}, got {vllm_sync_strategy!r}"
        )
    rl_cfg.vllm_sync_strategy = vllm_sync_strategy
    rl_cfg.vllm_model_impl = str(getattr(rl_cfg, "vllm_model_impl", "transformers") or "transformers")
    rl_cfg.vllm_dtype = str(getattr(rl_cfg, "vllm_dtype", "auto") or "auto")
    weight_transfer_backend = str(getattr(rl_cfg, "vllm_weight_transfer_backend", "nccl") or "nccl").strip().lower()
    if weight_transfer_backend not in VALID_VLLM_WEIGHT_TRANSFER_BACKENDS:
        raise ValueError(
            "rl.vllm_weight_transfer_backend must be one of "
            f"{sorted(VALID_VLLM_WEIGHT_TRANSFER_BACKENDS)}, got {weight_transfer_backend!r}"
        )
    rl_cfg.vllm_weight_transfer_backend = weight_transfer_backend
    transfer_scope = str(
        getattr(rl_cfg, "vllm_weight_transfer_scope", "full_policy") or "full_policy"
    ).strip().lower()
    if transfer_scope not in VALID_VLLM_WEIGHT_TRANSFER_SCOPES:
        raise ValueError(
            "rl.vllm_weight_transfer_scope must be one of "
            f"{sorted(VALID_VLLM_WEIGHT_TRANSFER_SCOPES)}, got {transfer_scope!r}"
        )
    rl_cfg.vllm_weight_transfer_scope = transfer_scope
    required_level = str(getattr(rl_cfg, "vllm_native_transfer_required_level", "four_phase") or "four_phase").strip().lower()
    if required_level not in VALID_VLLM_NATIVE_TRANSFER_LEVELS:
        raise ValueError(
            "rl.vllm_native_transfer_required_level must be one of "
            f"{sorted(VALID_VLLM_NATIVE_TRANSFER_LEVELS)}, got {required_level!r}"
        )
    rl_cfg.vllm_native_transfer_required_level = required_level
    if vllm_sync_strategy == "weight_transfer_nccl" and required_level == "none":
        raise ValueError(
            "rl.vllm_native_transfer_required_level='none' cannot authorize real NCCL transfer; "
            "use 'update_only' for explicit experimental support or 'four_phase' for the strict gate"
        )
    dryrun_mode = str(getattr(rl_cfg, "vllm_weight_transfer_dryrun_mode", "static") or "static").strip().lower()
    if dryrun_mode not in {"static", "runtime"}:
        raise ValueError("rl.vllm_weight_transfer_dryrun_mode must be 'static' or 'runtime'")
    rl_cfg.vllm_weight_transfer_dryrun_mode = dryrun_mode
    rl_cfg.vllm_weight_transfer_master_addr = str(
        getattr(rl_cfg, "vllm_weight_transfer_master_addr", "127.0.0.1") or "127.0.0.1"
    )
    vllm_device = getattr(rl_cfg, "vllm_device", None)
    rl_cfg.vllm_device = None if vllm_device is None or str(vllm_device).strip() == "" else str(vllm_device).strip()
    rl_cfg.vllm_enforce_eager = bool(getattr(rl_cfg, "vllm_enforce_eager", True))
    rl_cfg.vllm_disable_log_stats = bool(getattr(rl_cfg, "vllm_disable_log_stats", True))
    rl_cfg.vllm_export_validate_roundtrip = bool(getattr(rl_cfg, "vllm_export_validate_roundtrip", True))
    rl_cfg.vllm_weight_transfer_packed = bool(getattr(rl_cfg, "vllm_weight_transfer_packed", True))
    rl_cfg.vllm_weight_transfer_validate_coverage = bool(getattr(rl_cfg, "vllm_weight_transfer_validate_coverage", True))
    rl_cfg.vllm_weight_transfer_fail_on_partial = bool(getattr(rl_cfg, "vllm_weight_transfer_fail_on_partial", True))
    rl_cfg.vllm_weight_transfer_fallback_to_export_reload = bool(
        getattr(rl_cfg, "vllm_weight_transfer_fallback_to_export_reload", False)
    )
    rl_cfg.vllm_weight_transfer_validate_after_sync = bool(getattr(rl_cfg, "vllm_weight_transfer_validate_after_sync", True))
    rl_cfg.vllm_weight_transfer_require_runtime_checksums = bool(
        getattr(rl_cfg, "vllm_weight_transfer_require_runtime_checksums", False)
    )
    rl_cfg.vllm_enable_sleep_mode = bool(getattr(rl_cfg, "vllm_enable_sleep_mode", False))
    rl_cfg.vllm_wake_weights_before_update = bool(getattr(rl_cfg, "vllm_wake_weights_before_update", True))
    rl_cfg.vllm_wake_kv_cache_after_update = bool(getattr(rl_cfg, "vllm_wake_kv_cache_after_update", True))
    rl_cfg.vllm_fallback_to_hf = bool(getattr(rl_cfg, "vllm_fallback_to_hf", False))
    rl_cfg.allow_stale_vllm_policy = bool(getattr(rl_cfg, "allow_stale_vllm_policy", False))
    rl_cfg.vllm_allow_text_prompt_fallback = bool(getattr(rl_cfg, "vllm_allow_text_prompt_fallback", False))
    rl_cfg.vllm_fail_on_cuda_oom = bool(getattr(rl_cfg, "vllm_fail_on_cuda_oom", True))
    rl_cfg.vllm_empty_cache_before_engine_init = bool(getattr(rl_cfg, "vllm_empty_cache_before_engine_init", False))
    rl_cfg.vllm_verify_engine_policy = bool(getattr(rl_cfg, "vllm_verify_engine_policy", True))
    if transfer_scope == "trainable_patch":
        if rollout_backend != "vllm":
            raise ValueError(
                "rl.vllm_weight_transfer_scope='trainable_patch' requires rl.rollout_backend='vllm'"
            )
        if vllm_sync_strategy != "weight_transfer_nccl":
            raise ValueError(
                "rl.vllm_weight_transfer_scope='trainable_patch' requires "
                "rl.vllm_sync_strategy='weight_transfer_nccl'"
            )
        if str(getattr(rl_cfg, "trainable_mode", "patch_only") or "patch_only").strip().lower() != "patch_only":
            raise ValueError(
                "rl.vllm_weight_transfer_scope='trainable_patch' requires rl.trainable_mode='patch_only'"
            )
        if not bool(rl_cfg.vllm_weight_transfer_validate_coverage) or not bool(rl_cfg.vllm_weight_transfer_fail_on_partial):
            raise ValueError(
                "trainable_patch native sync requires vllm_weight_transfer_validate_coverage=true "
                "and vllm_weight_transfer_fail_on_partial=true"
            )
        if bool(rl_cfg.vllm_weight_transfer_fallback_to_export_reload) or bool(rl_cfg.vllm_fallback_to_hf):
            raise ValueError(
                "trainable_patch acceptance forbids export_reload and HF fallbacks"
            )
        if bool(rl_cfg.allow_stale_vllm_policy):
            raise ValueError("trainable_patch acceptance requires allow_stale_vllm_policy=false")
    execution_mode = str(getattr(rl_cfg, "vllm_execution_mode", "in_process") or "in_process").strip().lower()
    if execution_mode not in VALID_VLLM_EXECUTION_MODES:
        raise ValueError(
            f"rl.vllm_execution_mode must be one of {sorted(VALID_VLLM_EXECUTION_MODES)}, got {execution_mode!r}"
        )
    rl_cfg.vllm_execution_mode = execution_mode
    rl_cfg.vllm_actor_cuda_visible_devices = normalize_visible_devices(
        getattr(rl_cfg, "vllm_actor_cuda_visible_devices", None)
    )
    rl_cfg.vllm_rollout_actors = normalize_rollout_actor_configs(
        getattr(rl_cfg, "vllm_rollout_actors", None)
    )
    actor_start_method = str(getattr(rl_cfg, "vllm_actor_start_method", "spawn") or "spawn").strip().lower()
    if actor_start_method not in {"spawn", "forkserver"}:
        raise ValueError("rl.vllm_actor_start_method must be 'spawn' or 'forkserver'")
    rl_cfg.vllm_actor_start_method = actor_start_method
    if rl_cfg.vllm_actor_cuda_visible_devices and rl_cfg.vllm_device is None:
        rl_cfg.vllm_device = "cuda:0"
    validate_actor_topology(rl_cfg)
    if execution_mode == "subprocess" and vllm_sync_strategy in {"weight_transfer_ipc", "weight_transfer_dryrun_runtime"}:
        raise ValueError(
            "rl.vllm_execution_mode='subprocess' does not support IPC transfer or runtime dry-run inspection"
        )
    if execution_mode == "subprocess" and vllm_sync_strategy == "weight_transfer_nccl":
        if required_level != "update_only":
            raise ValueError(
                "subprocess NCCL sync currently targets the verified vLLM update-only API; set "
                "rl.vllm_native_transfer_required_level='update_only'"
            )
        actor_specs = rollout_actor_specs(rl_cfg)
        non_tp1 = [spec["name"] for spec in actor_specs if int(spec["tensor_parallel_size"]) != 1]
        if non_tp1:
            raise ValueError(
                "subprocess NCCL sync currently requires TP1 rollout actors; "
                f"non-TP1 actors={non_tp1}"
            )
    if execution_mode == "subprocess" and bool(rl_cfg.vllm_allow_text_prompt_fallback):
        raise ValueError(
            "rl.vllm_allow_text_prompt_fallback is not supported by the subprocess actor; "
            "token-id prompts are required to preserve tokenizer parity"
        )
    rl_cfg.rollout_inference_mode = bool(getattr(rl_cfg, "rollout_inference_mode", True))
    rl_cfg.rollout_log_timing = bool(getattr(rl_cfg, "rollout_log_timing", True))
    rl_cfg.skip_zero_advantage_updates = bool(getattr(rl_cfg, "skip_zero_advantage_updates", True))
    rl_cfg.shuffle_train_data = bool(getattr(rl_cfg, "shuffle_train_data", True))
    rl_cfg.mgpo_enabled = bool(getattr(rl_cfg, "mgpo_enabled", False))
    rl_cfg.long2short_enabled = bool(getattr(rl_cfg, "long2short_enabled", False))
    zero_advantage_retry_action = str(getattr(rl_cfg, "zero_advantage_retry_action", "warn_continue") or "warn_continue").strip().lower()
    if zero_advantage_retry_action not in VALID_ZERO_ADVANTAGE_RETRY_ACTIONS:
        raise ValueError(
            "rl.zero_advantage_retry_action must be one of "
            f"{sorted(VALID_ZERO_ADVANTAGE_RETRY_ACTIONS)}, got {zero_advantage_retry_action!r}"
        )
    rl_cfg.zero_advantage_retry_action = zero_advantage_retry_action

    int_fields = [
        "max_steps",
        "batch_size",
        "group_size",
        "grad_accum",
        "max_new_tokens",
        "log_every",
        "log_memory_every",
        "logprob_micro_batch_size",
        "rollout_micro_batch_size",
        "rollout_max_prompt_tokens",
        "vllm_tensor_parallel_size",
        "vllm_max_model_len",
        "vllm_max_num_seqs",
        "vllm_sync_every_updates",
        "vllm_keep_sync_exports",
        "vllm_export_roundtrip_validation_every",
        "vllm_policy_fingerprint_samples_per_tensor",
        "vllm_weight_transfer_master_port",
        "vllm_sync_validation_every",
        "vllm_sleep_level_before_sync",
        "vllm_actor_resource_log_every",
        "empty_cache_every",
        "max_zero_advantage_rollout_retries",
        "save_every_updates",
        "checkpoint_keep_last_n",
        "checkpoint_keep_every_n",
        "eval_every_updates",
        "sample_log_count",
        "eval_limit_gsm8k",
        "eval_max_gen_toks_gsm8k",
    ]
    for field_name in int_fields:
        setattr(rl_cfg, field_name, int(getattr(rl_cfg, field_name)))
    for field_name in [
        "lr",
        "eps_clip",
        "beta",
        "temperature",
        "top_p",
        "max_grad_norm",
        "vllm_gpu_memory_utilization",
        "vllm_weight_transfer_timeout_sec",
        "vllm_actor_request_timeout_sec",
        "vllm_actor_shutdown_timeout_sec",
        "vllm_export_temp_max_age_sec",
        "checkpoint_temp_max_age_sec",
    ]:
        setattr(rl_cfg, field_name, float(getattr(rl_cfg, field_name)))
    for field_name in ["mgpo_p0", "mgpo_gamma", "mgpo_weight_min", "mgpo_weight_max", "mgpo_eps", "long2short_lambda", "long2short_eps"]:
        setattr(rl_cfg, field_name, float(getattr(rl_cfg, field_name)))
    rl_cfg.long2short_min_correct = int(getattr(rl_cfg, "long2short_min_correct"))
    if not (0.0 < float(rl_cfg.mgpo_p0) < 1.0):
        raise ValueError(f"rl.mgpo_p0 must satisfy 0 < p0 < 1, got {rl_cfg.mgpo_p0}")
    if float(rl_cfg.mgpo_gamma) < 0.0:
        raise ValueError(f"rl.mgpo_gamma must be >= 0, got {rl_cfg.mgpo_gamma}")
    if float(rl_cfg.mgpo_eps) <= 0.0:
        raise ValueError(f"rl.mgpo_eps must be > 0, got {rl_cfg.mgpo_eps}")
    if float(rl_cfg.mgpo_weight_min) < 0.0 or float(rl_cfg.mgpo_weight_min) > float(rl_cfg.mgpo_weight_max):
        raise ValueError(
            "rl.mgpo_weight_min/max must satisfy 0 <= mgpo_weight_min <= mgpo_weight_max, "
            f"got {rl_cfg.mgpo_weight_min} > {rl_cfg.mgpo_weight_max}"
        )
    if float(rl_cfg.long2short_lambda) < 0.0:
        raise ValueError(f"rl.long2short_lambda must be >= 0, got {rl_cfg.long2short_lambda}")
    if int(rl_cfg.long2short_min_correct) < 1:
        raise ValueError(f"rl.long2short_min_correct must be >= 1, got {rl_cfg.long2short_min_correct}")
    if float(rl_cfg.long2short_eps) <= 0.0:
        raise ValueError(f"rl.long2short_eps must be > 0, got {rl_cfg.long2short_eps}")
    if getattr(rl_cfg, "debug_num_prompts", None) is not None:
        rl_cfg.debug_num_prompts = int(rl_cfg.debug_num_prompts)
        if rl_cfg.debug_num_prompts <= 0:
            raise ValueError(f"rl.debug_num_prompts must be > 0 when set, got {rl_cfg.debug_num_prompts}")
    if getattr(rl_cfg, "seed", None) is not None:
        rl_cfg.seed = int(rl_cfg.seed)
    if getattr(rl_cfg, "sampler_seed", None) is not None:
        rl_cfg.sampler_seed = int(rl_cfg.sampler_seed)

    if rl_cfg.enabled:
        positive_fields = ["max_steps", "batch_size", "group_size", "grad_accum", "max_new_tokens", "log_every", "sample_log_count"]
        for field_name in positive_fields:
            value = int(getattr(rl_cfg, field_name))
            if value <= 0:
                raise ValueError(f"rl.{field_name} must be > 0 when rl.enabled=true, got {value}")
        if float(rl_cfg.lr) <= 0.0:
            raise ValueError(f"rl.lr must be > 0 when rl.enabled=true, got {rl_cfg.lr}")
        if float(rl_cfg.eps_clip) <= 0.0:
            raise ValueError(f"rl.eps_clip must be > 0 when rl.enabled=true, got {rl_cfg.eps_clip}")
        if float(rl_cfg.max_grad_norm) <= 0.0:
            raise ValueError(f"rl.max_grad_norm must be > 0 when rl.enabled=true, got {rl_cfg.max_grad_norm}")
        if int(rl_cfg.save_every_updates) < 0:
            raise ValueError(f"rl.save_every_updates must be >= 0, got {rl_cfg.save_every_updates}")
        if int(rl_cfg.checkpoint_keep_last_n) < 0 or int(rl_cfg.checkpoint_keep_every_n) < 0:
            raise ValueError("rl checkpoint retention values must be >= 0")
        if float(rl_cfg.checkpoint_temp_max_age_sec) < 0.0:
            raise ValueError("rl.checkpoint_temp_max_age_sec must be >= 0")
        if int(rl_cfg.eval_every_updates) < 0:
            raise ValueError(f"rl.eval_every_updates must be >= 0, got {rl_cfg.eval_every_updates}")
        for field_name in ["log_memory_every", "logprob_micro_batch_size", "rollout_micro_batch_size", "rollout_max_prompt_tokens", "empty_cache_every"]:
            value = int(getattr(rl_cfg, field_name))
            if value < 0:
                raise ValueError(f"rl.{field_name} must be >= 0, got {value}")
        if int(rl_cfg.vllm_actor_resource_log_every) < 0:
            raise ValueError("rl.vllm_actor_resource_log_every must be >= 0")
        if int(rl_cfg.vllm_tensor_parallel_size) < 1:
            raise ValueError(f"rl.vllm_tensor_parallel_size must be >= 1, got {rl_cfg.vllm_tensor_parallel_size}")
        for field_name in ["vllm_max_model_len", "vllm_max_num_seqs"]:
            if int(getattr(rl_cfg, field_name)) < 0:
                raise ValueError(f"rl.{field_name} must be >= 0, got {getattr(rl_cfg, field_name)}")
        if int(rl_cfg.vllm_sync_every_updates) < 1:
            raise ValueError(f"rl.vllm_sync_every_updates must be >= 1, got {rl_cfg.vllm_sync_every_updates}")
        if int(rl_cfg.vllm_keep_sync_exports) < 0:
            raise ValueError(f"rl.vllm_keep_sync_exports must be >= 0, got {rl_cfg.vllm_keep_sync_exports}")
        if int(rl_cfg.vllm_export_roundtrip_validation_every) < 1:
            raise ValueError(
                "rl.vllm_export_roundtrip_validation_every must be >= 1, "
                f"got {rl_cfg.vllm_export_roundtrip_validation_every}"
            )
        if int(rl_cfg.vllm_policy_fingerprint_samples_per_tensor) < 1:
            raise ValueError(
                "rl.vllm_policy_fingerprint_samples_per_tensor must be >= 1, "
                f"got {rl_cfg.vllm_policy_fingerprint_samples_per_tensor}"
            )
        if int(rl_cfg.vllm_weight_transfer_master_port) < 0:
            raise ValueError(
                "rl.vllm_weight_transfer_master_port must be >= 0, "
                f"got {rl_cfg.vllm_weight_transfer_master_port}"
            )
        if int(rl_cfg.vllm_sync_validation_every) < 1:
            raise ValueError(f"rl.vllm_sync_validation_every must be >= 1, got {rl_cfg.vllm_sync_validation_every}")
        if int(rl_cfg.vllm_sleep_level_before_sync) < 0:
            raise ValueError(
                f"rl.vllm_sleep_level_before_sync must be >= 0, got {rl_cfg.vllm_sleep_level_before_sync}"
            )
        if float(rl_cfg.vllm_weight_transfer_timeout_sec) <= 0.0:
            raise ValueError(
                "rl.vllm_weight_transfer_timeout_sec must be > 0, "
                f"got {rl_cfg.vllm_weight_transfer_timeout_sec}"
            )
        if float(rl_cfg.vllm_actor_request_timeout_sec) <= 0.0:
            raise ValueError(
                f"rl.vllm_actor_request_timeout_sec must be > 0, got {rl_cfg.vllm_actor_request_timeout_sec}"
            )
        if float(rl_cfg.vllm_actor_shutdown_timeout_sec) <= 0.0:
            raise ValueError(
                f"rl.vllm_actor_shutdown_timeout_sec must be > 0, got {rl_cfg.vllm_actor_shutdown_timeout_sec}"
            )
        if float(rl_cfg.vllm_export_temp_max_age_sec) < 0.0:
            raise ValueError(
                "rl.vllm_export_temp_max_age_sec must be >= 0, "
                f"got {rl_cfg.vllm_export_temp_max_age_sec}"
            )
        if not (0.0 < float(rl_cfg.vllm_gpu_memory_utilization) <= 1.0):
            raise ValueError(
                "rl.vllm_gpu_memory_utilization must satisfy 0 < value <= 1, "
                f"got {rl_cfg.vllm_gpu_memory_utilization}"
            )
        if int(rl_cfg.max_zero_advantage_rollout_retries) < 1:
            raise ValueError(
                "rl.max_zero_advantage_rollout_retries must be >= 1, "
                f"got {rl_cfg.max_zero_advantage_rollout_retries}"
            )
        if float(rl_cfg.beta) < 0.0:
            raise ValueError(f"rl.beta must be >= 0, got {rl_cfg.beta}")
        if not rl_cfg.train_json and not bool(rl_cfg.use_config_data):
            raise ValueError("rl.train_json is required when rl.use_config_data=false")


def _validate_bucket_config(cfg: FitMoTNConfig) -> None:
    train_cfg = cfg.train
    mode = str(getattr(train_cfg, "task_bucket_mode", "flat")).strip().lower()
    if mode not in {"flat", "bucketed"}:
        raise ValueError(f"train.task_bucket_mode must be 'flat' or 'bucketed', got {mode!r}")
    train_cfg.task_bucket_mode = mode

    ratio_fields = [
        "stage_a_pretrain_ratio",
        "stage_a_task_ratio",
        "stage_a_core_task_ratio",
        "stage_a_aux_task_ratio",
        "stage_b_pretrain_ratio",
        "stage_b_task_ratio",
        "stage_b_core_task_ratio",
        "stage_b_aux_task_ratio",
    ]
    for field_name in ratio_fields:
        value = float(getattr(train_cfg, field_name, 0.0))
        if value < 0.0:
            raise ValueError(f"train.{field_name} must be >= 0, got {value}")

    if mode != "bucketed":
        return

    def _check_stage(stage_name: str) -> None:
        task_ratio = float(getattr(train_cfg, f"{stage_name}_task_ratio"))
        core_ratio = float(getattr(train_cfg, f"{stage_name}_core_task_ratio"))
        aux_ratio = float(getattr(train_cfg, f"{stage_name}_aux_task_ratio"))
        total = core_ratio + aux_ratio
        if abs(total - task_ratio) > 1e-8:
            raise ValueError(
                f"train.{stage_name}_core_task_ratio + train.{stage_name}_aux_task_ratio must equal "
                f"train.{stage_name}_task_ratio; got {core_ratio} + {aux_ratio} != {task_ratio}"
            )
        pretrain_ratio = float(getattr(train_cfg, f"{stage_name}_pretrain_ratio"))
        if abs((pretrain_ratio + task_ratio) - 1.0) > 1e-8:
            raise ValueError(
                f"train.{stage_name}_pretrain_ratio + train.{stage_name}_task_ratio must equal 1.0 in bucketed mode; "
                f"got {pretrain_ratio} + {task_ratio} != 1.0"
            )

    _check_stage("stage_a")
    if not bool(getattr(train_cfg, "stage_b_disable_pretrain", False)):
        _check_stage("stage_b")
    else:
        stage_b_pretrain_ratio = float(getattr(train_cfg, "stage_b_pretrain_ratio"))
        stage_b_task_ratio = float(getattr(train_cfg, "stage_b_task_ratio"))
        stage_b_core_ratio = float(getattr(train_cfg, "stage_b_core_task_ratio"))
        stage_b_aux_ratio = float(getattr(train_cfg, "stage_b_aux_task_ratio"))
        if abs(stage_b_pretrain_ratio) > 1e-8:
            raise ValueError(
                "train.stage_b_pretrain_ratio must be 0.0 when train.stage_b_disable_pretrain=true in bucketed mode"
            )
        if abs(stage_b_task_ratio - 1.0) > 1e-8:
            raise ValueError(
                "train.stage_b_task_ratio must be 1.0 when train.stage_b_disable_pretrain=true in bucketed mode"
            )
        if abs((stage_b_core_ratio + stage_b_aux_ratio) - 1.0) > 1e-8:
            raise ValueError(
                "train.stage_b_core_task_ratio + train.stage_b_aux_task_ratio must equal 1.0 when "
                "train.stage_b_disable_pretrain=true in bucketed mode"
            )


def apply_config_payload(cfg: FitMoTNConfig, payload: Mapping[str, Any]) -> FitMoTNConfig:
    payload = _ensure_mapping("config payload", payload)
    valid_sections = {field.name for field in fields(cfg)}
    unknown_sections = sorted(set(payload.keys()) - valid_sections)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {', '.join(unknown_sections)}")
    _mark_payload_path_sources(cfg, payload, "config")
    for section_name, section_values in payload.items():
        section_obj = getattr(cfg, section_name)
        if is_dataclass(section_obj):
            _apply_section_values(section_name, section_obj, _ensure_mapping(section_name, section_values))
        else:
            setattr(cfg, section_name, section_values)
    explicit_train_keys = set(payload.get("train", {}).keys()) if isinstance(payload.get("train"), Mapping) else set()
    explicit_model_keys = set(payload.get("model", {}).keys()) if isinstance(payload.get("model"), Mapping) else set()
    cumulative_train_keys = set(getattr(cfg, "_explicit_train_keys", set())) | explicit_train_keys
    cumulative_model_keys = set(getattr(cfg, "_explicit_model_keys", set())) | explicit_model_keys
    cfg = _finalize_train_config(cfg, cumulative_train_keys, validate_paths=True)
    setattr(cfg, "_explicit_train_keys", cumulative_train_keys)
    setattr(cfg, "_explicit_model_keys", cumulative_model_keys)
    return cfg


def load_config_from_json(config_json: str | Path, *, finalize: bool = True) -> FitMoTNConfig:
    cfg = make_default_config()
    path = Path(config_json).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if finalize:
        return apply_config_payload(cfg, payload)
    payload = _ensure_mapping("config payload", payload)
    valid_sections = {field.name for field in fields(cfg)}
    unknown_sections = sorted(set(payload.keys()) - valid_sections)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {', '.join(unknown_sections)}")
    _mark_payload_path_sources(cfg, payload, "config")
    for section_name, section_values in payload.items():
        section_obj = getattr(cfg, section_name)
        if is_dataclass(section_obj):
            _apply_section_values(section_name, section_obj, _ensure_mapping(section_name, section_values))
        else:
            setattr(cfg, section_name, section_values)
    explicit_train_keys = set(payload.get("train", {}).keys()) if isinstance(payload.get("train"), Mapping) else set()
    explicit_model_keys = set(payload.get("model", {}).keys()) if isinstance(payload.get("model"), Mapping) else set()
    setattr(cfg, "_explicit_train_keys", explicit_train_keys)
    setattr(cfg, "_explicit_model_keys", explicit_model_keys)
    return cfg


def apply_config_overrides(cfg: FitMoTNConfig, overrides: Mapping[str, Mapping[str, Any]] | None) -> FitMoTNConfig:
    if not overrides:
        return _finalize_train_config(cfg, set(getattr(cfg, "_explicit_train_keys", set())), validate_paths=True)
    _mark_payload_path_sources(cfg, overrides, "override")
    explicit_train_keys: set[str] = set()
    explicit_model_keys: set[str] = set()
    for section_name, section_values in overrides.items():
        if not section_values:
            continue
        section_obj = getattr(cfg, section_name, None)
        if section_obj is None:
            raise ValueError(f"Unknown override section: {section_name}")
        if is_dataclass(section_obj):
            _apply_section_values(section_name, section_obj, _ensure_mapping(section_name, section_values))
        else:
            setattr(cfg, section_name, section_values)
        if section_name == "train":
            explicit_train_keys.update(section_values.keys())
        if section_name == "model":
            explicit_model_keys.update(section_values.keys())
    cumulative_train_keys = set(getattr(cfg, "_explicit_train_keys", set())) | explicit_train_keys
    cumulative_model_keys = set(getattr(cfg, "_explicit_model_keys", set())) | explicit_model_keys
    cfg = _finalize_train_config(cfg, cumulative_train_keys, validate_paths=True)
    setattr(cfg, "_explicit_train_keys", cumulative_train_keys)
    setattr(cfg, "_explicit_model_keys", cumulative_model_keys)
    return cfg


def load_config(config_json: str | Path | None = None, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> FitMoTNConfig:
    cfg = make_default_config() if config_json is None else load_config_from_json(config_json, finalize=False)
    return apply_config_overrides(cfg, overrides)
