from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .defaults import make_default_config
from .schema import FitMoTNConfig


VALID_EVAL_BACKENDS = {"lm_eval", "evalscope", "both"}
SINGLE_BACKENDS = {"lm_eval", "evalscope"}


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


def _finalize_train_config(cfg: FitMoTNConfig, explicit_train_keys: set[str]) -> FitMoTNConfig:
    train_cfg = cfg.train
    default_log_every = int(getattr(train_cfg, "log_every", 1))

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

    train_cfg.usage_dump_every = int(getattr(train_cfg, "usage_report_every", 0))
    train_cfg.usage_light_every = int(getattr(train_cfg, "usage_light_every", 0))
    train_cfg.usage_light_jsonl_every = int(getattr(train_cfg, "usage_light_jsonl_every", 0))
    train_cfg.usage_report_every = int(getattr(train_cfg, "usage_report_every", 0))
    train_cfg.train_jsonl_every = int(getattr(train_cfg, "train_jsonl_every", default_log_every))
    train_cfg.heavy_log_every = int(getattr(train_cfg, "heavy_log_every", 500))
    _validate_bucket_config(cfg)
    _finalize_eval_config(cfg)
    return cfg


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
    for section_name, section_values in payload.items():
        section_obj = getattr(cfg, section_name)
        _apply_section_values(section_name, section_obj, _ensure_mapping(section_name, section_values))
    explicit_train_keys = set(payload.get("train", {}).keys()) if isinstance(payload.get("train"), Mapping) else set()
    return _finalize_train_config(cfg, explicit_train_keys)


def load_config_from_json(config_json: str | Path) -> FitMoTNConfig:
    cfg = make_default_config()
    path = Path(config_json).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return apply_config_payload(cfg, payload)


def apply_config_overrides(cfg: FitMoTNConfig, overrides: Mapping[str, Mapping[str, Any]] | None) -> FitMoTNConfig:
    if not overrides:
        return cfg
    explicit_train_keys: set[str] = set()
    for section_name, section_values in overrides.items():
        if not section_values:
            continue
        section_obj = getattr(cfg, section_name, None)
        if section_obj is None:
            raise ValueError(f"Unknown override section: {section_name}")
        _apply_section_values(section_name, section_obj, _ensure_mapping(section_name, section_values))
        if section_name == "train":
            explicit_train_keys.update(section_values.keys())
    return _finalize_train_config(cfg, explicit_train_keys)


def load_config(config_json: str | Path | None = None, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> FitMoTNConfig:
    cfg = make_default_config() if config_json is None else load_config_from_json(config_json)
    if config_json is None:
        cfg = _finalize_train_config(cfg, set())
    return apply_config_overrides(cfg, overrides)
