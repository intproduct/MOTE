from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .defaults import make_default_config
from .schema import FitMoTNConfig


def _ensure_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object/dict, got {type(value).__name__}")
    return value


def _apply_section_values(section_name: str, section_obj: Any, values: Mapping[str, Any]) -> None:
    if not is_dataclass(section_obj):
        raise TypeError(f"config section {section_name} is not a dataclass")
    valid_fields = {field.name for field in fields(section_obj)}
    unknown = sorted(set(values.keys()) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown keys in config section '{section_name}': {', '.join(unknown)}")
    for key, value in values.items():
        setattr(section_obj, key, value)


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
    return cfg


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
