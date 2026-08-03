from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


SUPPORTED_PATH_ENV_VARS = {"MODEL_ROOT", "DATA_ROOT", "OUTPUT_ROOT", "CACHE_ROOT", "PROJECT_ROOT"}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_LOGGER = logging.getLogger(__name__)
_WARNING_CACHE: set[tuple[str, str, str]] = set()


def reset_path_warning_cache() -> None:
    _WARNING_CACHE.clear()


def _path_debug_enabled() -> bool:
    return os.environ.get("FITMOTN_PATH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _project_root() -> Path:
    root = os.environ.get("PROJECT_ROOT")
    if root is None or str(root).strip() == "":
        return _PACKAGE_ROOT
    return Path(root).expanduser().resolve()


def _dev_mode(cfg: Any) -> bool:
    runtime_cfg = getattr(cfg, "runtime", None)
    return bool(getattr(runtime_cfg, "dev_mode", False))


def _fallback_root_for_env(name: str) -> Path:
    names = {
        "MODEL_ROOT": "models",
        "DATA_ROOT": "data",
        "OUTPUT_ROOT": "outputs",
        "CACHE_ROOT": "cache",
        "PROJECT_ROOT": "",
    }
    suffix = names.get(name, name.lower())
    root = _project_root()
    return root if suffix == "" else root / suffix


def _audit(cfg: Any, *, key: str, resolved_path: str | None, source: str, severity: str) -> None:
    if cfg is None:
        return
    event = {
        "key": str(key),
        "resolved_path": None if resolved_path is None else str(resolved_path),
        "source": str(source),
        "severity": str(severity),
    }
    events = getattr(cfg, "_path_audit_events", None)
    if isinstance(cfg, dict):
        events = cfg.setdefault("_path_audit_events", [])
        if event not in events:
            events.append(event)
        return
    if events is None:
        events = []
        setattr(cfg, "_path_audit_events", events)
    if event not in events:
        events.append(event)


def _log_once(level: str, message: str, *, key: str, resolved_path: str | None) -> bool:
    cache_key = (str(key), str(resolved_path), str(level))
    if cache_key in _WARNING_CACHE:
        return False
    _WARNING_CACHE.add(cache_key)
    if level == "error":
        _LOGGER.error(message)
    elif level in {"warning", "fallback"}:
        _LOGGER.warning(message)
    elif level == "debug":
        _LOGGER.info(message)
    return True


def check_path_safety(path: str | None) -> str | None:
    if path is None:
        return None
    value = str(path)
    if value.startswith("/home/"):
        return "error"
    if value.startswith("/work/") or value.startswith("/ssd/"):
        return "warning"
    return None


def _enforce_safety(path: str, *, key: str, source: str, cfg: Any) -> str:
    severity = check_path_safety(path)
    if severity == "error":
        suffix = ", schema default used" if source == "default" else ""
        message = f"PATH_RESOLVE_ERROR: {key} -> {path} (source={source}{suffix})"
        _audit(cfg, key=key, resolved_path=path, source=source, severity="error")
        _log_once("error", message, key=key, resolved_path=path)
        raise ValueError(f"Unsafe path is not allowed for {key}: {path}")
    if severity == "warning":
        suffix = ", schema default used" if source == "default" else ""
        message = f"PATH_RESOLVE_WARNING: {key} -> {path} (source={source}{suffix})"
        _audit(cfg, key=key, resolved_path=path, source=source, severity="warning")
        _log_once("warning", message, key=key, resolved_path=path)
    return severity or "info"


def resolve_path(
    path: str | os.PathLike[str] | None,
    *,
    key: str = "path",
    source: str = "config",
    cfg: Any = None,
    allow_none: bool = True,
) -> str | None:
    if path is None:
        if allow_none:
            return None
        raise ValueError(f"Missing required path for {key}")
    raw = str(path).strip()
    if raw == "":
        if allow_none:
            return None
        raise ValueError(f"Missing required path for {key}")

    used_fallback = False

    def replace_env(match: re.Match[str]) -> str:
        nonlocal used_fallback
        name = match.group(1)
        if name not in SUPPORTED_PATH_ENV_VARS:
            raise ValueError(f"Unsupported path environment variable ${{{name}}} for {key}")
        value = os.environ.get(name)
        if value is not None and value.strip() != "":
            return value
        if not _dev_mode(cfg):
            raise EnvironmentError(f"Environment variable {name} is required to resolve {key}: {raw!r}")
        used_fallback = True
        return str(_fallback_root_for_env(name))

    expanded = _ENV_PATTERN.sub(replace_env, raw)
    # Preserve explicitly POSIX-absolute paths when validation is executed on
    # Windows.  Converting "/work/..." through WindowsPath would silently turn
    # it into "C:\\work\\...", bypassing the cross-machine safety policy and
    # mutating checkpoint/config provenance.
    if os.name == "nt" and expanded.startswith("/"):
        _enforce_safety(expanded, key=key, source=source, cfg=cfg)
        return expanded
    path_obj = Path(expanded).expanduser()
    if path_obj.is_absolute():
        _enforce_safety(str(path_obj), key=key, source=source, cfg=cfg)
    if not path_obj.is_absolute():
        path_obj = _project_root() / path_obj
    resolved = str(path_obj.resolve())

    severity = _enforce_safety(resolved, key=key, source=source, cfg=cfg)
    if source == "default":
        severity = "warning" if severity == "info" else severity
        _audit(cfg, key=key, resolved_path=resolved, source=source, severity=severity)
        _log_once(
            "warning",
            f"PATH_RESOLVE_WARNING: {key} -> {resolved} (source=default, schema default used)",
            key=key,
            resolved_path=resolved,
        )
    elif used_fallback:
        severity = "fallback"
        _audit(cfg, key=key, resolved_path=resolved, source=source, severity=severity)
        _log_once(
            "fallback",
            f"PATH_RESOLVE_FALLBACK: {key} -> {resolved} (source={source})",
            key=key,
            resolved_path=resolved,
        )
    elif _path_debug_enabled():
        _audit(cfg, key=key, resolved_path=resolved, source=source, severity="info")
        _log_once(
            "debug",
            f"PATH_RESOLVE: {key} -> {resolved} (source={source})",
            key=key,
            resolved_path=resolved,
        )
    return resolved


def _iter_strings(value: Any, location: str = "config"):
    if is_dataclass(value):
        yield from _iter_strings(asdict(value), location)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple, set)):
        for idx, item in enumerate(value):
            yield from _iter_strings(item, f"{location}[{idx}]")
    elif isinstance(value, str):
        yield location, value


def assert_no_unsafe_paths(cfg: Any, *, context: str, logger: logging.Logger | None = None) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    active_logger = logger or _LOGGER
    for key, value in _iter_strings(cfg, context):
        severity = check_path_safety(value)
        if severity is None:
            continue
        event = {"key": str(key), "resolved_path": str(value), "source": "runtime", "severity": severity}
        events.append(event)
        _audit(cfg, key=event["key"], resolved_path=event["resolved_path"], source="runtime", severity=severity)
        if severity == "error":
            message = f"PATH_RESOLVE_ERROR: {event['key']} -> {event['resolved_path']} (source=runtime)"
            logged = _log_once("error", message, key=event["key"], resolved_path=event["resolved_path"])
            if logged and active_logger is not _LOGGER:
                active_logger.error(message)
            raise ValueError(f"Unsafe path is not allowed for {event['key']}: {event['resolved_path']}")
        message = f"PATH_RESOLVE_WARNING: {event['key']} -> {event['resolved_path']} (source=runtime)"
        logged = _log_once("warning", message, key=event["key"], resolved_path=event["resolved_path"])
        if logged and active_logger is not _LOGGER:
            active_logger.warning(message)
    return events


def assert_no_forbidden_paths(config_obj: Any) -> None:
    assert_no_unsafe_paths(config_obj, context="config")
