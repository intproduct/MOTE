from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


SYNC_MANIFEST_NAME = "fitmotn_vllm_sync_manifest.json"
SYNC_STATE_NAME = "fitmotn_vllm_sync_state.json"


def canonical_json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_batch_fingerprint(prompt_token_ids: Sequence[Sequence[int]]) -> str:
    return canonical_json_fingerprint([[int(token) for token in row] for row in prompt_token_ids])


def sampling_fingerprint(sampling_kwargs: Mapping[str, Any]) -> str:
    return canonical_json_fingerprint(dict(sampling_kwargs))


def _sample_tensor(tensor: torch.Tensor, sample_elements: int) -> Iterable[float | int | bool]:
    flat = tensor.detach().reshape(-1)
    count = int(flat.numel())
    if count == 0:
        return []
    take = min(count, max(1, int(sample_elements)))
    if take == count:
        sample = flat
    else:
        indices = torch.linspace(0, count - 1, steps=take, device=flat.device).round().long()
        sample = flat.index_select(0, indices)
    sample = sample.to(device="cpu")
    if sample.is_floating_point() or sample.is_complex():
        return [float(value) for value in sample.float().tolist()]
    return sample.tolist()


def model_policy_fingerprint(model: Any, *, sample_elements_per_tensor: int = 16) -> str:
    """Return a cheap deterministic identity for a policy snapshot.

    This intentionally samples every trainable parameter (or every parameter
    when none is marked trainable) instead of copying full model weights to
    CPU. Frozen base weights are identified by model type/config metadata.
    This is a freshness/provenance guard, not a cryptographic hash of complete
    checkpoint bytes.
    """

    if model is None:
        return canonical_json_fingerprint({"model": None})
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return canonical_json_fingerprint({"type": type(model).__qualname__})
    parameters = list(named_parameters())
    trainable = [(name, tensor) for name, tensor in parameters if bool(tensor.requires_grad)]
    selected = trainable or parameters
    return model_named_parameter_fingerprint(
        model,
        parameter_names=[str(name) for name, _tensor in selected],
        sample_elements_per_tensor=sample_elements_per_tensor,
        selection_label=f"policy_selected={len(selected)};total={len(parameters)}",
    )


def model_named_parameter_fingerprint(
    model: Any,
    *,
    parameter_names: Sequence[str],
    sample_elements_per_tensor: int = 16,
    selection_label: str = "explicit",
) -> str:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return canonical_json_fingerprint({"type": type(model).__qualname__, "selection": selection_label})
    parameters = dict(named_parameters())
    requested = {str(name) for name in parameter_names}
    missing = sorted(requested - set(parameters))
    if missing:
        raise RuntimeError(f"fingerprint selection contains unknown parameters: {missing}")
    digest = hashlib.sha256()
    digest.update(type(model).__qualname__.encode("utf-8"))
    digest.update(str(selection_label).encode("utf-8"))
    config = getattr(model, "config", None)
    digest.update(
        canonical_json_fingerprint(
            {
                "name_or_path": getattr(config, "_name_or_path", None),
                "model_type": getattr(config, "model_type", None),
                "architectures": getattr(config, "architectures", None),
            }
        ).encode("ascii")
    )
    digest.update(f"selected={len(requested)};total={len(parameters)}".encode("ascii"))
    for name in sorted(requested):
        tensor = parameters[name]
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        values = _sample_tensor(tensor, int(sample_elements_per_tensor))
        digest.update(json.dumps(list(values), separators=(",", ":"), allow_nan=True).encode("ascii"))
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def directory_size_bytes(path: Path) -> int:
    total = 0
    for item in Path(path).rglob("*"):
        try:
            if item.is_file():
                total += int(item.stat().st_size)
        except OSError:
            continue
    return total
