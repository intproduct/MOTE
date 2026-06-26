from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as tc

from ..config import load_config
from ..eval.runner import run_eval_tasks
from ..eval.restore import restore_fitmotn_model
from ..export.format import EXPORT_MANIFEST_FILENAME, EXPORT_STAGE_HF_ROUNDTRIP, classify_model_path
from ..runtime import load_causal_lm_and_tokenizer

def make_json_safe(obj):
    import dataclasses
    import enum
    from pathlib import Path
    from collections.abc import Mapping

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if dataclasses.is_dataclass(obj):
        return make_json_safe(dataclasses.asdict(obj))

    if isinstance(obj, enum.Enum):
        return make_json_safe(obj.value)

    if isinstance(obj, Path):
        return str(obj)

    try:
        import numpy as np

        if isinstance(obj, np.dtype):
            return str(obj)

        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    try:
        import torch

        if isinstance(obj, torch.dtype):
            return str(obj)

        if isinstance(obj, torch.device):
            return str(obj)

        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(obj, Mapping):
        return {str(make_json_safe(k)): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    return str(obj)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_or_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def _export_stage(target: Path) -> str | None:
    try:
        with (target / EXPORT_MANIFEST_FILENAME).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("export_stage")
    except Exception:
        return None


def _load_model_and_tokenizer(target: Path, cfg, device: str):
    path_kind = classify_model_path(target)
    if path_kind == "raw_fitmotn_checkpoint":
        return restore_fitmotn_model(target, device=device)
    if path_kind == "exported_fitmotn_dir":
        stage = _export_stage(target)
        if stage != EXPORT_STAGE_HF_ROUNDTRIP:
            raise ValueError(
                "Metadata-only FitMoTN exports are not loadable by eval_hf. "
                "Run `python -m fitmotn.cli.export_hf --checkpoint_dir ... --output_dir ... --no-metadata_only` first."
            )
        return load_causal_lm_and_tokenizer(
            target,
            device=tc.device(device),
            trust_remote_code=True,
            torch_dtype=cfg.model.torch_dtype,
            use_cache=True,
        )
    return load_causal_lm_and_tokenizer(
        target,
        device=tc.device(device),
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=cfg.model.torch_dtype,
        use_cache=True,
    )


def main():
    args = parse_args()
    cfg = load_config(
        config_json=args.config_json,
        overrides={
            "eval": {
                "eval_backend": "lm_eval",
                "primary_eval_backend": "lm_eval",
                "lm_eval_device": args.device,
                "backend_defaults": {"lm_eval": {"device": args.device}},
            }
        },
    )
    target = Path(args.model_or_ckpt).resolve()
    model, tokenizer, _ = _load_model_and_tokenizer(target, cfg, args.device)
    tasks = args.tasks or cfg.eval.final_tasks
    results = run_eval_tasks(
        cfg,
        tasks=tasks,
        eval_name="eval_hf",
        eval_mode="final",
        model=model,
        tokenizer=tokenizer,
        model_or_path=target,
    )
    output_json = Path(args.output_json).resolve() if args.output_json else target / "fitmotn_eval_hf.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
