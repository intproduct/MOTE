from __future__ import annotations

import argparse
from pathlib import Path

from ..export.format import EXPORT_CONFIG_FILENAME, EXPORT_MANIFEST_FILENAME, classify_model_path
from ..export.manifest import build_export_payloads, write_json
from ..export.validate import validate_export_layout


README_TEXT = """# FitMoTN metadata-only export

This directory was produced by `python -m fitmotn.cli.export_hf` during Stage 4A.

It contains metadata for a future Hugging Face/vLLM-compatible FitMoTN export, but it is not loadable with
`AutoModelForCausalLM.from_pretrained(...)` yet and does not enable patched FitMoTN vLLM inference.

Do not commit private local paths, checkpoints, exported weights, or tokenizer files copied from private models.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a raw FitMoTN checkpoint into a metadata-only HF export layout.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--metadata_only", dest="metadata_only", action="store_true", default=True)
    parser.add_argument("--no-metadata_only", dest="metadata_only", action="store_false")
    parser.add_argument("--copy_tokenizer", action="store_true", default=False)
    parser.add_argument("--validate", dest="validate", action="store_true", default=True)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    return parser.parse_args(argv)


def _ensure_output_ready(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"output_dir exists and is not a directory: {output_dir}")
    if any(output_dir.iterdir()) and not overwrite:
        raise ValueError(f"output_dir exists and is non-empty: {output_dir}. Pass --overwrite to replace Stage 4A metadata files.")


def _copy_tokenizer(tokenizer_source: str, output_dir: Path) -> None:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise ImportError("--copy_tokenizer requires transformers to be installed") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.metadata_only:
        raise ValueError("Stage 4A only supports --metadata_only; HF AutoModel roundtrip export starts in Stage 4B.")

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    input_kind = classify_model_path(checkpoint_dir)
    if input_kind == "missing":
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")
    if input_kind == "exported_fitmotn_dir":
        raise ValueError("Input is already an exported FitMoTN directory; Stage 4A expects a raw FitMoTN checkpoint.")
    if input_kind == "hf_model_dir":
        raise ValueError("Input appears to be an ordinary HF model directory; Stage 4A exports raw FitMoTN checkpoints only.")
    if input_kind != "raw_fitmotn_checkpoint":
        raise ValueError(f"checkpoint_dir is not a recognized raw FitMoTN checkpoint: {checkpoint_dir} ({input_kind})")

    _ensure_output_ready(output_dir, bool(args.overwrite))
    manifest, export_config = build_export_payloads(checkpoint_dir, base_model=args.base_model, tokenizer=args.tokenizer)

    if args.dry_run:
        print(f"[export_hf] dry run: would export {checkpoint_dir} -> {output_dir}")
        print(f"[export_hf] would write: {EXPORT_MANIFEST_FILENAME}, {EXPORT_CONFIG_FILENAME}, README.md")
        if args.copy_tokenizer:
            print(f"[export_hf] would copy tokenizer from: {export_config['tokenizer_source']}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / EXPORT_MANIFEST_FILENAME, manifest)
    write_json(output_dir / EXPORT_CONFIG_FILENAME, export_config)
    (output_dir / "README.md").write_text(README_TEXT, encoding="utf-8")
    if args.copy_tokenizer:
        _copy_tokenizer(str(export_config["tokenizer_source"]), output_dir)

    if args.validate:
        result = validate_export_layout(output_dir)
        if not result.ok:
            for err in result.errors:
                print(f"[export_hf] validation error: {err['message']}")
            raise ValueError(f"export layout validation failed for {output_dir}")
        for warning in result.warnings:
            print(f"[export_hf] validation warning: {warning['message']}")

    print(f"[export_hf] wrote metadata-only FitMoTN export: {output_dir}")
    print(
        "[export_hf] Stage 4B will add configuration_fitmotn.py, modeling_fitmotn.py, "
        "auto_map, weights, and HF AutoModel roundtrip support."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
