from __future__ import annotations

import argparse
from pathlib import Path

from ..export.format import EXPORT_CONFIG_FILENAME, EXPORT_MANIFEST_FILENAME, classify_model_path
from ..export.hf_export import export_fitmotn_hf_roundtrip
from ..export.manifest import build_export_payloads, write_json
from ..export.roundtrip import validate_hf_roundtrip
from ..export.validate import validate_export_layout


README_TEXT = """# FitMoTN metadata-only export

This directory was produced by `python -m fitmotn.cli.export_hf` during Stage 4A.

It contains metadata for a future Hugging Face/vLLM-compatible FitMoTN export, but it is not loadable with
`AutoModelForCausalLM.from_pretrained(...)` yet and does not enable patched FitMoTN vLLM inference.

Do not commit private local paths, checkpoints, exported weights, or tokenizer files copied from private models.
"""


def _str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a raw FitMoTN checkpoint into a FitMoTN HF export layout.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--metadata_only", dest="metadata_only", action="store_true", default=True)
    parser.add_argument("--no-metadata_only", dest="metadata_only", action="store_false")
    parser.add_argument("--copy_tokenizer", "--copy-tokenizer", dest="copy_tokenizer", action="store_true", default=None)
    parser.add_argument("--no-copy_tokenizer", "--no-copy-tokenizer", dest="copy_tokenizer", action="store_false")
    parser.add_argument("--safe_serialization", type=_str_bool, default=True)
    parser.add_argument("--max_shard_size", type=str, default="5GB")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--validate_layout", dest="validate_layout", action="store_true", default=True)
    parser.add_argument("--no-validate_layout", dest="validate_layout", action="store_false")
    parser.add_argument("--validate", dest="validate_layout", action="store_true")
    parser.add_argument("--no-validate", dest="validate_layout", action="store_false")
    parser.add_argument("--validate_roundtrip", dest="validate_roundtrip", action="store_true", default=False)
    parser.add_argument("--no-validate_roundtrip", dest="validate_roundtrip", action="store_false")
    parser.add_argument("--roundtrip_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--base_trust_remote_code", action="store_true", default=False)
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


def _check_input_kind(checkpoint_dir: Path, *, metadata_only: bool) -> str:
    input_kind = classify_model_path(checkpoint_dir)
    if input_kind == "missing":
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")
    if input_kind == "exported_fitmotn_dir":
        raise ValueError("Input is already an exported FitMoTN directory; raw FitMoTN checkpoint input is required.")
    if input_kind == "hf_model_dir":
        raise ValueError("Input appears to be an ordinary HF model directory; raw FitMoTN checkpoint input is required.")
    if input_kind != "raw_fitmotn_checkpoint":
        raise ValueError(f"checkpoint_dir is not a recognized raw FitMoTN checkpoint: {checkpoint_dir} ({input_kind})")
    return input_kind


def _print_layout_errors(output_dir: Path) -> None:
    result = validate_export_layout(output_dir)
    if not result.ok:
        for err in result.errors:
            print(f"[export_hf] validation error: {err['message']}")
        raise ValueError(f"export layout validation failed for {output_dir}")
    for warning in result.warnings:
        print(f"[export_hf] validation warning: {warning['message']}")


def _resolve_roundtrip_device(value: str) -> str:
    if str(value).lower() != "auto":
        return str(value)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _dry_run_full_export(args: argparse.Namespace, checkpoint_dir: Path, output_dir: Path, copy_tokenizer: bool) -> None:
    from ..export.manifest import read_raw_checkpoint_json

    metadata, _ = read_raw_checkpoint_json(checkpoint_dir)
    base_model = args.base_model or metadata.get("base_model_path")
    if not base_model:
        raise ValueError("base_model_name_or_path could not be inferred; pass --base_model")
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(str(base_model), trust_remote_code=bool(args.base_trust_remote_code))
    except Exception as exc:
        raise ValueError(f"dry run could not resolve base_model config {base_model!r}: {exc}") from exc
    print(f"[export_hf] dry run: would full-export {checkpoint_dir} -> {output_dir}")
    print("[export_hf] would write: config.json, weights, custom code, fitmotn export metadata, README.md")
    if copy_tokenizer:
        print(f"[export_hf] would copy tokenizer from: {args.tokenizer or metadata.get('tokenizer_path') or base_model}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    _check_input_kind(checkpoint_dir, metadata_only=bool(args.metadata_only))
    copy_tokenizer = bool((not args.metadata_only) if args.copy_tokenizer is None else args.copy_tokenizer)
    roundtrip_device = _resolve_roundtrip_device(str(args.roundtrip_device))

    _ensure_output_ready(output_dir, bool(args.overwrite))

    if args.dry_run:
        if not args.metadata_only:
            _dry_run_full_export(args, checkpoint_dir, output_dir, copy_tokenizer)
            return 0
        manifest, export_config = build_export_payloads(checkpoint_dir, base_model=args.base_model, tokenizer=args.tokenizer)
        print(f"[export_hf] dry run: would metadata-export {checkpoint_dir} -> {output_dir}")
        print(f"[export_hf] would write: {EXPORT_MANIFEST_FILENAME}, {EXPORT_CONFIG_FILENAME}, README.md")
        if copy_tokenizer:
            print(f"[export_hf] would copy tokenizer from: {export_config['tokenizer_source']}")
        return 0

    if not args.metadata_only:
        result = export_fitmotn_hf_roundtrip(
            checkpoint_dir,
            output_dir,
            base_model=args.base_model,
            tokenizer_source=args.tokenizer,
            copy_tokenizer=copy_tokenizer,
            safe_serialization=bool(args.safe_serialization),
            max_shard_size=str(args.max_shard_size),
            torch_dtype=str(args.torch_dtype),
            roundtrip_device=roundtrip_device,
            base_trust_remote_code=bool(args.base_trust_remote_code),
        )
        if args.validate_layout:
            _print_layout_errors(result.output_dir)
        if args.validate_roundtrip:
            rt = validate_hf_roundtrip(result.output_dir, device=roundtrip_device, torch_dtype=str(args.torch_dtype))
            if not rt.ok:
                for err in rt.errors:
                    print(f"[export_hf] roundtrip validation error: {err['message']}")
                raise ValueError(f"HF roundtrip validation failed for {result.output_dir}")
            for warning in rt.warnings:
                print(f"[export_hf] roundtrip validation warning: {warning['message']}")
        print(f"[export_hf] wrote HF roundtrip FitMoTN export: {result.output_dir}")
        print("[export_hf] HF roundtrip export is ready. Stage 4C will add vLLM offline runner support.")
        return 0

    manifest, export_config = build_export_payloads(checkpoint_dir, base_model=args.base_model, tokenizer=args.tokenizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / EXPORT_MANIFEST_FILENAME, manifest)
    write_json(output_dir / EXPORT_CONFIG_FILENAME, export_config)
    (output_dir / "README.md").write_text(README_TEXT, encoding="utf-8")
    if copy_tokenizer:
        _copy_tokenizer(str(export_config["tokenizer_source"]), output_dir)

    if args.validate_layout:
        _print_layout_errors(output_dir)

    print(f"[export_hf] wrote metadata-only FitMoTN export: {output_dir}")
    print("[export_hf] For HF AutoModel roundtrip export, rerun with --no-metadata_only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
