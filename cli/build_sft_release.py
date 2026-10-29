from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from ..data.release import build_frozen_sft_release
from ..data.shard_loader import iter_jsonl
from ..data.source_pipeline import build_release_from_source_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an immutable FitMoTN SFT release from canonical JSONL or local source caches")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input_jsonl", help="Pre-canonicalized JSONL input (compatibility path)")
    inputs.add_argument("--source_manifest", help="Production source-manifest JSON for local source adapters")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer_revision", help="Immutable tokenizer commit/release fingerprint recorded in the manifest")
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--pipeline_version", required=True)
    parser.add_argument("--format_version", default="legacy")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=bool(args.trust_remote_code))
    if args.source_manifest:
        if not args.tokenizer_revision:
            raise ValueError("--tokenizer_revision is required for production --source_manifest builds")
        source_manifest_path = Path(os.path.expandvars(args.source_manifest)).expanduser().resolve()
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        manifest = build_release_from_source_manifest(
            source_manifest,
            args.output_dir,
            tokenizer=tokenizer,
            tokenizer_revision=str(args.tokenizer_revision),
            max_length=int(args.max_length),
            pipeline_version=str(args.pipeline_version),
        )
    else:
        manifest = build_frozen_sft_release(
            iter_jsonl(Path(os.path.expandvars(args.input_jsonl)).expanduser().resolve()),
            args.output_dir,
            tokenizer=tokenizer,
            max_length=int(args.max_length),
            pipeline_version=str(args.pipeline_version),
            format_version=str(args.format_version),
            tokenizer_revision=str(args.tokenizer_revision or args.tokenizer),
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
