from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from ..data.release import build_frozen_sft_release
from ..data.shard_loader import iter_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an immutable FitMoTN SFT release from canonical JSONL")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--pipeline_version", required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=bool(args.trust_remote_code))
    manifest = build_frozen_sft_release(
        iter_jsonl(Path(args.input_jsonl).expanduser().resolve()),
        args.output_dir,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        pipeline_version=str(args.pipeline_version),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
