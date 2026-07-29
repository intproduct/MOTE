from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer

from ..data.training_release import bind_clean_release


def _csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind a tokenizer-independent clean release to a training contract")
    parser.add_argument("--clean_release_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer_revision", required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--planes", required=True, help="Comma-separated data planes; one training objective per release")
    parser.add_argument("--length_buckets", default="", help="Comma-separated inclusive token boundaries")
    parser.add_argument("--packing_mode", choices=["none", "pretrain_greedy"], default="none")
    parser.add_argument("--workers", type=int, default=1, help="Ordered tokenizer worker threads")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=bool(args.trust_remote_code),
    )
    manifest = bind_clean_release(
        args.clean_release_dir,
        args.output_dir,
        tokenizer=tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        max_length=args.max_length,
        selected_planes=_csv(args.planes),
        length_buckets=[int(value) for value in _csv(args.length_buckets)],
        packing_mode=args.packing_mode,
        tokenization_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
