from __future__ import annotations

import argparse
import json

from ..data.release import preflight_frozen_sft_release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a frozen SFT release without tokenizing or training")
    parser.add_argument("--release_dir", required=True)
    parser.add_argument("--min_records", type=int, default=1000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = preflight_frozen_sft_release(args.release_dir, min_records=int(args.min_records))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
