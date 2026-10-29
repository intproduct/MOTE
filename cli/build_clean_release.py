from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..data.clean_release import build_clean_release_from_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an immutable, tokenizer-independent FitMoTN clean data release")
    parser.add_argument("--registry", required=True, help="Local data registry JSON")
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    registry_path = Path(os.path.expandvars(args.registry)).expanduser().resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    manifest = build_clean_release_from_registry(registry, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
