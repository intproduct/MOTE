#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fitmotn.diagnostics.vllm_export_preflight import validate_vllm_export_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static fail-fast validation for a FitMoTN vLLM export")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--no-require-vllm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_remote_code = Path(__file__).resolve().parents[1] / "export" / "remote_code" / "modeling_fitmotn.py"
    report = validate_vllm_export_preflight(
        args.model,
        expected_remote_code=repository_remote_code,
        require_vllm=not bool(args.no_require_vllm),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        target = Path(args.output_json).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
