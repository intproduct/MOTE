from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..diagnostics.config_doctor import inspect_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight a FitMoTN experiment configuration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    try:
        cfg = load_config(config_json=args.config)
        result = inspect_config(cfg)
    except Exception as exc:
        result = {"ok": False, "errors": [str(exc)], "warnings": []}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        from pathlib import Path

        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
