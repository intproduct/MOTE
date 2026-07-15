from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..checkpointing import validate_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a committed FitMoTN SFT/RL checkpoint")
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--require-exact", choices=["sft", "rl"])
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    result = validate_checkpoint(
        args.checkpoint_dir,
        verify_hashes=not args.skip_hashes,
        require_exact_resume=args.require_exact,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
