from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


VAGUE_REVISIONS = {"", "latest", "main", "master", "unknown", "head", "todo"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a pinned Hugging Face dataset revision into an immutable local cache")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face commit or release tag")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--allowed_use", action="append", required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    revision = str(args.revision).strip()
    if revision.lower() in VAGUE_REVISIONS or "replace_with" in revision.lower():
        raise ValueError("--revision must be an immutable commit or release tag")
    target = Path(os.path.expandvars(args.output_dir)).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset cache: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))).resolve()
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            args.dataset,
            args.config,
            split=args.split,
            revision=revision,
            trust_remote_code=bool(args.trust_remote_code),
        )
        dataset.save_to_disk(str(temporary / "dataset"))
        metadata = {
            "format": "fitmotn_hf_cache_receipt_v1",
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "revision": revision,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "rows": len(dataset) if hasattr(dataset, "__len__") else None,
            "license": str(args.license).strip().lower(),
            "allowed_uses": sorted({str(value).strip().lower() for value in args.allowed_use if str(value).strip()}),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "source_receipt.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "_READY").write_text("ok\n", encoding="utf-8")
        os.replace(temporary, target)
        print(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except Exception:
        if temporary.exists() and temporary.parent == target.parent and temporary.name.startswith(f".{target.name}.tmp-"):
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
