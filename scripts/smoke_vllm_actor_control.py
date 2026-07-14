#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fitmotn.rl.vllm_actor import VLLMActorClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the vLLM subprocess actor control channel without loading CUDA")
    parser.add_argument("--start-method", default="spawn", choices=["spawn", "forkserver"])
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    args = parser.parse_args()
    client = VLLMActorClient(
        start_method=args.start_method,
        request_timeout_sec=args.timeout_sec,
        shutdown_timeout_sec=args.timeout_sec,
    )
    try:
        client.start()
        result = client.ping()
        print(json.dumps({"ok": True, "actor": result}, ensure_ascii=False, indent=2))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
