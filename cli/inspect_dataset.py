from __future__ import annotations

import argparse
import sys
from itertools import islice
from typing import Any

from ..config.loader import load_config
from ..data.adapters import normalize_chat_like_example, normalize_text_example
from ..data.mixed_iterable import apply_task_sample_limit, load_dataset_any
from ..data.specs import HFChatTask, HFTextTask
from ..data.tokenization import make_causal_lm_example_from_text, make_chat_supervised_example
from ..tasks.extra_dataset_specs import build_extra_dataset_tasks


def _load_tokenizer(path: str | None):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"transformers is required for --tokenizer: {exc!r}") from exc
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _tokenized_len(example: dict[str, Any]) -> int:
    ids = example.get("input_ids")
    try:
        return int(ids.numel())
    except Exception:
        return len(ids or [])


def inspect_task(task, *, max_samples: int, tokenizer=None) -> bool:
    print(
        f"\n[{task.name}] kind={task.kind} source={task.metadata.get('source')} split={task.split} "
        f"format={task.metadata.get('dataset_format', 'text')} group={task.group} bucket={task.bucket}"
    )
    raw = load_dataset_any(task.kind, task.path, task.split, hf_name=task.hf_name, hf_config=task.hf_config, metadata=task.metadata)
    ok = True
    for idx, ex in enumerate(islice(iter(apply_task_sample_limit(raw, task)), max_samples)):
        keys = sorted(list(ex.keys())) if isinstance(ex, dict) else []
        print(f"  sample[{idx}] raw_keys={keys}")
        try:
            if isinstance(task, HFTextTask):
                text = normalize_text_example(ex, text_field=task.text_field)
                if text is None:
                    raise ValueError("adapter returned None: missing/empty text")
                print(f"    normalized=text chars={len(text)}")
                if tokenizer is not None:
                    tokenized = make_causal_lm_example_from_text(tokenizer, text, max_len=4096, add_eos=True)
                    print(f"    tokenized_len={_tokenized_len(tokenized)}")
            elif isinstance(task, HFChatTask):
                messages = normalize_chat_like_example(ex, task)
                if messages is None:
                    raise ValueError("adapter returned None: missing fields, empty content, or no assistant message")
                roles = [msg["role"] for msg in messages]
                chars = sum(len(msg.get("content", "")) for msg in messages)
                print(f"    normalized=messages roles={roles} chars={chars}")
                if tokenizer is not None:
                    tokenized = make_chat_supervised_example(tokenizer, messages, max_len=4096, add_eos=True)
                    print(f"    tokenized_len={_tokenized_len(tokenized)}")
            else:
                print("    skipped: not an extra HFTextTask/HFChatTask")
        except Exception as exc:
            ok = False
            print(f"    ERROR: {exc!r}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect configured FitMoTN extra datasets.")
    parser.add_argument("--config", required=True, help="Path to FitMoTN JSON config.")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer/model path for tokenization dry-run.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tokenizer = _load_tokenizer(args.tokenizer)
    tasks = build_extra_dataset_tasks(cfg.data, group="pretrain") + build_extra_dataset_tasks(cfg.data, group="task")
    if not tasks:
        print("No enabled data.extra_datasets tasks found.")
        return 1
    ok = True
    for task in tasks:
        ok = inspect_task(task, max_samples=max(1, int(args.max_samples)), tokenizer=tokenizer) and ok
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
