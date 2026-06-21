from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..data.caching import download_and_cache_dataset, load_dataset_auto_cached
from ..runtime import normalize_hf_config


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"rl.train_json does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return [dict(row) for row in rows]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict):
        for key in ("train", "data", "examples", "samples"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value]
    raise ValueError(f"Unsupported rl.train_json format: {path}")


def _coerce_record(row: Dict[str, Any], idx: int, source: str) -> Optional[Dict[str, Any]]:
    question = row.get("question", row.get("prompt"))
    answer = row.get("answer")
    if question is None or answer is None:
        return None
    return {
        "question": str(question),
        "answer": str(answer),
        "idx": row.get("idx", idx),
        "source": source,
    }


def _limit_records(records: Iterable[Dict[str, Any]], debug_num_prompts: Optional[int]) -> List[Dict[str, Any]]:
    if debug_num_prompts is None:
        return list(records)
    return list(islice(records, int(debug_num_prompts)))


def load_gsm8k_rl_records(fit_cfg, logger=None) -> List[Dict[str, Any]]:
    rl_cfg = fit_cfg.rl
    debug_num_prompts = getattr(rl_cfg, "debug_num_prompts", None)
    if getattr(rl_cfg, "train_json", None):
        records = []
        for idx, row in enumerate(read_json_or_jsonl(rl_cfg.train_json)):
            record = _coerce_record(row, idx, source=str(getattr(rl_cfg, "train_source", "train_json")))
            if record is not None:
                records.append(record)
            if debug_num_prompts is not None and len(records) >= int(debug_num_prompts):
                break
        if not records:
            raise ValueError("No usable RL records found; expected question/answer or prompt/answer fields")
        if logger is not None:
            logger.info("[RLData] source=train_json path=%s records=%s", rl_cfg.train_json, len(records))
        return records

    if not bool(getattr(rl_cfg, "use_config_data", True)):
        raise ValueError("rl.train_json is required when rl.use_config_data=false")
    if str(getattr(rl_cfg, "train_source", "gsm8k_train")) != "gsm8k_train":
        raise ValueError("Only rl.train_source='gsm8k_train' is supported in this version")

    data_cfg = fit_cfg.data
    res = load_dataset_auto_cached(
        data_cfg.gsm8k_cache_path,
        "train",
        hf_name=data_cfg.gsm8k_hf_name,
        hf_config=normalize_hf_config(data_cfg.gsm8k_hf_config),
    )
    source_kind = "cached"
    if isinstance(res, tuple) and len(res) == 6:
        _, ready, cache_path, hf_name, hf_config, split = res
        source_kind = "downloaded_or_waited"
        res = download_and_cache_dataset(cache_path, ready, hf_name, hf_config, split)

    records = []
    for idx, row in enumerate(res):
        record = _coerce_record(dict(row), idx, source="gsm8k_train")
        if record is not None:
            records.append(record)
        if debug_num_prompts is not None and len(records) >= int(debug_num_prompts):
            break
    records = _limit_records(records, debug_num_prompts)
    if not records:
        raise ValueError("No usable GSM8K train records found in configured data source")
    if logger is not None:
        logger.info("[RLData] source=%s path=%s records=%s", source_kind, data_cfg.gsm8k_cache_path, len(records))
    return records


def format_rl_prompt(template: str, question: str) -> str:
    if "{question}" in template:
        return template.format(question=question)
    return f"{template}{question}"
