from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RAW_TEMPLATE = "{question}"
GSM8K_DIRECT_TEMPLATE = "Question:\n{question}\nAnswer:\n"
GSM8K_COT_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Solve the problem step by step. Put the final numeric answer after ####.\n"
    "Answer:\n"
)


def _arg_value(args, name: str, default=None):
    return getattr(args, name, default)


def _diag_cfg(fit_cfg):
    return getattr(fit_cfg, "diagnostics", None) if fit_cfg is not None else None


def _num_samples(args, fit_cfg) -> int:
    value = _arg_value(args, "num_samples", None)
    if value is None:
        cfg = _diag_cfg(fit_cfg)
        value = getattr(cfg, "num_samples", None) if cfg is not None else None
    if value is None:
        value = _arg_value(args, "max_prompts", 32)
    return max(1, int(value))


def _sample_seed(args, fit_cfg) -> int:
    value = _arg_value(args, "sample_seed", None)
    if value is None:
        cfg = _diag_cfg(fit_cfg)
        value = getattr(cfg, "sample_seed", 0) if cfg is not None else 0
    return int(value)


def _sample_split(args, fit_cfg) -> str:
    value = _arg_value(args, "sample_split", None)
    if value is None:
        cfg = _diag_cfg(fit_cfg)
        value = getattr(cfg, "sample_split", "train") if cfg is not None else "train"
    return str(value or "train")


def _prompt_template_name(args, fit_cfg) -> str:
    value = _arg_value(args, "prompt_template", None)
    if value is None:
        cfg = _diag_cfg(fit_cfg)
        value = getattr(cfg, "prompt_template", "config") if cfg is not None else "config"
    return str(value or "config").strip().lower()


def _tasks(args, fit_cfg) -> List[str]:
    cli_tasks = _arg_value(args, "tasks", None)
    if cli_tasks:
        return [str(t) for t in cli_tasks]
    cfg = _diag_cfg(fit_cfg)
    tasks = getattr(cfg, "tasks", None) if cfg is not None else None
    return [str(t) for t in tasks] if tasks else []


def _prompts_file(args, fit_cfg) -> Optional[str]:
    value = _arg_value(args, "prompts_file", None)
    if value:
        return str(value)
    cfg = _diag_cfg(fit_cfg)
    value = getattr(cfg, "prompts_file", None) if cfg is not None else None
    return None if value is None or str(value).strip() == "" else str(value)


def _messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages)
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", "") or "")
            content = str(msg.get("content", "") or "")
            parts.append(f"{role}: {content}" if role else content)
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def resolve_prompt_template(args, fit_cfg=None) -> str:
    name = _prompt_template_name(args, fit_cfg)
    if name == "raw":
        return RAW_TEMPLATE
    if name == "gsm8k_direct":
        return GSM8K_DIRECT_TEMPLATE
    if name == "gsm8k_cot":
        return GSM8K_COT_TEMPLATE
    if name == "config":
        rl_cfg = getattr(fit_cfg, "rl", None) if fit_cfg is not None else None
        template = getattr(rl_cfg, "prompt_template", None) if rl_cfg is not None else None
        if template:
            return str(template)
        return GSM8K_COT_TEMPLATE
    raise ValueError("prompt_template must be one of: config, raw, gsm8k_direct, gsm8k_cot")


def _format_question(question: Any, args, fit_cfg) -> str:
    from ..rl.data import format_rl_prompt

    return format_rl_prompt(resolve_prompt_template(args, fit_cfg), str(question))


def _row_to_prompt(row: Dict[str, Any], args, fit_cfg) -> Optional[str]:
    if "prompt" in row and row["prompt"] is not None:
        return str(row["prompt"])
    if "text" in row and row["text"] is not None:
        return str(row["text"])
    if "question" in row and row["question"] is not None:
        return _format_question(row["question"], args, fit_cfg)
    if "messages" in row and row["messages"] is not None:
        return _messages_to_text(row["messages"])
    return None


def _sample(items: List[str], num_samples: int, seed: int) -> List[str]:
    if len(items) <= num_samples:
        return items
    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    return [items[i] for i in idxs[:num_samples]]


def _load_file_prompts(path: str | Path, args, fit_cfg) -> List[str]:
    from ..rl.data import read_json_or_jsonl

    rows = read_json_or_jsonl(path)
    prompts = []
    for row in rows:
        prompt = _row_to_prompt(row, args, fit_cfg)
        if prompt is not None and prompt.strip():
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No usable prompts found in {path}; expected prompt/text/question/messages fields")
    return _sample(prompts, _num_samples(args, fit_cfg), _sample_seed(args, fit_cfg))


def _load_gsm8k_prompts(args, fit_cfg, logger=None) -> List[str]:
    split = _sample_split(args, fit_cfg)
    if fit_cfg is None:
        from ..config import load_config

        fit_cfg = load_config()
    if split == "train":
        from ..rl.data import load_gsm8k_rl_records

        records = load_gsm8k_rl_records(fit_cfg, logger=logger)
    else:
        from ..data.caching import download_and_cache_dataset, load_dataset_auto_cached
        from ..runtime import normalize_hf_config

        data_cfg = fit_cfg.data
        res = load_dataset_auto_cached(
            data_cfg.gsm8k_cache_path,
            split,
            hf_name=data_cfg.gsm8k_hf_name,
            hf_config=normalize_hf_config(data_cfg.gsm8k_hf_config),
        )
        if isinstance(res, tuple) and len(res) == 6:
            _, ready, cache_path, hf_name, hf_config, split_name = res
            res = download_and_cache_dataset(cache_path, ready, hf_name, hf_config, split_name)
        records = [dict(row) for row in res]
    prompts = [_format_question(row["question"], args, fit_cfg) for row in records if row.get("question") is not None]
    return _sample(prompts, _num_samples(args, fit_cfg), _sample_seed(args, fit_cfg))


def _format_mmlu_row(row: Dict[str, Any]) -> str:
    question = str(row.get("question", ""))
    choices = row.get("choices") or row.get("options") or []
    if isinstance(choices, dict):
        choices = [f"{k}. {v}" for k, v in choices.items()]
    elif isinstance(choices, list):
        labels = ["A", "B", "C", "D", "E", "F"]
        choices = [f"{labels[i]}. {v}" if i < len(labels) else str(v) for i, v in enumerate(choices)]
    suffix = "\n".join(str(c) for c in choices)
    return f"Question:\n{question}\n{suffix}\nAnswer:\n" if suffix else f"Question:\n{question}\nAnswer:\n"


def _load_mmlu_prompts(args, fit_cfg, logger=None) -> List[str]:
    if fit_cfg is None:
        from ..config import load_config

        fit_cfg = load_config()
    from ..data.caching import download_and_cache_dataset, load_dataset_auto_cached
    from ..runtime import normalize_hf_config

    data_cfg = fit_cfg.data
    split = _sample_split(args, fit_cfg) or getattr(data_cfg, "mmlu_split", "dev")
    res = load_dataset_auto_cached(
        data_cfg.mmlu_cache_path,
        split,
        hf_name=data_cfg.mmlu_hf_name,
        hf_config=normalize_hf_config(data_cfg.mmlu_hf_config),
    )
    if isinstance(res, tuple) and len(res) == 6:
        _, ready, cache_path, hf_name, hf_config, split_name = res
        res = download_and_cache_dataset(cache_path, ready, hf_name, hf_config, split_name)
    prompts = [_format_mmlu_row(dict(row)) for row in res]
    return _sample([p for p in prompts if p.strip()], _num_samples(args, fit_cfg), _sample_seed(args, fit_cfg))


def load_diagnostic_prompts(args, fit_cfg=None, logger=None) -> List[str]:
    prompt_text = _arg_value(args, "prompt_text", None)
    if prompt_text:
        return [str(prompt_text)]

    prompts_file = _prompts_file(args, fit_cfg)
    if prompts_file:
        return _load_file_prompts(prompts_file, args, fit_cfg)

    tasks = _tasks(args, fit_cfg)
    if not tasks:
        raise ValueError("provide --prompt_text, --prompts_file, --tasks, or diagnostics.tasks in config_json")

    prompts: List[str] = []
    for task in tasks:
        name = str(task).strip().lower()
        if name == "gsm8k":
            prompts.extend(_load_gsm8k_prompts(args, fit_cfg, logger=logger))
        elif name == "mmlu":
            prompts.extend(_load_mmlu_prompts(args, fit_cfg, logger=logger))
        else:
            raise ValueError(f"Unsupported diagnostics task: {task!r}")
    return _sample(prompts, _num_samples(args, fit_cfg), _sample_seed(args, fit_cfg))


def diagnostic_prompt_metadata(args, fit_cfg=None) -> Dict[str, Any]:
    prompt_text = _arg_value(args, "prompt_text", None)
    prompts_file = _prompts_file(args, fit_cfg)
    if prompt_text:
        source = "text"
    elif prompts_file:
        suffix = Path(prompts_file).suffix.lower()
        source = "jsonl" if suffix == ".jsonl" else "json"
    else:
        source = _arg_value(args, "prompt_source", None)
        if source is None:
            cfg = _diag_cfg(fit_cfg)
            source = getattr(cfg, "prompt_source", "task") if cfg is not None else "task"
    return {
        "prompt_source": str(source),
        "tasks": _tasks(args, fit_cfg),
        "sample_seed": _sample_seed(args, fit_cfg),
        "sample_split": _sample_split(args, fit_cfg),
        "prompt_template": _prompt_template_name(args, fit_cfg),
        "prompts_file": prompts_file,
    }
