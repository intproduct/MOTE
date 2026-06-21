from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass
class TaskSpec:
    name: str
    path: str
    split: str = "train"
    weight: float = 1.0
    max_samples: Optional[int] = None
    kind: str = "load_from_disk"
    hf_name: str | None = None
    hf_config: str | None = None
    group: str = "task"
    bucket: str = "task"
    source_family: str = "generic"
    supports_skip: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def map_example(self, ex: Dict[str, Any]) -> Tuple[str, str, str] | None:
        raise NotImplementedError

    def describe_data_policy(self) -> Mapping[str, Any]:
        return {}


@dataclass
class HFTextTask(TaskSpec):
    text_field: str = "text"
    max_samples: Optional[int] = None
    kind: str = "hf_text"
    group: str = "pretrain"
    source_family: str = "text"

    def map_example(self, ex: Dict[str, Any]) -> Tuple[str, str, str]:
        txt = ex.get(self.text_field)
        if txt is None:
            for cand in ["text", "content", "body", "document", "code", "completion"]:
                if cand in ex and ex[cand] is not None:
                    txt = ex[cand]
                    break
        if txt is None:
            raise KeyError(f"[{self.name}] cannot find text field in example keys={list(ex.keys())}")
        return "", str(txt), "causal_lm"


@dataclass
class HFChatTask(TaskSpec):
    messages_field: Optional[str] = None
    prompt_field: Optional[str] = None
    response_field: Optional[str] = None
    system_field: Optional[str] = None
    max_samples: Optional[int] = None
    kind: str = "hf_chat"
    group: str = "task"
    source_family: str = "chat"

    def map_example(self, ex: Dict[str, Any]) -> Tuple[str, str, str]:
        return "", "", "chat_sft"


@dataclass
class LocalTokenShardTask(TaskSpec):
    max_samples: Optional[int] = None
    kind: str = "local_token_shards"
    group: str = "pretrain"
    source_family: str = "token_shards"

    def map_example(self, ex: Dict[str, Any]) -> Tuple[str, str, str]:
        return "", "", "token_ids"
