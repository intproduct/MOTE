from .collate import pad_collate
from .specs import FrozenSFTTask, HFChatTask, HFTextTask, LocalTokenShardTask, TaskSpec


def __getattr__(name: str):
    if name == "build_stage_aware_train_dataset":
        from .builders import build_stage_aware_train_dataset

        return build_stage_aware_train_dataset
    raise AttributeError(name)

__all__ = [
    "build_stage_aware_train_dataset",
    "pad_collate",
    "TaskSpec",
    "HFTextTask",
    "HFChatTask",
    "LocalTokenShardTask",
    "FrozenSFTTask",
]
