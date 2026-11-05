"""Training entry points, loaded only when called or explicitly imported."""
from importlib import import_module

__all__ = ["run_fitmotn_training", "run_fitmotn_rl_training", "run_fitmotn_training_and_optional_rl"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = ".controller" if name == "run_fitmotn_training" else ".rl_controller"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
