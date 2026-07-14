from __future__ import annotations

from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parent


def _ensure_source_package(name: str) -> None:
    existing = sys.modules.get(name)
    existing_paths = [Path(path).resolve() for path in getattr(existing, "__path__", [])] if existing else []
    if ROOT.resolve() in existing_paths:
        return
    for module_name in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
        sys.modules.pop(module_name, None)
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT)]
    package.__file__ = str(ROOT / "__init__.py")
    package.__package__ = name
    sys.modules[name] = package


# Tests must remain runnable from any checkout directory name without mutating
# the Python environment through an editable install.
_ensure_source_package("fitmotn")
_ensure_source_package("MOTE")
