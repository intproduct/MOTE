"""Create an isolated vLLM copy with a partial-NVML availability fix.

The existing environment is read-only. Hardware queries use the upstream
NonNvmlCudaPlatform when WSL cannot query the host's TCC GPUs.
"""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

source = Path(importlib.util.find_spec("vllm").origin).parent
target = Path(sys.argv[1]) / "vllm"
if target.exists():
    raise FileExistsError(target)
shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
path = target / "platforms" / "cuda.py"
original = path.read_text()
start = original.index("nvml_available = False\ntry:", original.index("# Autodetect either"))
end = original.index("\nCudaPlatform =", start)
replacement = '''nvml_available = False
nvml_started = False
try:
    pynvml.nvmlInit()
    nvml_started = True
    for physical_index in range(pynvml.nvmlDeviceGetCount()):
        pynvml.nvmlDeviceGetHandleByIndex(physical_index)
    nvml_available = True
except Exception:
    nvml_available = False
finally:
    if nvml_started:
        pynvml.nvmlShutdown()
'''
path.write_text(original[:start] + replacement + original[end:])
manifest = dict(source=str(source), isolated_copy=str(target), source_cuda_sha256=hashlib.sha256(original.encode()).hexdigest(),
                patched_cuda_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                change="Use existing non-NVML CUDA backend when host NVML includes inaccessible TCC devices")
assert hashlib.sha256((source / "platforms" / "cuda.py").read_bytes()).hexdigest() == manifest["source_cuda_sha256"]
Path(__file__).with_name("wsl_vllm_environment_patch.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
