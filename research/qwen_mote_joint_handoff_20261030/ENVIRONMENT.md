# Environment instructions for the machine agents

Use Linux if available. Keep each machine's environment independent and save
`python -m pip freeze`, Python version, driver/runtime version and GPU identity.
The local evidence environment was Python with **torch 2.6.0+cu118,
transformers 4.57.6, numpy 2.2.6, pytest 8.3.5**. It is a historical reference,
not a Blackwell or DTK installation recipe.

## Common Python layer

Start with Python 3.10 or 3.11 and `transformers==4.57.6` for Qwen3 integration.
Required packages include accelerate, datasets, tokenizers, safetensors, numpy,
tqdm, pytest, scipy and matplotlib. Resolve a working environment once, record
exact versions, and freeze it for each comparison. Inspect `pyproject.toml`.
Install the source package using `python -m pip install -e . --no-deps` to preserve
the chosen torch build. Do not install optional vLLM/RL packages for this study.

## Vendor runtime

- PRO 4000 Blackwell: use a PyTorch CUDA build with Blackwell support (PyTorch
  2.7/CUDA 12.8 or newer compatible releases). Check the installed driver and run
  actual forward/backward and custom-router probes; importing torch is insufficient.
- A800: use a supported NVIDIA CUDA/PyTorch stack. Match the PRO integration's
  Python-layer versions where practical and revalidate kernels on A800.
- V100: use a build that still includes Volta/sm70; verify the installed CUDA
  architecture list and actual execution. FP16 with gradient scaling is the
  low-precision path. BF16-native results belong to other hardware cohorts.
- BW1000: use the lab's supported DTK image and adapted torch/communication/
  attention packages. Preserve vendor packages rather than replacing them with
  generic CUDA wheels. Confirm device identity and the `.cuda()` compatibility
  layer. Audit any replacement attention kernel or fallback in a separate profile.

`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are conservative numerical-probe
settings. Formal data loading can use more CPU workers after deterministic sample
order and throughput checks. The old 35% CUDA allocator cap was specific to the
shared TITAN V host; new machines record their own reservation and free memory.

## Initial checks

From the repository root:

```bash
python scripts/verify_quantum_handoff.py
python -m pytest -q tests/test_operator_geometry.py tests/test_operator_geometry_round2.py tests/test_operator_geometry_round3.py tests/test_operator_geometry_functional.py tests/test_gate_router_arch.py tests/test_quantum_gate.py tests/test_qwen_handoff.py
python scripts/qwen_joint_preflight.py --machine pro4000 --output outputs/qwen_joint/pro4000/preflight_001.json --probe-gpu
```

Replace the machine name and output directory appropriately. `--probe-gpu`
performs a tiny matrix forward/backward only; Qwen/MoTE integration and full model
memory are separate acceptance checks. It never downloads models or starts training.
For CPU tests, set `CUDA_VISIBLE_DEVICES=""`. For GPU probes, first choose an
available device. On PowerShell set `$env:CUDA_VISIBLE_DEVICES='0'`; on Linux
prefix the command with `CUDA_VISIBLE_DEVICES=0`.

Weights and data must be acquired explicitly on each machine or transferred from
a verified shared cache. Resolve revision hashes; keep local credentials outside
the repository. Normal training requires no vLLM service or RL rollout engine.
