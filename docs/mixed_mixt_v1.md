# Mixed MiXT v1

## Purpose

`mixed_mixt` is the four-MiXT backend for real dimensions that decompose into
two exact powers of `d`. It keeps one routing decision per projection and
replaces all four dense matrix quadrants with expert TensorBlocks. It does not
change the existing `sparse_mixt` backend.

For Qwen3.5-4B:

```text
hidden:       2560 = 2048 + 512  = 2^11 + 2^9
intermediate: 9216 = 8192 + 1024 = 2^13 + 2^10
```

## Algebra

Split `x = concat(x0, x1)` and compute one shared route:

```text
probs, mask, aux = Gate(x_real)
y0 = M00(x0; probs, mask) + M01(x1; probs, mask)
y1 = M10(x0; probs, mask) + M11(x1; probs, mask)
y  = concat(y0, y1)
```

Every `Mij` is a `GatedADTNLayer` TensorBlock group with its internal gate
removed. `MixedMiXTCore.gate` is the only router and the four groups receive
the same `probs/mask/aux`. Expert index `e` therefore identifies one complete
four-quadrant expert rather than four unrelated routing choices.

For each quadrant:

```text
k_out = k_in + q_out - q_in
```

The configured value is quadrant-specific `k_in`; `k_out` is derived and
validated. Full-site bonds (`k_in=q_in`, `k_out=q_out`) with `E=1` reproduce an
arbitrary dense quadrant exactly apart from floating-point accumulation order.
Local bonds are structured approximations and require calibration/training.

## Qwen3.5-4B bonds

Gate and Up:

| quadrant | dimensions | q | configured bond |
|---|---:|---:|---:|
| m00 | 2048 -> 8192 | 11 -> 13 | 8 -> 10 |
| m01 | 512 -> 8192 | 9 -> 13 | 5 -> 9 |
| m10 | 2048 -> 1024 | 11 -> 10 | 6 -> 5 |
| m11 | 512 -> 1024 | 9 -> 10 | 5 -> 6 |

Down:

| quadrant | dimensions | q | configured bond |
|---|---:|---:|---:|
| m00 | 8192 -> 2048 | 13 -> 11 | 10 -> 8 |
| m01 | 1024 -> 2048 | 10 -> 11 | 6 -> 7 |
| m10 | 8192 -> 512 | 13 -> 9 | 8 -> 4 |
| m11 | 1024 -> 512 | 10 -> 9 | 6 -> 5 |

## Configuration

Use `fitmotn_config.mixed_mixt_qwen35_4b.example.json`. Important fields:

```json
{
  "patch_backend": "mixed_mixt",
  "mixed_mixt": {
    "backend_version": 1,
    "hidden_main": 2048,
    "intermediate_main": 8192,
    "router_input_policy": "full_real",
    "dense_init": "none",
    "gate_bonds": {"m00": 8, "m01": 5, "m10": 6, "m11": 5},
    "up_bonds": {"m00": 8, "m01": 5, "m10": 6, "m11": 5},
    "down_bonds": {"m00": 10, "m01": 6, "m10": 8, "m11": 6}
  }
}
```

`dense_init=full_site_exact` is a correctness-test mode. It requires `E=1`
and full-site bonds in all quadrants. It has dense parameter count and is not a
compressed training configuration.

## Tests

CPU algebra, routing, gradients, patching, checkpoint, calibration, and vLLM:

```powershell
python -m pytest -q -s tests/test_mixed_mixt.py
```

Local Qwen3.5-4B layer-0 GPU validation (also runs without pytest):

```powershell
C:\Apps\Miniforge3\envs\sc\python.exe -u tests\test_mixed_mixt_qwen35_local.py
```

Override the local model location with `FITMOTN_QWEN35_4B_PATH`.

Observed on RTX PRO 4000 Blackwell with the local layer-0 BF16 weights:

```text
full-site relative L2: 5.5031e-3
full-site max abs:     3.90625e-3
structured E=8 parameters: 7,690,240
structured loss: 1.75050 -> 1.68784 (two optimizer steps)
structured peak PyTorch memory: 86.6 MiB
```
