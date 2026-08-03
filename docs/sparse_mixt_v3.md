# Sparse MiXT v3 backend

`patch_backend="sparse_mixt"` preserves the existing MoTN router, expert,
TopK/soft routing, global expert, usage tracking, and `q/k_in/k_out` behavior.
It replaces only projection padding/cropping with a real-dimension block map.

For `in=n0+n1` and `out=m0+m1`, each projection computes:

```text
y0 = MiXT00(x0) + LR01(x1)
y1 = LR10(x0)   + LR11(x1)
y  = concat(y0, y1)
```

Empty tails are omitted. For example, `4096 -> 12288` uses a `4096 -> 8192`
MiXT main block and only `LR10: 4096 -> 4096`.

## Qwen3.5-4B layout

```text
hidden:       2560 = 2048 + 512
intermediate: 9216 = 8192 + 1024
gate/up main: q_in=11, q_out=13, k_in=8,  k_out=10
down main:    q_in=13, q_out=11, k_in=10, k_out=8
```

No projection forward creates a 4096/16384 padded activation or crops a padded
output. `in_pad` and `out_pad` intentionally report the real dimensions for
generic observability compatibility.

## Initialization

Boundary modes:

- `zero`: zero-output low-rank residuals; fastest construction.
- `full_svd` (or legacy alias `svd`): exact SVD before truncation.
- `randomized_svd`: randomized low-rank initialization with configurable
  oversampling and power iterations.

The existing approximation calibration then jointly trains selected MiXT
blocks and all boundary factors against the complete dense projection.

## Auditable local tests

Run the staged neural-equivalence suite with printed calculation traces:

```bash
python -m pytest tests/test_sparse_mixt.py -q -s
```

The trace prints each real/main/tail shape, `d/q/k`, expert/router mode, Qwen
activation order, and an explicit `padding=False crop=False` assertion marker.

The full related regression suite covers patching, checkpoint restore, HF
remote-code roundtrip, export manifests, and vLLM weight mapping.
