# Qwen QuantumGate runtime source archive

This directory preserves the operational code that accompanied training commit
`860426ce3df30f5c2a74a030a43f22e3a76247c6` and the monitoring fixes through
2026-11-01. The snapshot was collected on 2026-11-05 from
`E:/AI/MOTE-quantum`. `SNAPSHOT.json` records every payload's relative path,
size, SHA-256, and inclusion reason.

## Contents

- Frozen v12 training supervision, v14 diagnostic coordination and numerical workers, together with their referenced frozen dependency bundles and original manifests.
- The canonical training entry, v24 health/resource observers, process-exit observers, regression tests, and the v23 boundary fixture.
- The local read-only dashboard source and its historical acceptance records.
- The formal matrix configurations, integration release, environment requirements, and acceptance binding.

Model weights, datasets, checkpoints, live queue/process state, generated run
outputs, and environment binaries remain external. Original absolute paths and
source hashes are intentionally preserved. The historical manifests describe
their original release stages; they are not claims that the full experiment
has finished. At collection, the recorded phase was
`formal_matrix_failed_preserved`, awaiting sufficient disk capacity and a
separately validated continuation.

## Integrity verification

From the repository root, run:

```powershell
python research/qwen_quantum_runtime_snapshot_20261105/verify_snapshot.py
```

On the original host, `--source-root E:/AI/MOTE-quantum` additionally verifies
the archive against the original files. After staging, `--git-index` verifies
that Git stores the exact payload bytes. The repository attributes disable
line-ending conversion for these payloads.

## CPU monitoring regression tests

Use the experiment's recorded Python environment, with CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pytest -q -p no:cacheprovider research/qwen_quantum_runtime_snapshot_20261105/runtime/test_formal_health_boundaries_v18.py research/qwen_quantum_runtime_snapshot_20261105/runtime/test_formal_health_freshness_v22.py research/qwen_quantum_runtime_snapshot_20261105/runtime/test_formal_health_freshness_v23.py research/qwen_quantum_runtime_snapshot_20261105/runtime/test_formal_health_processes_v24.py
```

These tests use saved fixtures and mocks. Archived execution helpers retain
their original machine bindings; validation does not require starting them.
The active experiment checkout stays at its frozen commit. Adding this archive
to that checkout would change its tracked-source inventory, so the publication
is prepared in a separate worktree.
