# Operator geometry experiments: historical replay instructions

These experiments and their configurations are frozen evidence from the small-model
research stages. The completed stages include utility prediction, teacher-score
calibration, functional geometry, corrected sparse replay and timing.

Start with the current [Qwen handoff](../../research/qwen_mote_joint_handoff_20261030/README.md)
for the new 0.6B–4B pretrained-model study. The configurations in this directory
depend on local `outputs/` checkpoints and caches. A fresh clone contains compact
results in `research/experiment_evidence_20261030/`, not those training assets.

## Safe entry points

Run the four `tests/test_operator_geometry*.py` files for CPU mathematical and
workflow checks. Historical training entry points require their exact original
assets and a new output directory. Read the configuration before invoking them.

- `runner.py`, `pilot.py`: original diagnostic and static-assignment pilot.
- `round2_*`: expanded utility, ridge, geometry, positive controls, risk and audits.
- `round3_capacity.py`: interleaved RY/CZ/RY and matched product calibration.
- `round3_replay_v2.py`: corrected sparse replay; `round3_replay.py` retains the
  superseded selector implementation as failure evidence.
- `round3_functional*.py`: fixed-expert coefficient/output/Fisher and diagonal
  objective controls, independent reviews and serial timing.
- `round3_metric.py`, `round3_sparse_tangent.py`: state and sparse-readout geometry.

`ReencodingPredictor.forward()` emits a **cost** (smaller is better). Qwen's
`RouterLogits` contract uses larger-is-better logits. A new integration must
explicitly convert this contract and test expert indices and mixture weights.
The new Qwen adapter is an implementation task in the handoff, not an existing
`gate_arch` value in this historical directory.
