# Qwen → MoTE joint-training handoff

Publication branch: **mote-quantum**. Read this file after `git pull --ff-only origin mote-quantum`.

## Current state

The small-model research stages are complete. The new study starts from pretrained
Qwen **0.6B–4B** weights, replaces selected FFNs with MoTE, and jointly updates gates
and experts using next-token task loss. Qwen integration of the newly studied
CZ/product routers is the first assigned task; formal Qwen runs have not started.

The [Chinese plan](PLAN_zh.md), [Typst source](PLAN_zh.typ) and [PDF](PLAN_zh.pdf)
explain conclusions and machine allocation. Detailed executable-agent instructions
are in `agents/`; plan JSONs are **research specifications, not CLI train configs**.

## Start on your machine

1. Check `git status --short --branch`; preserve local changes. Switch to the
   existing `mote-quantum` branch (or track `origin/mote-quantum` on a new clone),
   then pull with `--ff-only`. Record `git rev-parse HEAD`.
2. Read the matching instruction file:
   - [PRO 4000: integration and 0.6B](agents/pro4000.md)
   - [2× V100: FP16 compatibility and independent replicas](agents/v100_pair.md)
   - [A800 80GB: 1.7B formal matrix](agents/a800.md)
   - [BW1000 64GB: DTK compatibility and matched pilot](agents/bw1000.md)
3. Install the appropriate vendor PyTorch **before** project dependencies.
   Retain BW1000's vendor DTK build. Use `python -m pip install -e . --no-deps`
   after preparing the dependency versions in `ENVIRONMENT.md`.
4. Run the standard-library-only evidence check:
   `python scripts/verify_quantum_handoff.py`.
5. Record the environment with a fresh output path, for example:
   `python scripts/qwen_joint_preflight.py --machine pro4000 --output outputs/qwen_joint/pro4000/preflight_001.json`.
   Add `--probe-gpu` to run a tiny forward/backward numerical probe. The probe
   uses only the selected process-visible device; set `CUDA_VISIBLE_DEVICES` first.
6. Continue with the machine's agent instructions. All formal launch commands
   must be generated from the reviewed integration release. This handoff does
   not silently substitute an old training command for the four new routers.

## Evidence and reading order

- [Latest functional conclusions](../operator_family_quantum_gate_20261030/FUNCTIONAL_GEOMETRY_REVIEW_20261030_zh.pdf)
  and [numerical supplement](../operator_family_quantum_gate_20261030/FUNCTIONAL_GEOMETRY_SUPPLEMENT_20261030_zh.pdf).
- [Capacity / corrected replay](../operator_family_quantum_gate_20261030/CAPACITY_GEOMETRY_REVIEW_20261030_zh.pdf).
- [Utility / geometry stage](../operator_family_quantum_gate_20261030/STAGE_REVIEW_20261030_zh.pdf).
- [Novelty review](../operator_family_quantum_gate_20261030/NOVELTY_REVIEW_zh.pdf).
- `../experiment_evidence_20261030/MANIFEST.json`: original CSV/JSON/JSONL copies
  with SHA-256 and original paths, covering classification, language modelling,
  routing stability and all three operator-geometry rounds.
- `../experiment_evidence_20261030/EXTERNAL_ARTIFACTS.json`: retained local weights
  and arrays that are **not downloaded by git pull**. Fresh Qwen experiments use
  newly acquired Qwen weights and a separately frozen data release; historical
  checkpoint replay additionally requires transferring the listed old assets.

Historical state files, absolute Windows paths and obsolete deadlines in the
evidence are provenance. This README and `COMMON_PROTOCOL.json` define the next
stage. Report PDFs and figures are committed for offline reading.

Historical report PDFs/Typst/Markdown retain their original audited bytes. Some
Markdown links embed the original Windows workspace; use the PDF links above
when reading on another machine. A byte-preserved latest functional-stage source
snapshot is in `../operator_geometry_runtime_snapshot_20261030/`; its manifest
records the original paths and hashes. Git may normalize normal source files on
Linux, so exact historical raw-byte source replay should use that snapshot in a
separate reconstruction checkout. Never overwrite an active experiment checkout.

## Coordination and publication

PRO 4000 is the primary owner of new shared Qwen integration code. Other machines
may develop isolated compatibility fixes and inspect their own hardware immediately.
Use named `codex/qwen-joint-<machine>` branches for concurrent edits; submit them
for review/integration into `mote-quantum`. Record each run's exact commit and dirty
diff. Do not concurrently push unreviewed divergent implementations to the shared
branch or force-push it. GPU execution requires the same frozen code/data/config
release for methods in a comparison; hardware-only changes go in separate profiles.

Each machine writes unique runs under `outputs/qwen_joint/<machine>/<release>/<run>`
and exports compact results under `research/qwen_joint_results/<machine>/<release>`.
Preserve failed runs; create new run IDs for retries. Do not overwrite old outputs,
reuse an active lock, terminate another user's process, or auto-launch the historical
helper scripts. Ask only for missing credentials/data access or spending decisions;
carry out the assigned local implementation and validation work autonomously.
