# Stage 4F/4G vLLM validation

Stage 4 has CPU/mock coverage in the normal test suite, but CUDA acceptance must be run on a real NVIDIA host. A passing Mac or CPU test run does not prove that the exported model loads in vLLM, that NCCL transfer works, or that RL is faster end to end.

Stage 4F adds rollout provenance and policy-freshness gates. Every vLLM
rollout records a unique request ID plus prompt, sampling, and policy
fingerprints. The subprocess actor rejects a request if the expected policy
descriptor differs from the engine it loaded.

Stage 4G makes export_reload transactional. Raw checkpoints and HF exports are
first written under hidden temporary names, validated, given a committed
manifest, and atomically renamed. A state file points to the active committed
export. Matching policy fingerprints can reuse an already committed export.
Timing fields split checkpoint, conversion, validation, commit, cleanup, and
engine rebuild costs.

## Supported execution modes

- `rl.vllm_execution_mode="in_process"`: the training process owns the vLLM engine. This is required for native NCCL tensor inspection and transfer.
- `rl.vllm_execution_mode="subprocess"`: a persistent spawned rollout actor owns the vLLM engine. The first implementation is synchronous and supports `export_reload` and `weight_transfer_dryrun_static`. It is intended for a separate rollout GPU such as `cuda:1`.

With `vllm_enable_sleep_mode=true`, an engine whose policy is due for replacement is asked to sleep before checkpoint/export and rebuild. This lifecycle is code-tested only; GPU memory reclamation must be measured on the target vLLM build.

The subprocess actor deliberately rejects native NCCL, IPC, runtime dryrun inspection, and text-prompt fallback. Those combinations need an explicit cross-process CUDA/NCCL lifecycle design before they can be safe.

## Gate 0: environment record

Record the exact environment before running acceptance:

```bash
nvidia-smi
python -c "import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__, torch.version.cuda)"
```

Use an editable install so the exported remote code can import `fitmotn`:

```bash
pip install -e ".[dev]"
```

The subprocess control channel can be checked without CUDA:

```bash
python scripts/smoke_vllm_actor_control.py
```

This verifies spawn/request/shutdown only. It does not load a vLLM engine.

## Gate 1: export layout and HF roundtrip

```bash
python -m fitmotn.cli.export_hf \
  --checkpoint_dir /path/to/raw-checkpoint \
  --output_dir /path/to/stage4-export \
  --no-metadata_only \
  --base_model /path/to/base-model \
  --validate_layout
```

The command must complete without layout or HF roundtrip errors. Preserve its manifest in the test evidence.

## Gate 2: raw/HF/vLLM parity

```bash
python scripts/validate_stage4_cuda.py \
  --raw-checkpoint /path/to/raw-checkpoint \
  --export-dir /path/to/stage4-export \
  --output-json /path/to/evidence/stage4_parity.json \
  --device cuda:0 \
  --enforce-eager
```

Default acceptance:

- raw vs exported HF maximum final-position logit absolute error is at most `0.02` for BF16/FP16;
- raw and exported HF greedy output is exactly equal;
- exported HF vs vLLM greedy token match ratio is at least `0.99`;
- every test prompt has at least one unmasked token;
- include prompts of different lengths to exercise left-padding removal.

Tighten the logit threshold for FP32. Any greedy mismatch must be investigated rather than hidden by increasing the threshold.

## Gate 3: two-update RL smoke run

Edit `fitmotn_config.stage4_vllm_smoke.example.json` for the host. The checked-in example assumes training on `cuda:0` and the subprocess rollout actor on `cuda:1`.

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export CACHE_ROOT=/path/to/cache
export OUTPUT_ROOT=/path/to/outputs
python -m fitmotn.cli.train_rl --config_json fitmotn_config.stage4_vllm_smoke.example.json
```

Validate the resulting RL JSONL:

```bash
python scripts/validate_stage4_rl_run.py \
  "${OUTPUT_ROOT}/fitmotn_runs/stage4_vllm_smoke/rl_grpo/rl_train.jsonl" \
  --min-updates 2 \
  --output-json /path/to/evidence/stage4_rl_smoke.json
```

Acceptance:

- two optimizer updates finish;
- loss and policy loss are finite;
- `policy_lag_updates=0`;
- post-update vLLM policy version reaches update 2;
- no implicit HF fallback;
- the final checkpoint restores successfully.

## Gate 4: end-to-end benchmark

Run the same prompts, seeds, group size and generation settings with:

1. `rollout_backend="hf"`;
2. vLLM in-process plus `export_reload`;
3. vLLM subprocess actor plus `export_reload`.

Use at least 20 optimizer updates after warm-up and analyze the logs:

```bash
python scripts/analyze_rl_timing.py /path/to/rl_train.jsonl --last-n 20
```

Report total update wall time, rollout time, checkpoint/export time, engine rebuild time, generated tokens/s, peak CPU RAM, peak GPU memory, and bytes written. Generation tokens/s alone is not an acceptance metric.

## Gate 5: native transfer (experimental)

Start with `weight_transfer_dryrun_static`. Then use `weight_transfer_dryrun_runtime` in-process and inspect name, shape, dtype, and MoTN coverage.

For current `update_only` vLLM APIs, native transfer is enabled only by explicitly setting:

```json
{
  "rl": {
    "vllm_execution_mode": "in_process",
    "vllm_sync_strategy": "weight_transfer_nccl",
    "vllm_native_transfer_required_level": "update_only"
  }
}
```

This is not a claim of validation. Run it first with a tiny model and one update. Verify tensor fingerprints and a rollout after every sync. Keep the default `four_phase` gate for formal experiments until the target vLLM build and GPU topology have passed repeated transfer, timeout, and teardown tests.

## Stage 4F/4G strict acceptance addendum

The earlier two-update run is only a wiring check. Experimental use requires
the following strict acceptance.

First run all local checks:

~~~bash
pytest -q
python -m py_compile \
  rl/vllm_integrity.py rl/vllm_sync.py rl/vllm_rollout.py \
  scripts/validate_stage4_rl_run.py scripts/validate_stage4_sync_artifacts.py
~~~

For the first NVIDIA run, change rl.max_steps to 20 and keep:

~~~json
{
  "vllm_sync_every_updates": 1,
  "vllm_keep_sync_exports": 2,
  "vllm_export_validate_roundtrip": true,
  "vllm_export_roundtrip_validation_every": 1,
  "vllm_verify_engine_policy": true,
  "allow_stale_vllm_policy": false,
  "vllm_fallback_to_hf": false
}
~~~

The checked-in smoke config uses trainer cuda:0 and actor cuda:1. On a
one-GPU host, begin with in_process mode and conservative memory limits. A
two-GPU host is preferred.

Validate the 20-update run:

~~~bash
python scripts/validate_stage4_rl_run.py \
  /path/to/rl_grpo/rl_train.jsonl \
  --min-updates 20 \
  --output-json /path/to/evidence/stage4_rl_smoke.json

python scripts/validate_stage4_sync_artifacts.py \
  /path/to/rl_grpo/vllm_sync \
  --min-policy-version 20 \
  --output-json /path/to/evidence/stage4_sync_artifacts.json

python scripts/analyze_rl_timing.py \
  /path/to/rl_grpo/rl_train.jsonl --last-n 20 --json
~~~

Acceptance requires finite losses, zero policy lag, no fallback, unique
request IDs, a verified engine descriptor, an unbroken committed-sync to
next-rollout fingerprint chain, no incomplete transaction directory, and a
restorable final checkpoint.

The timing analyzer includes checkpoint, conversion, validation, commit,
cleanup, total sync, engine rebuild, generation time, and generation
throughput. If export plus rebuild is below roughly 10% of end-to-end update
time, native NCCL is not necessary for the current experiment. If it remains
above 10–15%, retain this evidence for a Stage 4H decision.

### Resume, reuse, and failure recovery

Run a second 20-update job with rl.resume_from pointing to the first job's
final checkpoint and a new output directory. The initial run-local version is
zero, but it must have a valid fingerprint and the actor must load that exact
descriptor. All later fingerprints must form an unbroken chain.

A forced retry of the same update with unchanged weights should report
vllm_export_reused=true and zero checkpoint/conversion time. A changed sampled
policy tensor must produce a different artifact.

For crash recovery, set vllm_export_temp_max_age_sec=0 in a disposable run,
terminate it during export, and restart. The next sync must remove hidden
temporary artifacts and must never load an export without a manifest whose
complete field is true.

### 200-update soak

After the strict 20-update run passes, execute at least 200 optimizer updates
with the intended model, prompt lengths, group size, and generation length.
Re-run both validators with minimum version/update 200. Acceptance additionally
requires no monotonic GPU/CPU memory or disk growth after retention pruning,
stable post-warmup latency and throughput, and successful final restore plus
the parity probe.

Only after the strict run passes may CPU roundtrip validation be reduced, for
example to every five updates. Layout validation, transaction manifests,
fingerprints, and engine provenance checks remain active on every sync.

## Evidence to archive

Archive the config, Git commit, model/export manifest, environment versions, parity JSON, RL validator JSON, timing summary, `nvidia-smi` output, and full logs. A Stage 4 GPU acceptance result without these artifacts is not reproducible.
