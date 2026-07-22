# GSM8K multi-rollout spectrum audit

`fitmotn.cli.build_boundary_gsm8k` is an evaluation-only command. It does not
create an optimizer, run backward, start GRPO/MGPO training, or update the
source checkpoint. The default `--rollout_backend hf` retains the original
boundary-builder path; `--rollout_backend vllm` uses the project's rollout
actors and performs one isolated static export/reload bootstrap.

## Reading the metrics

Sampled Pass@1 is the average reward across stochastic samples. It is not the
same measurement as greedy GSM8K accuracy: temperature/top-p sampling can
produce both better and worse trajectories for a prompt. Pass@16 estimates the
probability that at least one of 16 samples is correct. A wide gap between
Pass@1 and Pass@16 indicates latent correct trajectories that are not yet
reliably selected.

- `all_wrong`: no sampled trajectory was correct; this prompt supplies no
  positive member to a binary-reward GRPO group.
- `all_correct`: every trajectory was correct; the group has zero binary
  reward advantage despite high accuracy.
- `mixed` / `effective_rl_prompt`: both reward classes occur, so binary group
  advantages are non-zero.
- `boundary`: prompt accuracy lies within `--audit_boundary_min` and
  `--audit_boundary_max`. This classification is independent of
  `--min_correct_rate` / `--max_correct_rate`, which only control which prompt
  records are written to `--output_jsonl`.

Pass@k uses `1 - C(n-c,k)/C(n,k)`. Reward parsing and numerical equivalence are
the same `gsm8k_reward` used online. Match types are exclusive: strict `####`,
explicit final answer, answer marker, fallback-last-number-only, unparseable,
or incorrect/no-match. Positive fallback-only traces are marked for manual
inspection rather than silently promoted to high-quality distillation data.

`--all_rollouts_jsonl` contains one full response per line and can be large.
GSM8K train with 16 samples writes 119,568 lines; actual space depends on
response length and JSON metadata, so check a 32-prompt smoke run before the
full audit. Files are written incrementally and the summary is committed
atomically at completion.

## 32-prompt vLLM smoke test

```bash
cd /work/home/sugang2025/qxfang/MOTE-g
CONFIG=/work/home/sugang2025/qxfang/MOTE-g/online_mgpo_g16_s1000.json
MODEL=/work/home/sugang2025/qxfang/MOTE-g/fitmotn_runs/fitmotn_k8_nodecay_7e_tp8_tc/final_model
OUTPUT_DIR=/work/home/sugang2025/qxfang/MOTE-g/rollout_audit/sft_smoke
mkdir -p "$OUTPUT_DIR" ./logs
nohup python -m fitmotn.cli.build_boundary_gsm8k \
  --config_json "$CONFIG" --resume_from "$MODEL" \
  --rollout_backend vllm --num_rollouts 16 --prompt_batch_size 1 \
  --max_prompts 32 --min_correct_rate 0.0 --max_correct_rate 1.0 \
  --audit_boundary_min 0.25 --audit_boundary_max 0.75 \
  --pass_k 1 4 8 16 --max_new_tokens 512 --temperature 1.1 --top_p 0.95 \
  --seed 1234 --output_jsonl "$OUTPUT_DIR/all_prompts.jsonl" \
  --verified_traces_jsonl "$OUTPUT_DIR/verified_traces.jsonl" \
  --all_rollouts_jsonl "$OUTPUT_DIR/all_rollouts.jsonl" \
  --summary_json "$OUTPUT_DIR/summary.json" --progress_every 8 \
  > ./logs/nohup_sft_passk_smoke.log 2>&1 &
```

Do not prepend a global `CUDA_VISIBLE_DEVICES` override. In subprocess mode,
each configured rollout actor receives its own private device list (for
example physical GPUs 1 and 2) and sees those assigned devices as local
indices. Tensor parallelism, dtype, memory utilization, maximum sequence
length, and actor placement remain controlled by the RL config.

## Full train audits

Use identical sampling arguments for the two checkpoints:

```bash
cd /work/home/sugang2025/qxfang/MOTE-g
CONFIG=/work/home/sugang2025/qxfang/MOTE-g/online_mgpo_g16_s1000.json
SFT=/work/home/sugang2025/qxfang/MOTE-g/fitmotn_runs/fitmotn_k8_nodecay_7e_tp8_tc/final_model
RL200=/work/home/sugang2025/qxfang/MOTE-g/fitmotn_runs/fitmotn_k8_nodecay_7e_tp8_tc/rl_grpo/checkpoint-200
mkdir -p rollout_audit/sft_train rollout_audit/rl200_train logs

# Run this SFT audit first and wait for it to finish.
OUT=rollout_audit/sft_train
nohup python -m fitmotn.cli.build_boundary_gsm8k \
  --config_json "$CONFIG" --resume_from "$SFT" --rollout_backend vllm \
  --num_rollouts 16 --prompt_batch_size 1 \
  --min_correct_rate 0.0 --max_correct_rate 1.0 \
  --audit_boundary_min 0.25 --audit_boundary_max 0.75 \
  --pass_k 1 4 8 16 --max_new_tokens 512 --temperature 1.1 --top_p 0.95 \
  --seed 1234 --output_jsonl "$OUT/all_prompts.jsonl" \
  --verified_traces_jsonl "$OUT/verified_traces.jsonl" \
  --all_rollouts_jsonl "$OUT/all_rollouts.jsonl" \
  --summary_json "$OUT/summary.json" --progress_every 25 \
  > logs/nohup_sft_train.log 2>&1 &

# After the SFT process has exited, run checkpoint-200 with identical settings.
OUT=rollout_audit/rl200_train
nohup python -m fitmotn.cli.build_boundary_gsm8k \
  --config_json "$CONFIG" --resume_from "$RL200" --rollout_backend vllm \
  --num_rollouts 16 --prompt_batch_size 1 \
  --min_correct_rate 0.0 --max_correct_rate 1.0 \
  --audit_boundary_min 0.25 --audit_boundary_max 0.75 \
  --pass_k 1 4 8 16 --max_new_tokens 512 --temperature 1.1 --top_p 0.95 \
  --seed 1234 --output_jsonl "$OUT/all_prompts.jsonl" \
  --verified_traces_jsonl "$OUT/verified_traces.jsonl" \
  --all_rollouts_jsonl "$OUT/all_rollouts.jsonl" \
  --summary_json "$OUT/summary.json" --progress_every 25 \
  > logs/nohup_rl200_train.log 2>&1 &
```

The boundary CLI intentionally preserves the online GSM8K train prompt pool;
an independent test-split switch is not added in this version, avoiding a
second dataset-loading path that could drift from training.

## Progress, summary, and comparison

```bash
tail -f logs/nohup_sft_train.log
python -m json.tool rollout_audit/sft_train/summary.json | less
python - <<'PY'
import json
from pathlib import Path
paths = [Path("rollout_audit/sft_train/summary.json"), Path("rollout_audit/rl200_train/summary.json")]
keys = ["sampled_pass_at_1", "pass_at_4", "pass_at_8", "pass_at_16", "mixed_ratio", "boundary_ratio", "all_wrong_ratio", "all_correct_ratio"]
rows = [json.loads(path.read_text())["spectrum"] for path in paths]
for key in keys:
    print(f"{key:28s} SFT={rows[0][key]:.6f} RL200={rows[1][key]:.6f} delta={rows[1][key]-rows[0][key]:+.6f}")
PY
```
