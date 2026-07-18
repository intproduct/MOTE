# Stage 5E：patch-only 原生权重同步

Stage 5E 保留 Stage 5D 的完整 bootstrap、常驻 vLLM Engine、NCCL update-only、
policy barrier 和失败即丢弃语义，但在 optimizer update 后只传输当前
`patch_only` optimizer 实际更新的参数。传输内容是 patch 参数的完整绝对值，
不是数值 delta。

## 安全边界

- 默认 `vllm_weight_transfer_scope=full_policy`，不改变 Stage 5D 行为；
- Stage 5E 必须显式设置 `trainable_patch`；
- bootstrap 仍为完整 Stage 4C HF export；
- transfer allowlist、`requires_grad` 参数和 optimizer 参数必须完全一致；
- allowlist 必须全部属于模块感知的 MoTN patch 参数；
- 每次更新校验 transfer-plan fingerprint、patch coverage 和冻结参数 fingerprint；
- Actor 验证接收到的 scope、plan fingerprint 和 tensor count；
- 禁止 HF fallback、export_reload fallback 和 stale policy；
- 当前三卡生产路径仍要求两个 TP1 subprocess Actor；
- 本阶段不引入双 Actor 并发传输、TP2+、Trainer DDP 或数值 delta。

## 配置

在已经通过 Stage 5D 的配置上增加：

```json
{
  "rl": {
    "trainable_mode": "patch_only",
    "rollout_backend": "vllm",
    "vllm_execution_mode": "subprocess",
    "vllm_sync_strategy": "weight_transfer_nccl",
    "vllm_native_transfer_required_level": "update_only",
    "vllm_weight_transfer_scope": "trainable_patch",
    "vllm_sync_every_updates": 1,
    "vllm_weight_transfer_validate_coverage": true,
    "vllm_weight_transfer_fail_on_partial": true,
    "vllm_weight_transfer_validate_after_sync": true,
    "vllm_weight_transfer_fallback_to_export_reload": false,
    "vllm_fallback_to_hf": false,
    "allow_stale_vllm_policy": false,
    "log_every": 1,
    "shuffle_train_data": true,
    "sampler_seed": 1234
  }
}
```

不要复用已有非空 `output.run_name`。1、2、5、20 update 分别使用独立 run name。

## 本地/CPU 门槛

```bash
python -m pytest -q \
  tests/test_vllm_actor.py \
  tests/test_vllm_weight_transfer.py \
  tests/test_vllm_rollout_backend.py \
  tests/test_stage5e_patch_sync_validation.py

python scripts/validate_stage5e_patch_sync.py \
  --config /path/to/stage5e.json \
  --min-updates 1
```

第二条命令只做配置 preflight；没有提供 `--rl-train-jsonl` 时不会声称 GPU 已通过。

## 三卡手动验收

### 0. 清理和确认资源

```bash
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
pgrep -u "$USER" -af 'train_rl|vllm_actor|EngineCore|multiprocessing.spawn'
```

确认 GPU1/2 没有旧 Actor/EngineCore。只终止属于自己的残留进程。

### 1. 一个 update：验证 partial-update API

将配置设为 `max_steps=1` 和新 run name，然后：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m fitmotn.cli.train_rl \
  --config_json /path/to/stage5e_u1.json 2>&1 | tee stage5e_u1.log
```

训练完成后：

```bash
python scripts/validate_stage5e_patch_sync.py \
  --config /path/to/stage5e_u1.json \
  --rl-train-jsonl /path/to/run/rl_grpo/rl_train.jsonl \
  --export-root /path/to/run/rl_grpo/vllm_sync \
  --min-updates 1 \
  --max-payload-ratio 0.10 \
  --output-json stage5e_u1_validation.json
```

这一步是 vLLM 0.19 partial `update_weights` 的真实 GPU gate。必须看到：

- `vllm_weight_transfer_scope=trainable_patch`；
- Actor contract 的 tensor count 和 plan fingerprint 与 Trainer 一致；
- payload ratio 明显小于 1；
- Actor commit 成功；
- greedy rollout validation 成功；
- Engine rebuild 为 0。

### 2. 两个 update：验证常驻 Engine 和计划稳定性

用全新 run name 和 `max_steps=2` 重复训练，然后：

```bash
python scripts/validate_stage5e_patch_sync.py \
  --config /path/to/stage5e_u2.json \
  --rl-train-jsonl /path/to/run/rl_grpo/rl_train.jsonl \
  --export-root /path/to/run/rl_grpo/vllm_sync \
  --min-updates 2 \
  --max-payload-ratio 0.10 \
  --output-json stage5e_u2_validation.json
```

额外确认两个 Actor PID 跨 update 不变，transfer-plan fingerprint 不变，policy lag 为 0。

### 3. 五个 update：验证自动退出

运行 `max_steps=5`，验证器使用 `--min-updates 5`。退出后执行：

```bash
nvidia-smi
pgrep -u "$USER" -af 'train_rl|vllm_actor|EngineCore|multiprocessing.spawn'
```

不得残留本次 Actor 或 EngineCore。

### 4. 二十个 update：soak 与性能证据

运行 `max_steps=20`，然后：

```bash
python scripts/validate_stage5e_patch_sync.py \
  --config /path/to/stage5e_u20.json \
  --rl-train-jsonl /path/to/run/rl_grpo/rl_train.jsonl \
  --export-root /path/to/run/rl_grpo/vllm_sync \
  --min-updates 20 \
  --max-payload-ratio 0.10 \
  --max-sync-p95-sec 3.0 \
  --output-json stage5e_u20_validation.json
```

`3.0s` 是性能目标，不是首轮正确性硬门槛。如果只有该项失败，应保留结果并分析
packing、NCCL send、Actor validation 和 commit 分项，而不是判定 partial sync 不正确。

最终 checkpoint：

```bash
python -m fitmotn.cli.validate_checkpoint \
  /path/to/run/rl_grpo/final_model \
  --require-exact rl
```

## 正式通过条件

1. 只有 update 0 产生完整 bootstrap export；
2. update 1..N 全部使用 `trainable_patch` NCCL；
3. patch/optimizer/requires-grad coverage 均为 100%；
4. payload ratio 不超过 0.10，实际预期约为现有参数比例附近；
5. transfer-plan fingerprint 全程不变；
6. frozen parameter drift 为 false；
7. 双 Actor contract、checksum/生成验证和 commit barrier 全部通过；
8. policy lag 为 0，无 fallback、无 Engine rebuild；
9. Actor PID 跨 update 不变，退出后无残留；
10. checkpoint exact resume 可加载并继续至少两个 update。

GPU 验收前，代码只能标记为“CPU/mock 实现完成，GPU gate 待通过”，不能把本地测试
写成 vLLM partial update 的实机证据。
