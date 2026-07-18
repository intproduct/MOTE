# Stage 5D：常驻 vLLM Actor 全量权重原生同步

Stage 5D 将三卡 online MGPO 的逐更新 `export_reload` 替换为 vLLM 0.19
`update_only` NCCL 权重传输。训练仍然是 `patch_only`；“全量”仅表示每次把完整
checkpoint-format policy 发送给 vLLM，并不解冻 Dense 参数。

## 支持边界

- GPU0：单进程 HF trainer；
- GPU1、GPU2：两个相互隔离的常驻 vLLM subprocess Actor；
- 每个 Actor 为 TP1；
- vLLM 必须暴露 `init_weight_transfer_engine`、`update_weights` 和 NCCL trainer API；
- 首次启动仍执行一次 Stage 4C HF export，之后不再逐更新写出模型；
- Actor 更新当前按顺序进行，双通道并发发送留到 GPU soak 通过后；
- IPC、Actor TP2+、trainer DDP/FSDP、异步 stale-policy RL 和 patch-only 传输不在本阶段范围。

## 一致性协议

Actor 状态为：

```text
LOADING → READY → UPDATING → VALIDATING → UPDATED_PENDING_COMMIT → READY
                                  └──────── failure ─────────────→ FAILED
```

`UPDATING`、`VALIDATING`、`UPDATED_PENDING_COMMIT` 和 `FAILED` 状态都拒绝
rollout。trainer 只有在所有 Actor 完成接收与验证后才发送 commit。vLLM 0.19 的
update-only API 没有 rollback，因此这不是数据库意义的原子提交：任一 Actor
失败时整个 Actor 集合都会被丢弃，禁止使用半更新 Engine。若显式开启
`vllm_weight_transfer_fallback_to_export_reload`，将从新 export 重建所有 Actor；验收
阶段必须关闭该回退，防止假阳性。

每次原生更新记录：

- policy version 和 fingerprint；
- 完整 tensor 数量、字节数和 MoTN 覆盖率；
- 每个 Actor 的 capability、NCCL port、send/receive 时间和 Engine PID；
- all-actor commit barrier；
- prefix-cache reset 和固定 token prompt 的一 token 生成验证；
- 可用时的 Actor runtime selected-tensor checksum。

原生路径强制关闭 vLLM prefix caching，并在每次更新后调用可用的
`reset_prefix_cache`，避免旧 policy 的可复用前缀状态进入新 rollout。

## 三卡配置

复制：

```bash
cp fitmotn_config.stage5d_3xa800_native_sync.example.json /path/to/native_sync.json
```

至少修改模型、数据与输出路径。关键配置如下：

```json
{
  "rl": {
    "rollout_backend": "vllm",
    "vllm_execution_mode": "subprocess",
    "vllm_sync_strategy": "weight_transfer_nccl",
    "vllm_native_transfer_required_level": "update_only",
    "vllm_weight_transfer_backend": "nccl",
    "vllm_weight_transfer_master_port": 29580,
    "vllm_weight_transfer_fallback_to_export_reload": false,
    "vllm_fallback_to_hf": false,
    "allow_stale_vllm_policy": false,
    "vllm_rollout_actors": [
      {"name": "rollout_0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
      {"name": "rollout_1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1}
    ]
  }
}
```

base port `29580` 分配给第一个 Actor，后续 Actor 使用连续端口。确保这些端口未被
其他任务占用。也可以设为 `0` 自动选择，但固定端口更方便复现实验和排障。

## 验收顺序

先检查配置和本地测试：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m pytest -q \
  tests/test_vllm_actor.py \
  tests/test_vllm_weight_transfer.py \
  tests/test_vllm_rollout_backend.py

CUDA_VISIBLE_DEVICES=0,1,2 python scripts/validate_stage5d_native_sync.py \
  --config /path/to/native_sync.json
```

依次执行 1、2、20 update，使用不同 `output.run_name`，不要覆盖证据。推荐用一键
验收器（`--max-steps` 分别设为 1、2、20）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/run_stage5d_a800_acceptance.py \
  --config /path/to/native_sync.json \
  --max-steps 20 \
  --min-updates 20 \
  --run-name stage5d_native_u20
```

训练结束后：

```bash
python scripts/validate_stage5d_native_sync.py \
  --config /path/to/native_sync.json \
  --rl-train-jsonl /path/to/run/rl_grpo/rl_train.jsonl \
  --export-root /path/to/run/rl_grpo/vllm_sync \
  --min-updates 20 \
  --max-sync-p95-sec 60 \
  --output-json /path/to/evidence/stage5d_native_sync.json
```

通过条件：

1. run start 只有一次 export bootstrap；
2. update 1..N 全部为 `nccl_update_only_subprocess`；
3. policy lag 为 0，未发生 HF 或 export_reload 回退；
4. 两个 Actor 每次都到达 commit barrier；
5. 每个 Actor 的 Engine PID 在更新间保持不变；
6. 原生更新的 engine rebuild 时间为 0；
7. export root 最多保留一组 bootstrap raw/HF artifact；
8. 20-update native sync p95 首项目标不超过 60 秒；
9. 最终 checkpoint 通过 `fitmotn.cli.validate_checkpoint --require-exact rl`。

完成 20-update 正确性门槛后，再用相同 prompt、group size、token 上限和 update 数
分别运行 HF 与 Stage 5D。端到端时间以整个训练命令 wall time 为准；目标为 native
vLLM 至少达到 HF 的 2.0 倍。若未达到，应先分析 rollout、HF logprob/reward 和
NCCL sync 占比，再决定是否立项 patch-only 传输。

Stage 5E patch-only 参数子集同步的实现与三卡手动验收见
`docs/stage5e_patch_only_native_sync.md`。Stage 5D 默认行为仍为 `full_policy`。
