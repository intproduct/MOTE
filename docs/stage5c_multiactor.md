# Stage 5C：N 卡多 vLLM actor 并行 rollout

## 完成目标

Stage 5C 在 Stage 5B 的单 actor 隔离基础上增加任意数量的本地 rollout actor：

```text
HF/PyTorch trainer（单 GPU）
          │ 一次 policy export
          ├──────────┬──────────┐
          ▼          ▼          ▼
     actor 0     actor 1     actor N
       TP1         TP2         TP4
```

actor 数量没有写死的上限。单机通常使用不超过 8 张 GPU，更大的拓扑主要受操作系统、vLLM 和单机硬件限制。Stage 5C 仍然不是 DDP/FSDP 多卡训练；反向传播和 optimizer 位于一张 trainer GPU，多卡用于并行 rollout。

## 核心语义

配置使用 `rl.vllm_rollout_actors`：

```json
{
  "model": {"device": "cuda:0"},
  "rl": {
    "rollout_backend": "vllm",
    "vllm_execution_mode": "subprocess",
    "vllm_rollout_actors": [
      {
        "name": "rollout_0",
        "cuda_visible_devices": ["1"],
        "tensor_parallel_size": 1
      },
      {
        "name": "rollout_1",
        "cuda_visible_devices": ["2", "3"],
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.45,
        "max_model_len": 1024,
        "max_num_seqs": 8
      }
    ]
  }
}
```

每个 actor 支持：

- 独立 CUDA 设备集合；
- 独立 TP size；
- 独立 `gpu_memory_utilization`；
- 独立 `max_model_len` 和 `max_num_seqs`。
- 独立、可复现的 engine seed；默认根据全局 seed 和 actor index 派生，也可用 `seed_offset` 调整。

系统拒绝重复 actor 名称、设备集合重叠、actor 卡数与 TP size 不一致、越过调度器 CUDA mask，以及 actor 与 trainer 的设备重叠。

## 执行流程

每次 policy 同步只生成一份事务式 HF export。所有 actor 随后并行加载同一 export，并在进入 rollout 前形成 policy barrier：

```text
所有 actor policy version/fingerprint 一致
                         ↓
                    允许 rollout
```

展开后的 GRPO completion 行按照 group index 和 sample index 均匀分配到 actor。即使 `batch_size=1`，同一道题的 `group_size=16` 也能同时利用多个 actor。返回结果按照原始 row index 严格合并，然后才计算 reward、group advantage 和 MGPO 权重。

不同 actor 不共享相同 engine seed，避免同一 prompt 在多个副本上产生相关或重复轨迹。

任一 actor 发生错误、返回行数不符或 policy descriptor 不一致时，整批 rollout 作废。正式实验不允许混入部分 actor 的结果。

## 示例配置

- [3×A800：1 trainer + 2 个 TP1 actor](../fitmotn_config.stage5c_3xa800_2replica.example.json)
- [4×A800：1 trainer + 3 个 TP1 actor](../fitmotn_config.stage5c_4xa800.example.json)
- [8×A800：1 trainer + 7 个 TP1 actor](../fitmotn_config.stage5c_8xa800.example.json)

如果 8B 模型可以放入单张 A800，多个 TP1 actor 通常比单个大 TP actor 更适合增加 completion 吞吐。单卡放不下时可以把 actor 改成 TP2/TP4，但需要模型维度兼容和真实 GPU 验收。

# A800 一次性验收

## 1. 环境变量

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export CACHE_ROOT=/path/to/cache
export OUTPUT_ROOT=/path/to/outputs
```

3 卡测试：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
```

4 卡测试：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

8 卡测试：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

设备编号也可以是调度器提供的 GPU UUID/MIG UUID。actor 的数字编号按照 trainer 当前 CUDA 命名空间解析，不会绕过调度器 mask。

## 2. 只做预检，不启动训练

```bash
python scripts/run_stage5c_a800_acceptance.py \
  --config fitmotn_config.stage5c_3xa800_2replica.example.json \
  --skip-training
```

它会检查配置、模型 TP 维度、设备冲突、Python 测试，并保存 CUDA、vLLM、拓扑和 `nvidia-smi` 环境证据。

## 3. 两 update GPU smoke

```bash
python scripts/run_stage5c_a800_acceptance.py \
  --config fitmotn_config.stage5c_3xa800_2replica.example.json \
  --max-steps 2 \
  --min-updates 2 \
  --run-name stage5c_3gpu_smoke_u2
```

建议第一次先运行这一步。它会实际启动两个 vLLM actor、完成两次 online MGPO optimizer update，并自动执行所有验证器。

## 4. 二十 update 严格验收

```bash
python scripts/run_stage5c_a800_acceptance.py \
  --config fitmotn_config.stage5c_3xa800_2replica.example.json \
  --min-updates 20 \
  --run-name stage5c_3gpu_accept_u20
```

4 卡和 8 卡只需替换配置文件及 `CUDA_VISIBLE_DEVICES`。

## 一次性脚本自动完成的检查

[run_stage5c_a800_acceptance.py](../scripts/run_stage5c_a800_acceptance.py) 自动执行：

1. 记录 Python、Torch、Transformers、vLLM、CUDA 版本；
2. 保存 `nvidia-smi`、GPU 列表和 NVLink/PCIe topology；
3. 运行配置 doctor 和 Stage 5C topology preflight；
4. 运行 Stage 5B/5C、actor、rollout 和验证器相关测试；
5. 后台每 5 秒记录一次 GPU UUID、进程 PID 和显存；
6. 启动 online MGPO/GRPO 训练；
7. 验证所有 actor 的 CUDA mask 和实际 TP world size；
8. 验证所有 actor policy version/fingerprint barrier；
9. 验证 rollout 行无丢失、无重复、顺序恢复正确；
10. 验证 `policy_lag_updates=0` 且没有 HF fallback；
11. 验证事务式 sync artifacts 和最终 exact-resume checkpoint；
12. 输出 rollout、export、engine rebuild、old/new logprob 和 backward timing。

默认 evidence 位于：

```text
${OUTPUT_ROOT}/fitmotn_runs/stage5c_evidence/<timestamp>/
```

最重要的结果文件：

- `stage5c_acceptance_summary.json`：总结果；
- `stage5c_run.json`：多 actor 严格验证；
- `stage4_run.json`：online policy freshness 验证；
- `sync_artifacts.json`：事务式同步链；
- `final_checkpoint.json`：checkpoint 可恢复性；
- `nvidia_smi_compute_apps.csv`：GPU/进程/显存时间序列；
- `timing.log`：阶段耗时；
- `training.log`：完整训练日志。

测试后请优先保留或回传整个 evidence 目录，而不是只提供终端最后一行。

# 需要人工重点判断的部分

自动脚本可以判定结构正确性，但以下项目仍需要结合 A800 证据判断：

1. trainer 是否只占 GPU0，actor 是否只占各自分配卡；
2. 多 actor engine 重建后显存是否持续增长；
3. actor 退出后是否仍有 vLLM worker 残留；
4. 多 actor 相比单 actor 的 completion tokens/s 提升幅度；
5. export + engine rebuild 占总 update 时间的比例；
6. 4/8 卡下 CPU RAM、磁盘写入和进程启动是否成为瓶颈；
7. TP2/TP4 actor 的实际 vLLM 兼容性。

若 export + rebuild 长期超过 update 时间的 10%～15%，下一阶段应优先实现 subprocess 原生 patch 权重同步，而不是继续增加 actor。

# 尚不属于 Stage 5C 的能力

- trainer DDP/FSDP/ZeRO；
- 多机远程 actor；
- subprocess 原生 NCCL/IPC policy transfer；
- actor 动态扩缩容；
- actor 失败后继续使用剩余副本；
- dense 参数训练。

这些能力需要独立的训练、同步和 checkpoint 语义，不能由 Stage 5C 的多 rollout actor 直接等价替代。
