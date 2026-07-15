# Stage 5B：隔离式多卡 vLLM rollout

## 目标和完成边界

Stage 5B 把 FitMoTN online MGPO/GRPO 扩展为以下资源拓扑：

```text
单进程 HF trainer（1 GPU）
              │ policy export/reload
              ▼
独立 subprocess vLLM actor（1～N GPU，TP=N）
```

本阶段的代码目标是：

1. trainer 与 rollout actor 使用不重叠的 CUDA 设备集合；
2. 可靠表达和预检 2×A800 的 `1 trainer + 1 rollout`；
3. 可靠表达和预检 3×A800 的 `1 trainer + TP2 rollout`；
4. actor 回报实际 CUDA 可见设备、显存、进程和 engine TP 信息；
5. policy freshness、fingerprint 和 `export_reload` 正确性规则保持不变；
6. 提供机器验收脚本，代码完成与 GPU 验收结论严格分开。

本阶段不包含 trainer DDP/FSDP、多个 rollout replica、subprocess 原生 NCCL 权重传输和 dense 参数训练。这些能力的并行/保存语义不同，应作为后续阶段单独实现。

## 配置语义

`rl.vllm_actor_cuda_visible_devices` 使用启动 trainer 时所在命名空间的 CUDA 标识，可以是逻辑序号、GPU UUID 或 MIG UUID。若调度器已经设置 `CUDA_VISIBLE_DEVICES=4,5,6`，配置 `1,2` 会解析成调度器分配集合中的 `5,6`，不会越过调度器掩码。actor 会获得独立的 `CUDA_VISIBLE_DEVICES`，其中设备重新从零编号。

因此物理 GPU 1、2 上的 TP2 actor 应配置为：

```json
{
  "model": {
    "device": "cuda:0"
  },
  "rl": {
    "rollout_backend": "vllm",
    "vllm_execution_mode": "subprocess",
    "vllm_actor_start_method": "spawn",
    "vllm_actor_cuda_visible_devices": ["1", "2"],
    "vllm_device": "cuda:0",
    "vllm_tensor_parallel_size": 2
  }
}
```

这里 `vllm_device=cuda:0` 是 actor 内部重映射后的局部编号，不是 trainer 的 GPU 0。

系统会拒绝：

- actor 卡数与 TP size 不一致；
- 隔离模式使用 `forkserver`；
- 隔离模式使用 `vllm_device=cuda:1`；
- subprocess TP>1 却没有配置 actor 设备集合；
- doctor 能识别出的 trainer/actor 物理设备重叠；
- 本地模型维度不能被 TP size 整除。

## 2×A800

使用 [fitmotn_config.stage5b_2xa800.example.json](../fitmotn_config.stage5b_2xa800.example.json)：

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export CACHE_ROOT=/path/to/cache
export OUTPUT_ROOT=/path/to/outputs
export CUDA_VISIBLE_DEVICES=0,1

python -m fitmotn.cli.doctor \
  --config fitmotn_config.stage5b_2xa800.example.json

python -m fitmotn.cli.train_rl \
  --config_json fitmotn_config.stage5b_2xa800.example.json
```

## 3×A800（TP2 rollout）

使用 [fitmotn_config.stage5b_3xa800_tp2.example.json](../fitmotn_config.stage5b_3xa800_tp2.example.json)：

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export CACHE_ROOT=/path/to/cache
export OUTPUT_ROOT=/path/to/outputs
export CUDA_VISIBLE_DEVICES=0,1,2

python -m fitmotn.cli.doctor \
  --config fitmotn_config.stage5b_3xa800_tp2.example.json

python -m fitmotn.cli.train_rl \
  --config_json fitmotn_config.stage5b_3xa800_tp2.example.json
```

不要直接使用 TP3。具体模型的 hidden size、attention heads、KV heads 和 intermediate size 必须满足 TP 整除约束；对 Qwen 8B 优先验证 TP2。

## 资源与故障可观测性

actor 的启动、engine load 和周期性 ping 会记录：

- actor 实际 `CUDA_VISIBLE_DEVICES`；
- `torch.cuda.device_count()`；
- 每张 actor 逻辑 GPU 的名称和总显存；
- allocated/reserved/peak bytes；
- actor PID、进程组和可见子进程；
- engine device 和 tensor parallel size；
- policy version、fingerprint 和 export 目录。

`rl.vllm_actor_resource_log_every` 控制每多少个 rollout 请求采集一次；设为 `0` 可关闭周期采集。actor 建立独立进程组，正常关闭会先调用 vLLM shutdown；超时关闭会终止整个 actor 进程组，降低遗留 TP worker 的风险。

## 验收

配置级预检：

```bash
python scripts/validate_stage5b_multigpu.py \
  --config fitmotn_config.stage5b_2xa800.example.json
```

完成至少 20 个 update 后：

```bash
python scripts/validate_stage5b_multigpu.py \
  --config fitmotn_config.stage5b_2xa800.example.json \
  --rl-train-jsonl /path/to/rl_grpo/rl_train.jsonl \
  --min-updates 20 \
  --output-json /path/to/evidence/stage5b_2xa800.json
```

3 卡验收把配置换为 TP2 示例。通过条件：

- doctor 无错误；
- actor 观察到的 GPU 数等于配置卡数；
- engine TP size 等于配置值；
- 至少完成指定 update 数；
- 所有记录 `policy_lag_updates=0`；
- 未使用 HF fallback；
- actor resource handshake 存在。

此外必须人工保存 `nvidia-smi` 或 DCGM 证据，确认 trainer 仅占用训练卡、actor/TP workers 仅占用分配的 rollout 卡。PyTorch 无法从 trainer 进程准确读取其他进程占用，因此这项不能由单元测试代替。

## 已知性能边界

Stage 5B 继续使用 correctness-first `export_reload`。它每次真实 optimizer update 都可能保存、导出并重建 vLLM engine。多卡 TP 解决放置和模型容量问题，不自动解决同步开销。正式长跑前必须用 timing analyzer 判断 export + rebuild 占比；若持续超过 10%～15%，下一阶段应实现 subprocess 原生 patch 权重同步。

`trainable_mode=all` 已被运行时拒绝，因为当前 `patch_state_only_v2` checkpoint 不能保证 dense 更新恢复和导出。未来添加 `patch+dense` 前必须先完成 checkpoint v3。
