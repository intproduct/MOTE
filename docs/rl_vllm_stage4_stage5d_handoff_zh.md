# FitMoTN RL / vLLM Stage 4–5D 工作交接与后续计划

> 更新时间：2026-07-17  
> 当前分支：`mote-vllm-v1`  
> 文档生成前 HEAD：`9fdb7e18d9052384b1a326809220df9c51525960`  
> 远端分支：`origin/mote-vllm-v1`  
> 用途：供新的 Codex/ChatGPT 对话快速继承已有工程背景、实验结论、风险和下一步任务。

## 1. 当前工作的研究目标

FitMoTN 的主要目标是压缩/替换原始大模型 FFN 中的部分线性层，同时尽可能维持或恢复模型推理能力。本轮工作的焦点是强化学习部分：让 FitMoTN 模型能够使用 vLLM 高吞吐生成 rollout，并完成真正的在线 GRPO/MGPO 训练。

在线训练期望形成如下闭环：

1. GPU0 上的 Hugging Face Trainer 使用当前 FitMoTN policy 计算 logprob、loss 和梯度；
2. optimizer update 后，把最新 policy 权重同步给一个或多个常驻 vLLM Actor；
3. vLLM Actor 使用最新 policy 为下一批 prompt 生成多条 completion；
4. 根据 GSM8K 最终答案计算 reward；
5. 计算组内 advantage 和 MGPO 难度权重；
6. 继续下一次 optimizer update，且不允许 policy lag、HF fallback 或静默使用旧权重。

工程改造不仅为了“训练更快”，还为了支持更大的 group size、更稳定的能力边界估计和更多有效的组内正负样本对比。

## 2. 已完成的主要阶段

### 2.1 Stage 4：FitMoTN 模型接入 vLLM

已完成：

- 将 FitMoTN checkpoint 导出为完整 Hugging Face roundtrip export；
- 通过 Transformers backend 让 vLLM 加载 FitMoTN 自定义结构；
- 增加静态 vLLM contract、配置、权重、tokenizer、fingerprint 和 parity 检查；
- 修复 Qwen3 未绑定 `lm_head` 被错误覆盖的问题；
- 验证输入 embedding 和输出 `lm_head` 保持 Qwen3 原模型语义，FitMoTN patch 只作用于预期的 FFN 结构；
- 增加 HF/vLLM 输出一致性诊断和 GPU smoke gate；
- 支持 HF rollout 与 vLLM rollout 基准对比。

相关提交：

- `fa7cf56`：vLLM RL rollout 与原生同步能力门槛；
- `efd4a4f`：Stage 4 vLLM 集成加固；
- `e47c47e`：Stage 4F/4G 加固；
- `015324c`：导出和 GPU 验收加固；
- `60b28ee`：vLLM parity 诊断；
- `0e4c610`：保留 Qwen3 untied lm head。

### 2.2 Stage 5A：RL checkpoint 可靠性

已完成：

- 支持按 `rl.save_every_updates` 在训练过程中保存多个 checkpoint；
- 支持 optimizer、训练位置、随机状态等精确恢复信息；
- 支持 transactional save、临时目录清理和 checkpoint 保留策略；
- 区分仅加载模型权重和 exact RL resume；
- checkpoint 保存失败可配置为直接终止，避免训练继续但没有可恢复产物。

相关提交：`75668c6`。

### 2.3 Stage 5B/5C：多 GPU rollout Actor

已完成：

- Trainer 和 rollout 显卡显式隔离；
- 支持 subprocess vLLM Actor；
- 支持多个 rollout Actor 分片生成 completion 并合并结果；
- Actor 故障传播、资源日志、超时和一致性检查；
- 支持向更多 Actor 扩展，而非把全部生成压在单张卡上；
- 三张 A800 环境已经验证 GPU0 Trainer、GPU1/2 两个 TP1 vLLM Actor 的主流程。

相关提交：

- `5ff3d44`：可扩展多 GPU vLLM rollout Actor；
- `c4d5577`：vLLM 0.19 Actor 设备兼容性修复。

必须注意：当前“多卡能力”主要是 **1 张 Trainer 卡 + N 张 rollout 卡**。当前没有完成 Trainer 侧 DDP/FSDP/ZeRO 数据并行训练。也就是说，GPU0 上仍是单进程 HF 训练，GPU1..N 用于并行 vLLM 推理。对于 `patch_only` 训练目前基本符合实验需求，但不能把它描述成通用多卡训练框架。

### 2.4 Stage 5D：常驻 Engine 与 NCCL 原生权重同步

旧实现 `export_reload` 每次 update 都执行：

1. 导出完整 HF 模型；
2. 关闭或重建 vLLM Engine；
3. 从磁盘重新加载权重；
4. 再开始下一轮 rollout。

这成为在线 RL 的主要工程瓶颈。

Stage 5D 已完成：

- 针对 vLLM 0.19.0 `update_only` API 实现 NCCL 权重传输；
- Engine 常驻，不再每个 update 重建；
- Trainer 广播权重到两个 TP1 Actor；
- 增加 policy version、fingerprint 和 commit barrier；
- 只有所有 Actor 均完成同一 policy version 更新后，才允许下一轮 rollout；
- 禁止静默退回 `export_reload` 或 HF rollout；
- 增加覆盖率、部分映射失败、同步后校验和错误传播；
- 增加 Actor teardown 超时、进程组 TERM/KILL 和 `atexit` 清理，避免训练完成后无限卡住。

相关提交：

- `2f3b201`：常驻 vLLM 原生权重同步；
- `9fdb7e1`：限制 Actor teardown，修复退出挂死。

## 3. 当前架构与显卡布局

当前在 3×A800 80GB 上使用：

| GPU | 角色 | 说明 |
|---|---|---|
| GPU0 | HF Trainer | 加载 FitMoTN policy，计算 old/new logprob、backward、optimizer update |
| GPU1 | vLLM Actor 0 | TP1，负责部分 completion |
| GPU2 | vLLM Actor 1 | TP1，负责其余 completion |

关键约束：

- subprocess NCCL 原生同步当前要求 rollout Actor 为 TP1；
- 可以增加更多 TP1 Actor，通常可扩展到 8 卡以内，但需逐机验证端口、NCCL、显存和 Actor teardown；
- 单卡 H200 141GB 可以运行单 Engine，但不能同时复现当前 Trainer/Actor 物理隔离；
- 当前原生同步传输完整映射权重，而不是只传训练后发生变化的 patch 参数；
- Trainer 多卡训练属于未来工作，尚未实现。

## 4. 模型导出与加载注意事项

### 4.1 raw checkpoint 与 Stage 4C HF export 的区别

训练输出中的 `final_model` 可以作为 HF Trainer 恢复权重的来源，例如：

```text
/work/home/sugang2025/qxfang/MOTE-g/fitmotn_runs/
  fitmotn_k8_nodecay_7e_tp8_tc/final_model
```

但是 vLLM 初始加载需要完整的 Stage 4C Hugging Face export，其中包含：

- HF `config.json`；
- 自定义 FitMoTN remote code；
- 完整分片权重；
- tokenizer；
- generation 配置；
- FitMoTN 导出元数据和静态 contract。

典型导出命令：

```bash
python -m fitmotn.cli.export_hf \
  --checkpoint_dir /work/home/sugang2025/qxfang/MOTE-g/fitmotn_runs/fitmotn_k8_nodecay_7e_tp8_tc/final_model \
  --output_dir /work/home/sugang2025/qxfang/MOTE-g/fitmotn_exports/k8_nodecay_7e_tp8_tc_hf \
  --no-metadata_only \
  --torch_dtype bfloat16 \
  --roundtrip_device cuda \
  --validate_roundtrip
```

如果代码改动仅涉及 benchmark、Actor 调度或同步逻辑，通常不需要重新导出；如果改动 remote model code、导出格式、权重键映射、embedding/lm head 恢复逻辑，则应删除旧 export 后重新导出。

### 4.2 已遇到但不一定阻塞的问题

- Mistral tokenizer regex warning 与当前 Qwen3 训练主问题无直接关系，但加载 Mistral tokenizer 时应使用相应修复参数；
- `requests`、`urllib3`、`chardet`/`charset-normalizer` 版本警告来自环境依赖组合，在 `pip check` 无 broken requirements 且训练不涉及网络请求时通常不是 RL 阻塞项；
- vLLM static preflight 通过不等于 GPU generation 通过，CUDA attention、KV cache、token generation 和多 Actor 必须在 GPU 上验收；
- HF 与 vLLM 的文本不能只比较第一 token，应使用相同 tokenizer、token IDs、sampling 参数、stop 条件和完整 completion 进行 parity 分析。

## 5. 已取得的性能证据

### 5.1 单卡 rollout benchmark（实测）

同一批预分词 prompt 下：

| 指标 | HF | vLLM | 改善 |
|---|---:|---:|---:|
| generated tokens/s | 38.3 | 118.1 | 3.09× |
| prompts/s | 0.154 | 0.608 | 3.96× |
| 64 条生成时间 | 416.8 s | 105.2 s | 约 4× |
| 模型加载时间 | 109.0 s | 134.7 s | vLLM 多 25.8 s |
| GPU 显存峰值 | 约 14.3 GiB | 约 52.5 GiB | vLLM 用显存换吞吐 |

HF 平均输出约 249 tokens，vLLM 约 194 tokens，因此报告时应把 `generated tokens/s` 作为主要速度指标，`prompts/s` 只能作为辅助指标。

原始证据曾位于本地：`/Users/qixuanfang/Downloads/comparison.json`。

### 5.2 原生同步（实测）

| 项目 | export_reload | NCCL 首次 | NCCL 稳定阶段 |
|---|---:|---:|---:|
| policy 同步 | 约 420.3 s | 30.7 s | 6.48 s |
| 同步加速 | 1× | 13.8× | 64.8× |

补充信息：

- 首次完整权重传输约 23.93 s；
- 第二次传输约 2.79 s；
- 每个 Actor 接收 831 个 tensor，约 14.13GB；
- 旧路径每次 update 约产生 209–210 s Engine rebuild；
- 原生同步路径 update 间 Engine rebuild 为 0。

### 5.3 两步 successful-update 对比（实测）

- 旧路径 successful optimizer update：约 648–650 s；
- NCCL 原生路径第二次 update：约 24.2 s；
- 端到端 successful-update 对比约 26.8×。

这个数字来自短程 smoke，不能直接等价为长程训练整体加速倍数。

### 5.4 20 步 online MGPO（主流程实测）

- 开始时间：00:31:03；
- 完成第 20 个 optimizer update：01:20:28；
- 主流程耗时：49 分 25 秒，即 2965 s；
- 启动期包含 bootstrap export 约 431 s、双 vLLM Engine 首次加载约 222 s；
- 扣除已知启动项后约 2312 s，约 115.6 s/update；
- 两个 Actor 全程使用同一最新 policy；
- 无 HF fallback；
- 无 export_reload 回退；
- Engine 常驻；
- checkpoint 正常保存；
- policy version/fingerprint 一致。

这次运行在完成全部 update 后，旧 teardown 路径退出挂住并被手动终止。`9fdb7e1` 已修复 teardown，但该修复仍应在三卡 GPU 上再做一次 2–5 步退出验收。

### 5.5 长程速度估算（不是实测）

使用旧 export_reload 的同负载时间估算约 4 小时 16 分，原生同步约 49 分 25 秒，约 5.2×。这个数字必须标注为“同负载估算”，不能写成已完成的同条件长程 A/B 实测。

## 6. 旧 MGPO 实验复盘

旧 HF rollout + boundary-only MGPO 实验：

- GSM8K：63.06 → 63.15，约 +0.09 分；
- Flexible：61.71 → 60.65，约 -1.06 分；
- 训练集：911 道 boundary prompts；
- group size：8；
- optimizer updates：228；
- rollout micro-steps：2093；
- 有效 optimizer micro-steps：1824；
- zero-advantage skip：269，占全部 micro-step 的 12.85%；
- 其中全错组 178，全对组 91；
- 相比 1824 个有效组，额外无效 rollout 开销约 14.75%；
- `patch_only` 可训练参数 116,785,152 / 6,948,566,016，约 1.68%。

对 1319 道 GSM8K 测试题而言，+0.09 分大致只相当于多答对 1 道题，因此不能认为已经证明 MGPO 显著有效。Flexible 下跌还提示可能存在格式过拟合、分布偏移或推理多样性下降。

正确的信号解释是：

- 二值最终答案 reward 能产生训练信号，但信用分配粗粒度；
- 全对或全错组的组内 advantage 为零；
- G=8 的正确率分辨率为 12.5%，G=32 为 3.125%；
- 当真实正确率接近 0.5 时，二项估计标准误差约从 G=8 的 0.177 降至 G=32 的 0.088；
- 增大 group size 有助于稳定边界估计和增加 mixed-reward group，但不能保证采样完全独立；高概率模式相关性会降低有效样本量。

不应使用以下不严谨说法：

- “G 小必然让 advantage 绝对值更小”；
- “标量 policy loss 接近零就说明没有梯度”；
- “论文建议 G=32”，除非能给出论文正式配置或具体页码。部分材料里的 K=32 可能是测试时采样，而非 MGPO 训练 group size。

更准确的表述是：**旧实验的有效训练信号部分稀疏、难度估计噪声较大、最终答案 reward 的信用分配较粗；vLLM 改造使 G=16/32 和多 Actor 在线采样成为可行。**

旧实验文件曾位于：

- `/Users/qixuanfang/Downloads/rl_grpo/rl_run_summary.json`；
- `/Users/qixuanfang/Downloads/rl_grpo/rl_train.jsonl`；
- `/Users/qixuanfang/Downloads/rl_grpo/trainable_summary.json`。

## 7. Boundary 数据构建的意义

旧能力边界构建：

- GSM8K train：7473 题；
- boundary prompts：911 题，占 12.2%；
- 运行时间：6 月 23 日 21:30 至 6 月 29 日 21:50；
- 总耗时：144 小时 20 分。

流程是多轨迹采样、答案验证、难度估计、边界筛选和正确轨迹保存。它证明 rollout 是实际研究流程中的核心瓶颈，也产出了可用于 RL 的边界数据。

但是当前 boundary build CLI 仍主要走 HF `model.generate`。已有 3.09× vLLM benchmark 说明迁移后有显著潜力，但不能声称这次 6 天流程已经被实测缩短。后续应把 boundary build 正式迁移到多 Actor vLLM，再运行同条件 A/B。

## 8. 当前推荐的 200-update online MGPO pilot

目标：在三张 A800 上运行第一轮真实在线 MGPO 工程与信号试验。

推荐关键参数：

```json
{
  "max_steps": 200,
  "batch_size": 1,
  "group_size": 16,
  "grad_accum": 1,
  "max_new_tokens": 256,
  "temperature": 1.1,
  "top_p": 0.95,
  "lr": 5e-7,
  "trainable_mode": "patch_only",
  "rollout_backend": "vllm",
  "vllm_execution_mode": "subprocess",
  "vllm_sync_strategy": "weight_transfer_nccl",
  "vllm_sync_every_updates": 1,
  "vllm_gpu_memory_utilization": 0.45,
  "vllm_max_model_len": 1024,
  "vllm_max_num_seqs": 8,
  "mgpo_enabled": true,
  "mgpo_p0": 0.5,
  "mgpo_gamma": 1.0,
  "save_every_updates": 25,
  "eval_every_updates": 0,
  "seed": 1234
}
```

Actor 布局：

```json
[
  {"name": "rollout_0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
  {"name": "rollout_1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1}
]
```

重要事项：

1. 不要保留 smoke 配置中的 `"debug_num_prompts": 16`；真实训练应删除该字段，使其为默认 `null`；
2. `max_steps` 是成功 optimizer update 数，zero-advantage 重试会增加 rollout 数和总耗时；
3. 200×G16 至少产生 3200 条用于成功 update 的 completion，显著少于旧实验约 1.46 万条有效 completion，因此它是 pilot，不是最终效果定论；
4. 20 步 smoke 使用 G=4、64 tokens，不能直接外推 G=16、256 tokens 的时间；
5. 若要控制在 10 小时内，扣除启动时间后的稳定 update 均值最好不超过约 175 s；
6. 不建议为了压时间把 `max_new_tokens` 强行降到 128，这会引入明显的推理截断；如果超时，优先把本轮缩短为 100–120 updates；
7. `vllm_enforce_eager=true` 是当前已经验证的稳妥路径，先不要在真实实验中同时引入 CUDA graph 变量。

运行命令：

```bash
cd /work/home/sugang2025/qxfang/MOTE-g
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m fitmotn.cli.train \
  --config_json /work/home/sugang2025/qxfang/MOTE-g/RLconfig/online_mgpo_g16_u200.json
```

## 9. 新发现的数据采样风险

当前 `train/rl_controller.py` 的 `_cycle_batch` 按数据集顺序循环读取，`seed` 只控制生成随机性，不会打乱 GSM8K 题目。

因此：

- 200 steps、batch size 1 主要训练 GSM8K train 的前约 200 道题；
- zero-advantage 重试会继续推进数据位置，因此实际看到的题数可能略多；
- 当前配置不能声称已经从 7473 题中随机采样 200 题；
- 这不影响验证 online MGPO 工程闭环，但会削弱效果实验的统计代表性。

这是下一项优先代码改动：增加可复现的 shuffle/sampler，并把 sampler 顺序、epoch 和 RNG 状态写入 exact-resume checkpoint。正式 MGPO 对照实验应在该功能完成并验收后运行。

## 10. 200 步实验应收集的证据

不要只看最终 GSM8K 分数。至少分析：

- 每个 update 的 wall time；
- generate、reward、old logprob、new logprob/backward、NCCL sync 分项耗时；
- Actor Engine PID 是否跨 update 保持不变；
- policy version、fingerprint 和 policy lag；
- HF fallback/export_reload fallback 是否发生；
- reward mean/std；
- 每题 `correct_count / group_size`；
- 全对、全错和 mixed group 比例；
- zero-advantage skip 和额外 rollout 比例；
- advantage absolute mean；
- MGPO raw/normalized weight 分布；
- response 长度和截断比例；
- grad norm、ratio、clip fraction、KL；
- checkpoint-25/50/75/.../200 是否可加载和 exact resume。

训练后建议评测：

- 初始模型；
- checkpoint-50；
- checkpoint-100；
- checkpoint-150；
- checkpoint-200/final。

每个 checkpoint 使用相同 prompt 模板、generation 参数和解析规则测试：

- GSM8K 标准分数；
- Flexible 分数；
- 最好重复不同 evaluation seed，避免把约 1 道题的波动误认为有效提升。

## 11. 当前测试状态

本地 CPU/模拟测试在 teardown 修复后：

- 266 passed；
- 3 skipped；
- 1 warning；
- 32 subtests passed。

三卡 GPU 已通过的主要内容：

- FitMoTN vLLM GPU generation；
- HF/vLLM rollout benchmark；
- GPU0 Trainer + GPU1/2 Actor 隔离；
- 两 Actor completion 分片与合并；
- 1/2/20 update 原生 NCCL 同步主流程；
- policy version/fingerprint/commit barrier；
- 常驻 Engine；
- checkpoint 保存；
- 无 HF/export_reload 回退。

尚需 GPU 补验：

- `9fdb7e1` teardown 修复后的 2–5 update 自动退出；
- 退出后 `nvidia-smi` 和进程表无残留 Actor/EngineCore；
- G=16、256 tokens 的真实显存、吞吐和稳定时间；
- 200-update 过程中 checkpoint 和 exact resume；
- 真正的同条件长程 HF vs vLLM A/B（如还需要严格速度论文证据）。

## 12. 后续代码计划与优先级

### P0：立即完成

1. 在三卡机器上做 teardown 修复后的 2–5 update 验收；
2. 启动 G=16、200-update online MGPO pilot；
3. 补可复现训练集 shuffle/sampler；
4. 从 checkpoint-50/100/150/200 做统一离线评测；
5. 统计 G=16 相比旧 G=8 的 mixed group、zero-advantage 和难度估计分布。

### P1：近期工程优化

1. **Patch-only 增量同步**：当前每次广播 831 tensors、约 14.13GB；未来只同步实际变化的 patch 参数，进一步降低 NCCL 时间；
2. 把 boundary build CLI 迁移到多 Actor vLLM；
3. 增加 G=8/G=16/G=32 的同数据、同 token budget 消融；
4. 增加训练过程自动 ETA、分阶段耗时汇总和零优势统计；
5. 增加固定 eval manifest，保证各 checkpoint 比较一致；
6. 增加 rollout semantic diversity/重复轨迹分析，估计有效 group size。

### P2：模型与算法研究

1. 比较 boundary-only MGPO、full GSM8K MGPO、普通 GRPO；
2. 研究 boundary oversampling，而不是把 boundary 作为唯一训练分布；
3. 改善 reward：格式 reward、部分过程 reward、错误步骤定位或 verifier；
4. 研究 long-to-short、长度惩罚和防止输出模式坍缩；
5. 在未来需要时增加 `patch + last-quarter dense` 或全参训练模式，但这不是当前实验目标；
6. 如果 G=16/32 和改进 reward 后仍没有收益，再判断 1.68% patch-only 可训练空间是否构成能力上限。

### P3：通用多卡训练能力

1. Trainer 侧 DDP/FSDP/ZeRO；
2. Trainer world size 与 rollout Actor 池的联合拓扑规划；
3. 多节点 NCCL rendezvous、故障恢复和弹性 Actor；
4. TP2+ 自定义 ADTN tensor sharding 验证；
5. N 卡大规模扩展和跨节点性能建模。

## 13. 汇报/PPT 建议与口径

目前 3–5 张工程汇报可以覆盖：

1. 旧 boundary 构建耗时 144 小时 20 分，rollout 是主要瓶颈；
2. Stage 4–5D 分层演进：可加载、隔离、多 Actor、原生同步；
3. 单卡 vLLM rollout 3.09× token 吞吐以及显存交换；
4. 原生同步 420.3 s → 6.48 s，Engine rebuild 归零；
5. 20-update 主流程 49 分 25 秒及 commit barrier 架构。

建议新增动机页：

> 原有 MGPO 收益有限：小组采样使边界估计较粗，有效训练信号部分稀疏。

该页可放：

- GSM8K +0.09、Flexible -1.06；
- 2093 micro-steps 中 269 次零优势跳过；
- G=8 与 G=32 的正确率分辨率和标准误差；
- 结论：vLLM 改造同时解决吞吐和扩大 group size 的工程可行性。

汇报数字标注规则：

- `3.09×`：单卡、单次 benchmark 实测；
- `64.8×`：稳定原生 policy sync 实测；
- `26.8×`：两步 smoke successful-update 对比；
- `5.2×`：同负载长程估算，非严格 A/B 实测；
- `49:25`：20 update 主流程，不含旧版退出挂死；
- boundary 六天流程尚未用 vLLM 重新实测。

## 14. 新对话的推荐起始提示

可以在新对话中直接写：

> 请先完整阅读 `docs/rl_vllm_stage4_stage5d_handoff_zh.md`，并结合 `docs/stage4_vllm_validation.md`、`docs/stage5b_multigpu.md`、`docs/stage5c_multiactor.md` 和 `docs/stage5d_full_weight_native_sync.md` 继承当前工作。当前分支为 `mote-vllm-v1`，Stage 5D 主功能已完成。下一步先确认 teardown GPU 验收状态，再分析或启动 G=16、200-update online MGPO pilot；不要重新实现已经完成的 Stage 4–5D 功能。

## 15. 相关仓库文档

- `docs/vllm_readiness_audit.md`：vLLM readiness 审计；
- `docs/stage4_vllm_validation.md`：Stage 4 验证方法；
- `docs/rollout_benchmark_1xa800.md`：单卡 rollout benchmark；
- `docs/stage5b_multigpu.md`：多 GPU 资源隔离；
- `docs/stage5c_multiactor.md`：多 Actor 调度；
- `docs/stage5d_full_weight_native_sync.md`：原生 NCCL 同步与验收。

## 16. 当前最终判断

Stage 4–5D 已经把 FitMoTN RL 从“单卡 HF 慢速 rollout + 每次重载模型”推进到“Trainer/Actor 分离、双 Actor vLLM 并行、常驻 Engine、每 update 原生同步、policy 一致性 barrier”的可运行在线 MGPO 工程框架。

工程加速已经得到明确证据，但算法收益尚未得到证明。下一阶段的核心不再是继续证明 vLLM 能否运行，而是：

1. 用合理的数据采样和 G=16/32 验证有效训练信号是否改善；
2. 用统一 checkpoint 评测判断 GSM8K 和 Flexible 是否真正提升；
3. 如果仍无收益，再依次定位数据、reward、训练规模和 patch-only 参数空间瓶颈。
