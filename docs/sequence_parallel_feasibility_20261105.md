# MoTE 跨卡序列并行：成熟库适配审查

2026-11-05 · 发布工作树基线 98659ed3 · 审查与小规模兼容性测试

## 结论与优先路线

统一训练入口、配置、数据约定和验收流程可行。现阶段应优先适配 DeepSpeed-Ulysses，保留 Megatron Core / Bridge 的 NVIDIA 专用后端路线。Linux 两进程实测已验证小型 Qwen3、使用 Llama3.1 RoPE 配置的 Llama，以及关闭裁剪、仅计算 CE 的 Qwen3-MoTE 前反向一致。不同模型和设备通过适配层接入，通信算法、参数分片和分布式优化器尽量复用上游库。

当前仓库尚不具备直接开启跨卡序列并行的条件。已经测出的阻碍集中在 MoTE 全局路由语义、训练标签与位置传递、数据分组以及分布式检查点；这些是项目自身的接入工作。审查没有发现必须另写一套通用序列并行通信算法的理由。

| 路线 | 与当前代码的关系 | 判断 |
| --- | --- | --- |
| DeepSpeed-Ulysses / ALST | 官方提供 Hugging Face 注意力与 DataLoader 适配器，接近当前 Trainer 和模型路径 | 优先开展正确性适配，再接 ZeRO 与长序列测试 |
| Megatron-SP + Context Parallelism | 需要接入 Megatron 模型接口、分片权重映射、训练状态和优化器；MoTE 投影不等于标准 Dense / MoE 层 | 作为 NVIDIA Linux 多卡后端，迁移成本更高 |
| 自行实现通信算法 | 与成熟库重复，扩大跨模型与硬件维护范围 | 目前不采用；仅针对经测试确认的缺口补适配层 |

这里需区分三种功能：数据并行分配不同样本；Megatron 的 SP 通常配合张量并行减少部分激活；CP 或 Ulysses 才把一条长序列的计算分散到多卡。ZeRO / FSDP 主要减少参数、梯度和优化器状态，可以与序列切分组合。Megatron 官方明确区分 SP 与 CP，并要求启用 SP 时张量并行度大于 1。[Megatron CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)、[Bridge 并行指南](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)

16K～32K 能否容纳还取决于模型规模、精度、有效监督 token、优化器、激活重计算、词表损失和互连。单独扩大并行度不等于同比提高吞吐。有效全局 batch 按数据并行副本数计算，序列并行组内各卡共同处理同一逻辑样本。

## 模型与平台适配边界

| 模型 | 应保留的计算语义 | 优先验证内容 |
| --- | --- | --- |
| Qwen3 普通全注意力版本 | 因果注意力、GQA、RoPE、QK 归一化 | Ulysses 官方 HF 接口；按具体配置核对头数与并行度 |
| Llama3.1 | GQA、全局位置、对应的 RoPE 缩放 | 复用全注意力适配层，另测实际 3.1 配置与长位置 |
| Gemma3 | 滑动窗口与全局注意力交替，两类位置编码 | 必须验证窗口掩码；先限定文本路径，视觉输入另列验收 |
| Qwen3.5 | 全注意力与 Gated DeltaNet 混合，包含短卷积和递归状态 | 先验证现有混合注意力适配，不能仅注册普通注意力函数 |

DeepSpeed 0.19.9 的 HF Ulysses 包装会丢弃常规张量注意力掩码；其 flex 路径也在源码中明确提示需补非标准窗口支持。因此，“能返回输出”不足以判定 Gemma3 保持原模型行为，应直接比较完整 logits 和梯度。[固定版本源码](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.9/deepspeed/runtime/sequence_parallel/ulysses_sp.py)

Qwen3.5 已有可复用的公开方向：Unsloth 合入了专门面向 Gated DeltaNet 的 Ulysses 适配，让分片投影经 all-to-all 后由每个头分片处理完整序列的卷积与递归，再还原序列切分。作者报告了小配置双进程及双 GPU 对照，真实大权重与更多 GPU 的覆盖仍需单独验证。该证据支持优先适配现有实现。[上游实现与测试范围](https://github.com/unslothai/unsloth/pull/13920)

| 平台 | 当前证据 | 接入条件 |
| --- | --- | --- |
| A800 80GB、H100、H200，Linux 同构多卡 | 成熟 NVIDIA 并行软件栈的优先目标；本轮没有这些机器的实测 | 固定驱动、PyTorch、NCCL、注意力后端，逐级测试 |
| 本机 Windows：RTX PRO 4000 + 两张 V100 | 实测 NCCL 不可用，Gloo 初始化失败；卡的计算能力和显存不同 | 不能据此认证生产跨卡训练；需独立平台验证 |
| 同机 Linux / WSL 异构 NVIDIA | 可用于探索共同支持的通信、精度与算子组合 | GPU 实测需核对卡序号、P2P、互连、最小显存和慢卡限制 |
| BW1000 DCU | 厂商已有软件栈与模型适配资料；本轮没有设备实测 | 核验厂商 PyTorch、集合通信、前反向注意力与线性注意力算子 |

共同框架应提供平台能力检查和明确失败原因。A800 与 H100/H200 可共用高层后端接口；V100 不应直接套用较新 GPU 的 FlashAttention 内核要求；BW1000 需要厂商后端，不能由 NVIDIA 运行结果推导兼容性。[PyTorch 分布式平台说明](https://docs.pytorch.org/docs/stable/distributed.html)、[FlashAttention 支持范围](https://github.com/Dao-AILab/flash-attention)、[海光开发者平台](https://developer.sourcefind.cn/)

## 本地可复现的语义测试

以下数值来自实际调用当前代码的 CPU 小张量测试。输入与测试代码保存在 research/sequence_parallel_compat_20261105/probe_contracts.py，完整结果见同目录 contracts.json。它们用于说明具体接入缺口，不代表模型评测成绩。

| 检查 | 完整序列 | 两个局部分片 | 含义 |
| --- | --- | --- | --- |
| TopKGate 容量裁剪 | 保留 8 个路由 | 仅保留 4 个，4 个 token 的掩码改变 | 局部容量和局部排名改变了函数 |
| 关闭裁剪后的负载均衡损失 | 1.000000 | 局部损失平均 1.951212 | 全局均值的乘积不等于局部乘积的平均 |
| 8-token 样本的监督目标 | 7 个 | 局部移位剩 6 个 | 分片边界的下一个 token 丢失 |
| 同一示例的交叉熵 | 3.582460 | 局部移位 3.619435 | 全局预移位恢复到 3.582459 |
| 官方预移位 batch 传给现有 Trainer | 应传位置并计算 shift_labels 损失 | 未传 position_ids；返回损失为 None | 需要修改训练接口约定 |
| 两进程原子检查点入口 | 应保存全部 rank 状态 | 现有代码明确拒绝 world_size > 1 | 需接成熟库分布式保存流程 |

容量问题来自 gate.py 的容量计算及按概率或位置进行的 token 筛选。QuantumGate 产生的路由 logits 仍由相同的 TopKGate / SoftGate 框架处理，因此量子路由同样需要审查全局统计。适配不能通过悄悄关闭裁剪或辅助损失来宣称等价。

训练接口还需处理 loss_weights、KD 教师目标、无监督 token 的分片和统计分母。全局损失应按有效 token 或有效权重合并；参数梯度采用 sum 还是 mean，必须与 DeepSpeed / DDP 的归约约定配套，避免重复除以并行度。

数据分配的源码检查和模拟 rank 测试表明，现有采样器使用 WORLD rank 切样本。若采用“SP 组共享同一样本”的分配方式，采样器需改为按 DP 副本分片；若采用官方 Ulysses DataLoader 适配器，则由它收集各 rank 的样本并依次切分，需同步重算迭代数、梯度累积与恢复游标。两种方式不能叠加使用。[Ulysses 官方接入说明](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)

## 成熟库接入测试记录

Windows 隔离环境沿用只读基础依赖 PyTorch 2.10.0+cu128、Transformers 4.57.6、Accelerate 1.15.0，并单独安装测试库。冻结实验环境没有升级。

DeepSpeed 0.19.9 原始源码包在 Windows 打包时缺少 bin/deepspeed.bat。为继续 CPU 诊断，仅在临时下载目录将安装入口脚本列表置空；Ulysses、通信和训练源码没有修改。随后官方 HF Ulysses 模块导入成功。两个真实进程尝试 Gloo 通信时，默认方式和显式 UV 方式均在初始化阶段报 unsupported gloo device，尚未进入模型前向。

Megatron Core 0.20.0 在隔离环境中成功构建并安装。实际导入训练相关模块时报缺少 triton；Transformer Engine 也未安装。固定版本的普通 DotProductAttention 明确拒绝 CP > 1，并要求使用 TE 注意力实现。因此本轮 Windows 结果应记为环境阻塞，不能写成 CP 数值测试通过或算法不可行。

随后在已有 Ubuntu 24.04 WSL 内创建独立 CPU 环境，使用未经源码修改的 DeepSpeed 0.19.9、PyTorch 2.10.0+cpu、Transformers 4.57.6 和 Accelerate 1.15.0。测试关闭了可选的共享内存 C++ 加速扩展，实际通信由两个独立进程通过 Gloo all-to-all 完成。随机小模型采用 2 层、隐藏维度 32、4 个 query 头、2 个 KV 头、16 token、FP32；以完整序列计算为参照，绝对误差容差为 2e-5。

| 两进程测试对象 | logits 最大绝对误差 | 参数梯度最大绝对误差 | 判定 |
| --- | --- | --- | --- |
| Qwen3 | 0 | 2.98e-8 | 通过 |
| Llama，采用 3.1 RoPE 配置 | 0 | 2.98e-8 | 通过；尚未覆盖长位置 |
| Qwen3-MoTE，关闭裁剪，仅 CE | 0 | 1.19e-7 | 此限定配置通过 |
| Qwen3-MoTE，开启裁剪 | 0.428700 | 0.641977 | 不一致，需全局路由适配 |
| Gemma3，窗口 + 全局注意力 | 0.287804 | 0.350032 | 不一致，需窗口适配 |
| Gemma3 全局注意力诊断对照 | 0 | 7.82e-8 | 通过；用于定位窗口问题 |

Gemma3 的对照仅用于定位原因：保留窗口的真实结构失败，改为全局注意力的小配置通过，配合包装器丢弃窗口掩码的源码证据，支持将窗口处理列为必要接入工作。该对照不作为 Gemma3 兼容方案。MoTE 关闭裁剪的通过项仅包含 CE；辅助损失仍须根据前节反例做全局归约。这两种负结果说明通用接口需要能力检查，不能统一打上“已支持”的标记。

六个场景的数据、位置和预移位标签检查均通过。把官方 batch 直接交给当前 FitMoTNTrainer，六个场景都返回 None 损失，进一步确认现有训练入口仍需适配。完整结果位于 research/sequence_parallel_compat_20261105/linux_run_04，两个 rank 的结果一致。

Megatron 在 Linux 补齐 Triton 3.6.0 后，实际导入和 CP=2 配置构造成功；尝试普通注意力时触发官方“CP 仅支持 TEDotProductAttention”的检查。当前 CPU 环境没有 Transformer Engine，因此验证止于此处，未执行 Megatron CP 数值或 GPU 吞吐测试。Qwen3.5 与 BW1000 本轮为来源及接口审查，尚无本地前反向实测。

测试脚本分别为 probe_ulysses.py、probe_megatron.py。在 requirements-probe.txt 对应的独立 CPU 环境中，运行 probe_ulysses.py --gloo-only --output 新目录可复现；输出目录必须不存在。测试使用官方注册函数、官方 DataLoader 适配器和真实反向传播，并在探测代码中显式计算预移位 CE、对各副本的参数梯度求和以还原完整序列目标。此层通过后还需要完整 Trainer、ZeRO、保存恢复与真实长序列 GPU 验证。

## 适配范围与验收顺序

统一框架以现有训练功能为中心，分成四个明确接口：模型适配决定注意力类型和 MoTE 全局语义；并行后端调用 Ulysses 或 Megatron；训练约定统一数据、标签、loss 与梯度缩放；平台配置验证设备、通信和算子。用户选择模型、平台、DP / SP / TP / CP 度数，底层组合必须经过验证才能标记可用。

| 阶段 | 复用与适配 | 通过条件 |
| --- | --- | --- |
| 1. 全注意力正确性 | 官方 Ulysses + 当前 HF 模型 + 最小训练接口 | SP=1/2 对照 logits、损失、输入和参数梯度；含跨分片监督边界 |
| 2. MoTE 语义 | 保持 gate / MoTE 数学定义，适配全局容量、统计与随机性 | 打开实际裁剪、aux、QuantumGate 后仍通过对照 |
| 3. 完整训练状态 | 复用 ZeRO / 分布式检查点，保存 rank RNG 与采样游标 | 连续训练与中断恢复的下一批、损失和更新一致 |
| 4. 长序列资源 | Linux NVIDIA 两卡起步，结合重计算与分块词表损失 | 4K→8K→16K→32K，记录峰值显存、tokens/s、通信和数值稳定性 |
| 5. 扩展 | Gemma3 窗口、Qwen3.5 混合注意力、Megatron 与 DCU 后端 | 各模型 × 平台组合独立通过，不继承其他组合的认证 |

首个生产适配应优先 Qwen3 全注意力或 Llama3.1、Linux 同构 NVIDIA 多卡。Megatron 适配可复用 Bridge 已有模型导入导出，但需要额外定义 MoTE 替换层的模块规范和权重映射；原始模型的支持清单不包含我们自定义投影的验证。[Bridge Qwen3](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen3-8b.html)、[Bridge Llama3](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llama/llama3.html)、[Bridge Gemma3](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/gemma/gemma3.html)

## 下载代码核对与已完成整理

检查了 Downloads/fitmotn (1)/fitmotn、其 .orig 文件及同目录 9 个相关压缩包，未找到跨 GPU 序列并行实现。发现的收缩计划缓存、token 展平和矩阵乘法优化已在当前分支；既有多 GPU rollout / 张量并行推理不等于多 GPU 训练反向传播。

本轮完成的代码整理包括：共享文件哈希、规范 JSON 序列化、分位数统计、日志创建和显存采集；复用 rollout 分块与补齐函数；将 CE / KD 损失与延迟统计从大训练模块中分离；合并学习率计划元数据的重复计算；训练入口改为按需导入。旧的导入名称仍可用。隐藏 GPU 时的显存诊断异常和重复运行日志文件句柄未关闭问题一并修复。

整理后的相关回归为 239 passed、1 skipped。8 个迁出的损失与统计函数与基线 AST 一致，gate.py、quantum_gate.py、ADTN.py、model.py、sparse_mixt.py、patching.py 未修改。此次检查没有测得训练吞吐收益；效率改进主要来自减少重复实现、无关模块导入和资源泄漏。

活动实验 D:/AI/MOTE 保持 860426ce3df30f5c2a74a030a43f22e3a76247c6，工作树干净。运行时归档校验通过 117 个文件。当前修改与探测均位于独立发布工作树；本轮尚未提交或推送，也未启动实际模型的 16K～32K 训练。
