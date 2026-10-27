# 第二轮：直接状态 gate 的能力与参数效率

本轮实现两个新结构，训练结果另存 `outputs/gate_benchmark_round2`。不修改或覆盖第一轮结果。

## 实现

- `residual_mpo` / `ResidualMPOGate`：LayerNorm → D→r → tanh → r→E，得到 q=log2(E) 个二值索引上的联合状态。执行两层 `a <- a + lambda*tanh(MPO(a))`，最后 E→E 线性读出 logits。MPO 核 `[chi_left,2_out,2_in,chi_right]`，开边界、无截断。每次 forward 先精确收缩成一个共享 E×E 算符，再批量作用，算符构建保留梯度且计入耗时。它是经典非线性 TN，不是幺正电路。
- `amplitude_born` / `AmplitudeReuploadBornGate`：LayerNorm → D→r → tanh；同一个 code 生成 E 个幅度和各层 q 个 RY 数据角。初态为 `normalize(1 + 0.5*tanh(V code))`，然后每层 RY 重上传和交替相邻配对的双向 CRY。使用实数 FP32 精确模拟，保留完整 E 个幅度；平方归一化、加 1e-6 平滑后返回 log(p)。所有制备后的演化均为正交/幺正；制备编码是经典神经网络，初始幅度限制为正，不声称任意态制备或量子加速。
- MPO 的 options 沿用已有 `gate_mps` 配置通道，字段为 `rank/depth/bond_dim`；原 `mps` 的 `sites/bond_dim` 语义不变。Born options 使用 `gate_quantum`。专家、TopK 与辅助损失共用。

## 公共训练协议

继续第一轮 FP32、同数据缓存、共享专家初始化、训练步数、优化器和评估规则。特别保留原 TopK 的 hard+soft 梯度表达式，本轮不同时更换 STE。MLP baseline 仍为 SiLU 两层网络；旧 compact_born 作为参照。

完整 gate 参数预算包含 LayerNorm 和公共 gain；逐投影以旧默认 MLP 的实际参数量为 full 目标，half 为其 50%，求最近整数 r。MPO 的核、残差系数、输入编码、E→E 读出以及 Born 的全部编码和电路角都计数。记录实际偏差，不把专家或整个模型减少量说成 gate 参数减少量。

## Pilot 与选择规则（在读取 pilot 结果前确定）

Fashion-MNIST、SST-2，seed=17，各 100 步。MLP full/half，旧 compact_born，加 MPO chi=1/2/4 × full/half，幅度 Born depth=1/2 × full/half，共 13 个变体、26 个 run。

先要求所有配置 loss/梯度有限、参数有更新、checkpoint/评估完整及共享状态/数据 hash 一致。保留所有失败及异常记录。

结构选择仅使用两个任务的内部 dev 最佳 score 的等权均值，最终评估 split 不用于选择：

1. 在 full/half 内分别选择 MPO chi，以及幅度 Born depth。
2. 距离最佳 dev 均值不超过 0.005 的配置视为 pilot 无法可靠区分，选择更小 chi / 更浅 depth；不把一个种子的短程排序当结论。
3. 保留两个家族和两个预算档进入正式评估。MLP 两档及旧 compact_born 进入同组正式矩阵。

候选家族进行了有限结构搜索，而 baseline 使用第一轮固定架构和学习率；报告必须披露搜索次数，不能声称已经比较了各类模型全面调参后的最佳能力。

## 正式评估

沿用 6 个任务、3 个种子、图像 1000 步与 BERT 600 步。7 个变体，共 126 项比较，新增训练先图像后 BERT。结构确定后冻结配置；真实时长由 pilot 更新。

18:44 的协议执行更新：本轮四个 MLP/原 Born pilot 与上一轮逐项精确复现，包括非计时训练轨迹、共享状态、最终分数和路由。因此复用上一轮同协议、已审计的 36 个正式 MLP/原 Born 对照，本轮新增训练 90 个 run。原始 manifest/result/curves/checkpoint 原样复制并逐文件 SHA256 校验，`reference.json` 明确来源；不把历史训练冒充本轮新训练。所有 126 个 checkpoint 将在当前代码中独立重评估、重新串行计时。其他条件、pilot 选型规则和正式矩阵不变。

使用原内部 dev 选 checkpoint；图像官方 test、GLUE 官方 validation 仅作最终评估。汇总 accuracy、MRPC F1/score、均值/样本标准差、对 full MLP 和 half MLP 的同 seed 配对差值。把 gate 参数量、总参数量、时间、吞吐、显存、路由指标与学习曲线共同报告。

若并行训练，吞吐只代表共享 GPU 时的端到端运行；效率结论另用串行短程测量，不能从并发日志给某种 gate 宣称固有效率优势。

三个种子只能提供有限证据。“相近”报告实际差值和离散程度，不以不显著作为等价证明，也不从少参数或高分推断量子优势/纠缠收益。
