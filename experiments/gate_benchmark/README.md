# TITAN V：MLP / 联合 Born / MPS gate 实验安排

本实验比较同一套 MoTE 专家与 Top-k 路由下，三种**路由分数生成器**的短程监督学习表现。先运行较大的残差 MLP，再运行 2–3 层 BERT + MoTE-FFN。本机只完成 CPU 正确性检查和合成数据冒烟；本文不包含真实任务成绩，也不预判量子 gate 会胜出。

## 1. 克隆与环境

```bash
git clone --branch mote-quantum --single-branch https://github.com/intproduct/MOTE.git
cd MOTE
git branch --show-current
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r experiments/gate_benchmark/requirements-titan-v.txt
python -m pip install -e . --no-deps
python -m pip check
nvidia-smi
python -m fitmotn.experiments.gate_benchmark doctor
```

TITAN V 是 compute capability 7.0，环境选用 PyTorch 2.6.0 的 CUDA 11.8 wheel。请保留 `doctor` 输出与 `nvidia-smi`；`doctor` 实际执行 64 专家 gate 和 MoTE 专家收缩的反向传播，比仅检查 `torch.cuda.is_available()` 更有用。若驱动不能运行该 wheel，先解决驱动/环境，不要把失败归因于模型。无需安装 vLLM、FlashAttention、PennyLane、Qiskit 或 torchvision。

参考：[NVIDIA 架构表](https://developer.nvidia.com/cuda/gpus/legacy)、[PyTorch 官方历史安装命令](https://docs.pytorch.org/get-started/previous-versions/)。默认 **FP32**；不使用 BF16、TF32 或 `torch.compile`。FP16 是独立对照，需要先运行：

```bash
python -m fitmotn.experiments.gate_benchmark doctor --precision fp16
python -m fitmotn.experiments.gate_benchmark smoke --precision fp16
```

结构化 gate 内部保留 FP32/complex64；FP16 使用 GradScaler，并记录跳过的优化步数。若溢出频繁，以 FP32 结果为准。

## 2. 先做离线冒烟，然后跑 pilot

```bash
python -m pytest -q tests/test_structured_routers.py tests/test_gate_benchmark.py tests/test_quantum_gate.py
python -m fitmotn.experiments.gate_benchmark smoke
```

`smoke` 无需网络或数据集：使用随机初始化的微型模型和合成数据，对 MLP/BERT × 三种 gate 各执行 2 步，检查反向传播、checkpoint、dev 选模和最终评估。**这些分数没有任务研究意义。** 输出默认在临时目录；`--output outputs/smoke_01` 可指定新目录。

Pilot 保留正式网络宽度，选择 Fashion-MNIST 和 SST-2，每组 100 步、一个种子，检查真实数据与 GPU 性能：

```bash
python -m fitmotn.experiments.gate_benchmark prepare --config experiments/gate_benchmark/configs/pilot.json
python -m fitmotn.experiments.gate_benchmark run --config experiments/gate_benchmark/configs/pilot.json
python -m fitmotn.experiments.gate_benchmark summarize
```

`prepare` 需要访问 Hugging Face，下载数据、模型并缓存 token/tensor。`run` 只读取已准备的本地资源。模型与数据的下载 revision、样本索引、数据文件 SHA256 会被记录。Pilot 与正式实验的数据规模不同，会使用不同缓存。

先检查：loss 是否下降、梯度是否有限、三种 gate 的共享权重 hash 是否相同、专家实际数量是否正确、GPU 显存与吞吐是否可接受。初期不根据单个种子的一次成绩决定胜负。

## 3. 正式实验矩阵

| 阶段 | 主网络 | 任务 | 单次预算 | 种子 |
|---|---|---|---|---|
| A | 宽度 1024、4 个残差块的 MLP；每块含 1024×1024 dense mixer 和 MoTE 1024→2048→1024 FFN | MNIST、Fashion-MNIST、CIFAR-10 | 1000 步，batch 128，lr 1e-3 | 17 / 29 / 43 |
| B | 预训练小 BERT 截取前 3 层；hidden 256 / FFN 1024；每层两个 FFN Linear 换成 MoTE | SST-2、MRPC、RTE | 600 步，batch 16，lr 1e-4 | 17 / 29 / 43 |

每个任务分别跑 `mlp`、`compact_born`、`mps` gate，主矩阵共 **54 个训练 run**。阶段 A 有数百万共享 dense 参数，不是只训练一个小 gate 的玩具分类器；图像直接展平、归一化至 [-1,1]，不使用增强，其目的在于观察路由差异，不与 CNN 的 CIFAR-10 最佳成绩竞争。

BERT 来源：[google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4)。这是**截取四层预训练模型的前三层/前两层**，不是声称使用原生预训练的三层模型。保留注意力、残差、LayerNorm、pooler；分类头重新初始化，全部模型参数参与任务训练。正式 default 的 E=16、Top-k=4、k_in=4；hidden=256 时随机位置组合也足以支持 E=64。上/下投影分别使用实际输入维度的 router。

```bash
python -m fitmotn.experiments.gate_benchmark prepare --config experiments/gate_benchmark/configs/titan_v.json
python -m fitmotn.experiments.gate_benchmark run --config experiments/gate_benchmark/configs/titan_v.json --stage mlp
python -m fitmotn.experiments.gate_benchmark run --config experiments/gate_benchmark/configs/titan_v.json --stage bert
python -m fitmotn.experiments.gate_benchmark summarize
```

可以先跑一个任务、一个种子，后续同配置补齐，已完成 run 自动跳过：

```bash
python -m fitmotn.experiments.gate_benchmark run --config experiments/gate_benchmark/configs/titan_v.json --stage bert --tasks sst2 --seeds 17
```

`--seeds`、`--gates`、`--tasks`、`--stage` 是执行子集，**不改变实验分组 ID**；因此可依次补齐并做配对比较。它们必须属于配置中的计划。不要用两个进程同时写同一个 run 目录。命令行 `--device`、`--precision`、`--bert-layers`、`--experts`、`--topk`、`--budget` 改变实验条件，会生成独立分组。

不提供任意中途训练的 optimizer/RNG 恢复。进程中断后保留已有日志和最佳 checkpoint；重跑时会明确报出未完成目录，先将该目录改名移出结果树，再从相同种子重跑。不会悄悄覆盖或将半程结果当成完成。

## 4. 三种 gate 的确切定义

共同外壳：`LayerNorm → score generator → positive learnable gain → temperature → 原 TopKGate`。默认 temperature=1，gain 初值 1；三组均有该标量。Dropout=0、jitter=0、STE 相同、balance 系数 0.01、z-loss 系数 0。现有 Top-k、mask、capacity、负载损失实现共用，实验默认 `drop_tokens=false`，使 token 丢弃不成为第一阶段的额外变量。

- **MLP**：`d → r → SiLU → E`，两层带 bias 的线性层。
- **联合 Born**：`d → r → 2Lq` 无 bias 投影，角度为 `π tanh(a)`，`q=log2(E)`。每层在各比特施加 `RY(输入角+可训练偏置)`、`RZ(输入角+偏置)`，再对交替邻接配对施加两个相反控制方向的 CRY；末尾加可训练 RY。初态是 `|0…0⟩`，读出完整 Z 基联合概率，按大端 bitstring 的整数值映射专家。E=64 仅保存 64 个复振幅，默认 depth=2。相邻两比特的单元为 `CRY(q_i→q_j)` 后接 `CRY(q_j→q_i)`；下一层配对平移一格。RZ 的相位经后续旋转转化为可观测概率。
- **经典 MPS**：`u=tanh(Wx)`，16 个 site、键维 χ=8；`v_i=normalize(v_{i-1}+u_i v_{i-1}A_i/√16)`，最后 `W_out v_16` 输出 E 个分数。这是稳定化的矩阵乘积特征模型，不是幺正量子线路。

Born 原始概率为 `p_e=|ψ_e|²`；平滑后 `p̃_e=(p_e+ε)/(1+Eε)`，ε=1e-6；分数为 `gain·log(p̃_e)`。因此公共 softmax 给出 `p̃_e^(gain/temperature)` 的归一化结果。在 gain=temperature=1 时正好还原平滑联合概率。之后公共 Top-k 再选专家并归一化。

这与“一专家一个 Z 期望值”的旧 `gate_arch="quantum"` 不同；旧实现保留兼容，本实验使用 **`gate_arch="compact_born"`**。生产入口在 `gate.py`、`structured_routers.py`；独立探索原型仍在 `research/gate_design/`。

## 5. 公平性与数据划分

1. 默认 `budget="matched"`：以 MPS 参数量为目标，逐投影求最接近的整数 MLP hidden width 和 Born rank。不是强行使用完全相同 rank 后忽略 MPS 额外参数。整数取整会有小幅偏差，manifest 记录每层实际 router 参数、总参数和专家位置。另有 `budget="rank"` 对照：MLP hidden=Born rank=16，MPS sites=16，此时参数不完全相同。
2. 每个任务/种子先生成一份共享 backbone、专家、bias、分类头初态，再复制到三种 gate；记录并验证非 router state 的 SHA256。三组训练重新设置相同 RNG、同一个 DataLoader 种子，使用相同 batch 顺序、步数、优化器和学习率日程。
3. BERT FFN 不能在多专家结构下一般性地精确恢复原 dense 矩阵。默认先对每层做 40 步 **公共专家初始化**：绕过 gate，以所有专家的均匀平均拟合原 dense 权重在高斯输入上的输出，只优化专家。它近似最小化线性算子的 Frobenius 误差，不使用分类标签。用独立高斯 probe 记录拟合前后相对 MSE，然后复制同一份专家给全部 gate。它不是量子 gate 专有训练，也不能保证恢复原 BERT 精度。
4. BERT 的 padding token 在进入 MoTE 前就被滤除，不进入 Top-k、负载损失或路由熵统计；有效输出再散射回原序列位置。
5. 数据 seed 固定为 20261027。先从官方训练集按标签分层切出内部 dev（15%，最多1000），剩余部分取最多10000条训练数据。图像官方 test 最多取2000条作为最终评估；GLUE 官方 validation 用作最终评估，**不参与选 checkpoint**。GLUE 真正 test 无标签，本实验不声称提交其隐藏测试榜单。任务来源：[GLUE 数据集](https://huggingface.co/datasets/nyu-mll/glue)。
6. 按内部 dev 最优 score 选 checkpoint，仅随后评估最终 split。MRPC score=(accuracy+F1)/2，其余任务为 accuracy；都记录 CE。`initial_dev.json` 是监督训练前随机分类头的诊断，不叫 zero-shot 成绩。

本实验能检验“在固定初始化与短程预算下，哪一种路由参数化更好训练”，不能独自证明量子优势、纠缠是唯一收益来源，或证明某 gate 在最佳调参后一定占优。

## 6. 输出与应当阅读的指标

默认目录 `outputs/gate_benchmark/<config_hash>/<stage>_<task>_<gate>_seed<seed>/`：

- `manifest.json`：完整配置、版本、Git commit、数据 revision/索引/hash、参数量、共享初态 hash、专家位置、公共初始化误差。
- `initial_dev.json`：初始 dev 指标与路由分布。
- `curves.jsonl`：train CE、aux、梯度范数、学习率、耗时，以及 dev 曲线。`routing_last_batch` 明确仅代表最后一个训练 batch。
- `best.pt`：按内部 dev 选出的模型 state，不是 optimizer resume checkpoint。
- `result.json`：完成标志、最佳步数、最终任务指标、全最终 split 的 token 加权路由统计、训练吞吐、CUDA 峰值显存。
- 汇总根目录的 `runs.csv`、`summary.json`：按实验配置分别汇总 mean/std、相同 seed 对 MLP 的配对差值；同时报告未完成和未运行的计划组合。

重点看：最终 score 与跨 seed 方差、同预算 dev 学习曲线、score 相对训练秒数、router 参数量、`H_tok`、`H_dataset`、load CV、unused experts。两种熵分别描述单 token 路由的不确定性和整个 split 的专家使用广度：两者不能混为一谈。全数据路由统计对有效 token 加权，不是简单平均不同长度 batch 的 entropy。

训练吞吐包括数据取 batch、搬运、前向、反向和 optimizer step，不包括 dev 评估和公共初始化；`wall_seconds` 包含训练过程中的 dev、checkpoint、最终评估，但不含模型加载、公共初始化和初始 dev。GPU 峰值覆盖本次训练与评估。不要将 CPU 原型计时当成 TITAN V 的实测。

主矩阵是同优化步数对照，时间效率从曲线和吞吐另行比较。不要仅选一组的最佳 seed 与另一组平均值比较；只有一个 seed 时标准差输出 null，不伪装为0。

## 7. 第二轮：规模与机制消融

先完成 pilot、主矩阵，再按结果选择，不建议一开始全排列。

**2 层 BERT 对照**（相同数据缓存、独立实验分组）：

```bash
python -m fitmotn.experiments.gate_benchmark run --config experiments/gate_benchmark/configs/titan_v.json --stage bert --bert-layers 2
```

**64 专家、6 比特**（先一个任务、一个种子；Top-k=8）：

```bash
python -m fitmotn.experiments.gate_benchmark run --config experiments/gate_benchmark/configs/titan_v.json --stage bert --tasks sst2 --seeds 17 --experts 64 --topk 8
```

**同 rank 对照**：添加 `--budget rank`，三种 gate 都重跑，和参数匹配分组分开报告。

**纠缠消融**：复制配置后加入 `"quantum_entangle": false`，同一配置下仍运行三组；其 Born gate 不施加 CRY，保留编码、局部旋转和联合概率读出。这样才有依据讨论两比特耦合是否有贡献。`depth` 可设1/2/3/4，MPS 的 `sites`/`bond_dim` 可设8/4等较小组合；每次改变都形成独立分组。

如短程结果很接近，先给三种 gate 相同大小的学习率搜索预算（例如主值×0.3/1/3，内部 dev 选超参），再做固定超参的多种子验证。不要只给 Born 单独大规模调参后把差值归因于量子结构。若公共 FFN 近似误差过大，应先统一增加 warm_init_steps 或做保留 dense FFN 的额外基线，以分辨 FFN 替换损失和路由收益。

## 8. 显存、耗时及异常处理

- 不预估未经实测的 TITAN V 完成时长。用 pilot 中每种 gate 的训练秒/步，乘正式步数与 run 数，再加公共初始化、评估、下载开销。
- 显存不足：复制配置，统一将 BERT batch 16 改为8或将 max_length 128改为64，**三种 gate 全部使用同一新配置**。max_length 变化需重新 prepare。E=64 比 E=16 的专家分派开销更高，即使量子态只有64维。
- 训练中断：保留失败目录供诊断，移出汇总树后重跑；不会自动覆盖。磁盘空间不足时，可在结果完整汇总并另行归档后自行删除 best.pt。
- 若默认严格确定性在特定 CUDA 算子上报错，先保留错误和 doctor 输出；统一在新配置设置 `"deterministic": false` 后重跑所有对照，并明确记录。不要只对其中一个 gate 放宽条件。
- 如果 Hugging Face 网络访问失败，先使 `prepare` 完整成功；不能用随机模型悄悄替代预训练 BERT，或用合成数据代替真实任务。

## 9. 本次验证的边界

CPU 测试覆盖：独立 NumPy 量子线路参考、联合概率归一化、梯度检查、64 输出、MPS 显式矩阵乘积参考、共享路由/辅助损失接口、HF 保存重载、2/3层 BERT padding 隔离、公共专家初始化、参数预算匹配，以及六组合成任务的完整短程训练流程。

真实数据的完整训练、TITAN V CUDA/FP16 行为及最终任务成绩，需在目标机器执行以上命令后确认。本分支的目标是交付可复现实验，而不是把冒烟成绩当作实验结论。

实际本地验收：Python 3.12 / PyTorch 2.6.0 / Transformers 4.57.6 / Datasets 3.6.0 的 CPU 环境，相关回归 **128 passed**；包含上述六组短程冒烟与真实宽度 E=64 MoTE 反向传播。另一个 PyTorch 2.14.1 CPU 环境也通过同一相关套件。Mac 的 NumPy 2.2.6 在部分旧稠密参考测试的矩阵乘法上产生 RuntimeWarning，但参考值与生产输出的有限值/数值一致性断言均通过。全库 `pytest tests` 另有已有测试模块的导入冲突（测试替换 MOTE.config）及缺少可选 vLLM 的收集失败；本次不将其标记为全库通过。

复现相关回归的命令：

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_structured_routers.py tests/test_gate_benchmark.py tests/test_quantum_gate.py tests/test_sparse_mixt.py tests/test_gate_router_arch.py tests/test_config_resume.py tests/test_repo_hygiene.py research/gate_design/test_candidates.py
```
