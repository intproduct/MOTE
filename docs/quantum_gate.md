# QuantumGate：小块变分量子线路路由

本模块替换 MoTE/MiXT 的 router logits 特征计算。它不向 FFN 额外添加 LoRA 分支；TopK/SoftGate 继续负责 temperature、mask、capacity、drop、STE、辅助 loss 和使用率统计。默认 gate 仍为 linear，已有配置不受影响。`adtn_fixed` 没有动态 router，因此不会启用它。

## 配置

将下列 model 字段合并到正在使用的训练配置中；完整路径模板见 `fitmotn_config.quantum_gate.example.json`。

```json
{
  "gate_type": "topk",
  "gate_arch": "quantum",
  "gate_norm": "none",
  "gate_dropout": 0.0,
  "gate_output_init_std": 0.001,
  "zloss_coeff": 0.0,
  "gate_quantum": {
    "rank": 16,
    "depth": 2,
    "block_size": 2,
    "readout": "born",
    "angle_scale": 3.141592653589793,
    "probability_floor": 0.0001
  }
}
```

`motn` 与 `sparse_mixt` 均通过同一 factory 接入。量子比特数量来自实际传入 router 的 `num_experts`，不另行配置，也不盲目使用模型配置的 E；例如实际可路由块数可能受位置策略影响。global expert 不参与竞争，因而不额外分配 router qubit。

`gate_type="softmax"` 可使用全软路由；`gate_type="quantum"` 是 topk + quantum 的便捷别名。`gate_hidden_dim` 等 MLP 专用字段不控制本模块，使用 `gate_quantum.rank`。不引入 PennyLane/Qiskit/量子设备依赖。

## 线路与物理含义

输入 x 先经过可选 LayerNorm，再计算 a = π tanh(W_angle W_in x)，中间维数为 r。每个专家对应一个 qubit 和一个局部 Z 读出。各 qubit 从 |0> 开始，每层执行：

1. 数据重上传：RY(s[l,q] a[q] + θ[l,q])。
2. 块内最近邻 controlled-RY(γ[l,e])；相邻层反转控制方向。
3. 所有层结束后执行可训练的 RY(β[q])，然后读取 z[q] = <Z[q]>。

RY(θ) = exp(-i θY/2)，controlled-RY = |0><0|⊗I + |1><1|⊗RY(γ)。它们均严格幺正，后者可以产生纠缠；这些操作保持实振幅，因此只需实数状态张量。默认每两个 qubit 为一块，各块互不纠缠；最后不足一块时使用实际剩余 qubit。block_size 可选 1、2、4，depth 限制 1–4。

这里使用的是 VQC 作为可训练特征层，任务 loss 的反向传播构成混合经典/量子变分优化，不需要额外量子优化目标。幺正演化对态矢量仍然线性；输入到测量值的非线性来自角度编码、三角函数及 Born 读出。不存在语言专家必须服从该物理系统的先验。

数据重上传由 [Pérez-Salinas 等](https://arxiv.org/abs/1907.02085) 启发；编码方式和可用频率之间的关系见 [Schuld 等](https://arxiv.org/abs/2008.08605)。这些文献不保证当前受限实振幅线路具有通用表达能力，也不保证优于 MLP。[局部 cost 与梯度的研究](https://arxiv.org/abs/2001.00550) 提供设计动机，但其定理不能直接套用到整个语言模型 loss。

只用 RY 且关闭所有耦合时，各层旋转可合并，z[q] = cos(sum_l(s[l,q]a[q]+θ[l,q])+β[q])。这是一个明确的经典三角特征基线；重复编码本身在该无纠缠极限不会神奇地增加结构深度。初始 β=π/2、较小角度投影和小幅非零旋转，使路由接近均匀，并避免在 Z 极点初始化造成零梯度。

## 读出与既有 loss

- `born`：p[q]=(1-z[q])/2，平滑为 ε+(1−2ε)p[q]，返回 log p[q]。现有 softmax(log p/T) 等于 p[q]^(1/T)/sum_j p[j]^(1/T)。随后由原有 TopK/capacity 执行选取和重新归一化。
- `expectation`：返回 −z[q]，继续使用标准 temperature-softmax，用于区分测量概率归一化与特征本身的贡献。

这些 p 是不同 qubit 的边缘概率，和不必等于 1，不能称为直接对专家做联合 Born 采样。量子比特数等于专家数只是当前接口选择；也可设计 log2(E) qubit 的联合输出编码，但会改变读出、相关性和复杂度。

mask、balance loss、router z-loss 公式均沿用原 TopKGate。注意仓库现有 z-loss 是 mean(logsumexp(logits))，并非平方形式，对 logits 公共平移敏感；born 与普通 logits 的 z-loss 数值不直接可比。初始实验保留默认 zloss_coeff=0，若打开，应单独控制其影响。SoftGate 原本的 l_aux 行为也未改动。

## 成本与边界

令 N 为 token 数，d 为输入宽度，E 为实际专家数，r 为压缩维数，L 为深度，b 为块宽，G=ceil(E/b)。不含可选 LayerNorm 时可训练参数数为：

    dr + rE + 2LE + L(E−G) + E

其中绝大多数通常在经典输入压缩层。d=2048、E=16、r=16、L=2、b=2 时为 33,120 参数；linear 为 32,768，隐藏维数 16 且带 bias 的 MLP 为 33,056。QuantumGate 并不一定大幅增加参数，但会增加计算算子数。

编码成本 O(Ndr+NrE)，线路成本 O(NLE·2^b)。默认 E=64、b=2 时，每 token 的当前状态只存 32×4=128 个实振幅，不存 2^64 个振幅。反向传播还需保存各层中间量，不能把当前状态大小当作全部训练显存。默认不跨块纠缠，输入投影仍可混合所有输入特征。

线路内部始终 fp32（double 验证时 fp64），包括模型整体转成 bf16/fp16 后；梯度回传到原参数 dtype。这里实现的是 CPU/GPU 上精确模拟，没有实机训练、shot 噪声或量子加速声明。高数量小算子可能成为 GPU 瓶颈，应实测真实 batch；没有通过单元测试就推断训练收益。

基准脚本（仓库作为 fitmotn 安装后）：

```bash
python scripts/benchmark_quantum_gate.py --device cpu
python scripts/benchmark_quantum_gate.py --device cuda --tokens 8192 --experts 16 --iterations 50
```

脚本只测 router logits，不能代替全模型吞吐/显存评测。它比较 linear、相近参数量的 MLP 和 QuantumGate，报告参数量、forward 和 forward+backward 中位耗时，CUDA 时另报峰值 allocated memory。

2026-10-26 本机 CPU 单线程、PyTorch 2.14.1、float32、256 tokens、d=2048、E=16、r=16、L=2、b=2，预热 3 次后 20 次中位数：

| router | 参数 | forward / ms | forward+backward / ms |
| --- | ---: | ---: | ---: |
| linear | 32,768 | 0.062 | 0.134 |
| MLP | 33,056 | 0.071 | 0.168 |
| QuantumGate | 33,120 | 0.413 | 1.371 |

这是当前实现的局部成本检查，不能外推为 CUDA 全模型速度。QuantumGate 参数接近 MLP，但这次测量算子成本明显更高。

## 检验与建议实验

`tests/test_quantum_gate.py` 使用独立 NumPy 全寄存器 Kronecker 矩阵对照分块模拟，检验幺正范数、Bell 纠缠、无耦合解析退化、输入/参数梯度、两种读出、TopK capacity/mask/loss 等价、低精度、配置流转、SparseMiXT 反传及两个 backend 的 HF 模型保存重载。

下一步应固定相同 backbone、训练 token、更新次数、专家数、topk、初始化任务与评测协议，以至少三随机种子比较：linear、参数匹配 MLP、block_size=1 三角特征、block_size=2 VQC、block_size=4 VQC；再分别比较 depth=1/2 和 born/expectation。若要严格隔离纠缠，请固定其他权重，把 coupling 置零并冻结，同时从头重训；仅推理时置零还包含分布偏移。报告任务准确率、专家利用率/熵、drop rate、梯度、训练吞吐、峰值显存和总参数，而非只报告准确率。

旧 `Quantum_layer_Gate` 名称保留为新路由器 wrapper，现返回标准 `(probs, mask, aux)` 三元组。旧的 Gumbel/full-state 逻辑及 checkpoint 不自动转换；不要把旧类的二元组接口当作兼容承诺。切换现有 linear checkpoint 到 quantum 是架构变更，需重新初始化 router，不能当作同构训练恢复。

## 联合 Born 与经典 MPS 路由（mote-quantum）

新增 `gate_arch="compact_born"`：使用 `log2(E)` 个量子比特的联合 Z 基概率，E 必须为2至64的二次幂；配置为 `gate_quantum={"rank":16,"depth":2,"entangle":true}`。这与本文前述逐专家边缘概率的 `quantum` 架构是两个实现。

新增 `gate_arch="mps"`：配置 `gate_mps={"sites":16,"bond_dim":8}`，使用经典归一化矩阵乘积特征输出 E 个 logits。两者共用原来的 Top-k/softmax 外壳。`gate_learnable_gain=true` 可以给所有架构统一启用初值1的正值分数缩放，默认关闭以保持已有 checkpoint 兼容。

TITAN V 的 MLP、2/3层 BERT 三组对照、命令和公平性设计见 [完整实验安排](../experiments/gate_benchmark/README.md)。
