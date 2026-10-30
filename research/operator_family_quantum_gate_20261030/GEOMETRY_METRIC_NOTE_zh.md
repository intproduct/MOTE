# QuantumGate：从 Fisher 到选择边界与专家功能

数学说明与已完成机制核验｜2026-10-30｜MoTE 功能几何第二轮研究附件

## 1. 结论：设计对象应是功能相关的选择边界

Fisher 度量适合描述小扰动如何改变概率分布。QuantumGate 的设计还必须回答两个问题：这些概率变化是否跨过 top-k 选择边界，以及跨界后替换的专家会怎样改变模型输出。将状态几何、离散选择和专家功能分别建模，才能把量子态或线路的结构与实际语言建模行为联系起来。

对当前实幅度 RY/CZ 线路，固定幺正读出在编码角坐标中保持量子 Fisher 不变，同时能改变 Born 测量的 top-k 边界。这个性质给出了一个可检验的设计自由度：在相同状态局部可分辨性的条件下，改变哪些专家容易交换，以及交换发生在哪些输入附近。它的实际价值由语言质量、功能损伤和计算开销共同判定。

## 2. Softmax 与 Born：同一个概率度量，不同参数化

设 p 是 E 个专家的严格正概率。在概率单纯形上，类别分布的 Fisher 线元只由 p 和概率变化 dp 决定，生成概率的机制可以是 softmax，也可以是 Born 测量。

$$
ds^2=\sum_{e=1}^{E}\frac{(dp_e)^2}{p_e},\qquad \sum_e dp_e=0.
$$

采用 softmax logits z 作为坐标时，其拉回度量为下式。logits 整体平移对应零方向。

$$
F_z=\operatorname{diag}(p)-pp^{\mathsf T}.
$$

采用线路参数或输入编码角作为坐标时，需要经过相应的概率 Jacobian 拉回。因此，“输出概率相同”意味着同一点上的类别分布 Fisher 相同；“同样维数的可训练参数”或“同样使用概率路由”并不足以保证参数度量、可表达分布族和优化行为相同。

当前角编码使用四个角产生十六个概率。固定线路下，这是一族至多四维的概率流形。softmax 对全部独立 logits 的参数化覆盖十五维概率单纯形；低秩 MLP 又有自身的输入瓶颈。这些约束需要按实际架构比较。概率下限、top-k 截断和重新归一化也分别改变映射。

## 3. 当前实线路中可以严格得到什么

设 ψ(θ) 是归一化实状态，θ 为进入固定已训练线路之前的四个 RY 编码角。计算基测量给出 pₑ=ψₑ²。在非零幅度处，经典 Fisher 与纯态量子 Fisher 相等：归一化条件使 ψᵀdψ=0，实振幅没有被计算基测量丢失的相对复相位方向。

$$
F_{\mathrm{meas}}=\sum_e\frac{\partial_\theta p_e\,\partial_\theta p_e^{\mathsf T}}{p_e}=4J_\psi^{\mathsf T}J_\psi=F_Q.
$$

四个独立 RY 编码的 QFI 为单位矩阵。固定在编码之后的实幺正线路保持内积，所以当前两类线路在这组角坐标下均满足 FQ=I。这里固定的是读出参数；若把编码器权重、线路旋转参数全部作为联合坐标，需要重新计算其 Jacobian。一般复线路的测量 Fisher 与 QFI 关系也要单独分析。

$$
\left\langle\psi(\theta)\middle|\psi(\theta+\delta)\right\rangle=\prod_{j=1}^{4}\cos\frac{\delta_j}{2}.
$$

本轮在 128 个实线路点上核验，QFI 与单位矩阵、测量 Fisher 与 QFI 的最大绝对差均低于 10⁻¹⁵。加入概率下限后，近零概率方向的测量 Fisher 会缩小。无下限的恒等式和实际带下限的路由器应分别报告。

## 4. 从状态距离到边界：充分条件与局部预测

令 m=log(pᵢ/pⱼ) 表示选中专家与候选专家的分数间隔。沿小扰动的变化取决于读出和方向；Fisher 为该变化给出上界。对上述全支持 Born 分布，有：

$$
|dm|\leq\sqrt{p_i^{-1}+p_j^{-1}}\sqrt{d\theta^{\mathsf T}F_Qd\theta}.
$$

这是对归一化状态切空间应用 Cauchy-Schwarz 得到的局部界。对于任意两个归一化纯态，令 D 为它们的迹距离，则所有专家概率差的变化均不超过 2D。因此原始第四、第五名的概率间隔大于 2D，是 top-4 集合不变的充分条件。

$$
D=\sqrt{1-|\langle\psi|\phi\rangle|^2},\qquad p_{(4)}-p_{(5)}>2D\;\Longrightarrow\;S_4(\psi)=S_4(\phi).
$$

128 个受控状态扰动通过这两项数值核验；其中 72 个满足充分条件的案例均保持 top-4。充分条件偏保守。实际输入的局部预测使用 m(x+δx)≈m(x)+∇ₓm·δx，编码器 Jacobian 已包含在 ∇ₓm 中。

## 5. 专家功能应通过下游输出度量进入设计

令 V(x) 的每一列为一个专家的输出，w 为路由混合权重。最后一个 MoTE 模块的局部隐状态变化为 Δh=VΔw。设 J 为从该隐状态到语言模型 logits 的 Jacobian，pLM 为原始 next-token 分布。局部输出 KL 的二次形式为：

$$
G_{\mathrm{func}}(x)=V^{\mathsf T}J^{\mathsf T}\left[\operatorname{diag}(p_{\mathrm{LM}})-p_{\mathrm{LM}}p_{\mathrm{LM}}^{\mathsf T}\right]JV.
$$

$$
\operatorname{KL}(p_{\mathrm{LM}}\Vert q_{\mathrm{LM}})\approx\tfrac12\Delta w^{\mathsf T}G_{\mathrm{func}}(x)\Delta w.
$$

该度量同时包含专家输出、最终 LayerNorm 和词表投影。同一替换权重下的两个专家可以在隐藏空间相距较远，却沿下游不敏感方向变化；反过来也可能存在隐藏距离较小、预测变化较大的替换。专家输出欧氏距离、gate 自身概率的 Fisher 和语言模型输出 KL 的 Fisher 对应不同对象。

在选中集合不变的区域，可把 Gfunc 再沿 w 对线路参数的 Jacobian 拉回。跨越 top-k 边界时，离散权重出现跳变，应直接测量有限替换，或明确检验其局部 KL 近似精度。当前实验使用固定替换质量的全部 4×12 个候选；常规 sparse 重加权属于另一个需要验证的干预协议。

独立数学核验中，拉回二次型与解析 head JVP 相差约 1.1×10⁻¹⁴。扰动幅度由 10⁻² 缩至 10⁻⁴ 时，KL 二次近似相对误差从约 1.88% 缩至 0.0195%。这些随机小模型测试用于验证公式和实现。真实 checkpoint 的有限替换误差另由风险实验测量。

## 6. 训练中的联合适应与推理中的单向选择

训练可以观察专家输出与目标标签，通过反向传播同时调整 gate 和专家；推理 gate 必须在执行所选专家之前，只根据当前可用特征选择算子。两者的顺序差异提示：训练阶段可把昂贵的反事实功能信息作为教师或约束，最终应蒸馏进低成本 gate。是否能学会，需要用输入内可得信息的留出预测和实际干预来检验。

第一阶有符号标签损失不等于无标签功能风险。在模型自身预测分布下，标签 CE 的期望梯度为零；真实数据与模型的条件分布偏差决定可学习的有符号改进。下面的恒等式适用于取期望时固定模型参数的情形：

$$
\mathbb E_{y\sim p_{\mathrm{LM}}}[\nabla_h(-\log p_{\mathrm{LM}}(y\mid h))]=0.
$$

本轮分别学习有符号反事实效用、无标签专家输出特征和确定性的原 gate 分数，以区分目标可预测性、表示能力及优化过程。模型是否校准、是否存在不可约噪声下界，需要另外的证据。

## 7. 实际边界核验：主机制与三个例外

使用上一轮冻结的四种角编码路由器、三个种子和两个数据切片，共 24 组重放，每组 4,096 个 token。CPU FP32 计算中，仅对输入做 BF16 舍入往返，发生 295 次专家集合变化。固定第四、第五名的一阶预测正确识别了其中 293 次，另外两次分别为原排名 3→5 和 4→6 的交换。后续目标应覆盖所有相近的选中/未选中专家对。

FP16 舍入往返产生 30 次集合变化。一阶固定边界预测检出这 30 次，并多预测一次。该例的实际最终间隔与一阶预测均处于 10⁻⁷ 量级：改变 CPU batch 形状改变了一阶预测的符号；FP64 重放改变了实际最终间隔的符号。这三个特例已单独保留原始与 FP64 结果。这里统计的是相关模型重放中的事件，原生 GPU 低精度内核和跨推理引擎属于后续独立协议。

四比特独立乘积态还有一个结构约束：在非退化、无并列的情形下，第四、第五名的基态 Hamming 距离只能为 2 或 3。因此把单比特 Hamming 邻接直接当作全部危险交换边界，会遗漏该类路由器的真实 top-4 边界。100,000 个随机乘积分布均符合推导，冻结 checkpoint 的扩展重放也观察到 Hamming-1 比例为零。

## 8. 与已有研究的关系及下一步判据

NeurIPS 2022 的 On the Adversarial Robustness of Mixture of Experts 已把路由敏感性与高权重专家的函数差异联系起来，并分析平滑 MoE、检验 ImageNet 对抗鲁棒性。因此本研究应将具体贡献落在稀疏数值交换、量子读出结构和实际功能代价的联系，以及与经典控制的实测权衡。

下一步的候选是保持编码器和专家不变、只调整少量读出参数，并比较无干预、普通间隔约束和功能代价加权约束。启动条件是：真实内部开发集结果支持专家对信息超出单个专家能量；各方案采用相同监督、样本和选择规则；语言质量、负载及推理成本同时记录。正向学习对照和 KL/Fisher 风险测量完成后再据此作决定。

该说明记录已完成的数学与边界核验。第二轮全部学习实验、checkpoint 核验及阶段综合报告继续按原计划推进。

## 证据与参考文献

本地证据：outputs/operator_geometry_round2/theory/real_state_top4_v1；theory/metric_bridge_v1；boundary_gradient/input_sensitivity_v1；reviews/boundary_exceptions_v1。源代码、配置、逐样本数组及哈希均保存在对应目录。

Puigcerver et al., 2022. On the Adversarial Robustness of Mixture of Experts. https://papers.nips.cc/paper_files/paper/2022/file/3effb91593c4fb42b1da1528328eff49-Paper-Conference.pdf

Imai, Yang, Pezzè. Semi-classical geometric tensor in multiparameter quantum information. https://arxiv.org/html/2504.06812v3。该文区分测量 Fisher 与状态 QFI；本文的实线路恒等式和边界特化为独立推导与核验。
