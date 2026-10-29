// Self-contained paper-style edition. No network packages or external bibliography.
// Compile: typst compile QuantumGate_review_paper.typ QuantumGate_review_paper.pdf
// Tested with Typst 0.11.1. Recommended fonts: Noto Serif SC / Noto Sans SC.
#set document(title: "量子机器学习优势向专家路由的迁移：QuantumGate 的信息几何、持续学习与验证框架", date: datetime(year: 2026, month: 10, day: 29))
#set page(paper: "a4", margin: (top: 23mm, bottom: 22mm, left: 24mm, right: 24mm), numbering: "1", number-align: center,
  header: align(right, text(size: 8pt, fill: rgb("666666"))[QuantumGate · 文献综述与机制分析]))
#set text(font: ("Noto Serif SC", "Cambria"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.72em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(font: ("Noto Sans SC", "Cambria"), size: 14pt, weight: "bold")
#show heading.where(level: 2): set text(font: ("Noto Sans SC", "Cambria"), size: 11.5pt, weight: "bold")
#set math.equation(numbering: "(1)")
#show math.equation: set text(font: "New Computer Modern Math")
#show raw: set text(font: ("Consolas", "Noto Sans SC"), size: 9pt)
#show link: set text(fill: rgb("1c4264"))
#set figure(supplement: [图])
#let c(n) = super(link(label("ref-" + str(n)), text("[" + str(n) + "]")))
#let reference(n, body) = block(above: 0.1em, below: 0.35em)[
  #set par(leading: 0.4em, justify: false)
  #text(size: 9pt)[#strong[#n.] #body]
]
#let finding(title, body) = block(inset: (left: 10pt, right: 10pt, top: 8pt, bottom: 8pt), stroke: (left: 1.2pt + rgb("555555")), fill: rgb("f6f6f6"))[
  *#title* #body
]

#align(center)[
  #text(font: "Noto Sans SC", size: 20pt, weight: "bold")[量子机器学习优势向专家路由的迁移]
  #v(0.45em)
  #text(size: 13pt)[QuantumGate 的信息几何、持续学习与验证框架]
  #v(0.6em)
  #text(size: 10pt)[研究综述与机制分析 · 2026 年 10 月 29 日]
]
#v(0.7em)
#block(inset: (x: 12pt, y: 7pt))[
  #align(center)[*摘要*]
  量子机器学习在持续学习、泛化、优化几何及噪声数据处理方面提供了多种潜在优势，但这些结论能否迁移到经典硬件上模拟的专家路由器，需要分别检验其假设与资源条件。本文结合一手文献与 MoTE 的 QuantumGate 实现，区分输出概率空间、输入相关函数族和参数优化几何三个层次。分析表明，固定测量的 Born 概率与 softmax 均位于普通概率单纯形，潜在差异主要来自态编码、受约束交互及其诱导的训练几何。对于局部非零的归一化实振幅，计算基读出的经典 Fisher 信息等于纯态量子 Fisher 信息；对于复振幅，部分相位方向则可能不被当前读出观测。结合量子持续学习、数据污染与泛化研究，本文提出参数化消融、可塑性与旧任务保持的联合评估，以及分层稳定性测试。现有证据支持开展结构与任务匹配的机制研究，尚不足以证明 QuantumGate 的普遍性能优势。本文不报告新增模型训练，数值核验仅用于检查数学关系。

  *关键词：* 量子机器学习；混合专家；Born 规则；Fisher 信息；持续学习；路由稳定性
]

= 引言

混合专家模型将输入映射为专家选择及组合权重，路由器因此同时影响条件计算、专家分工和训练动态。将量子态编码与 Born 读出用于路由，提出了一个自然的问题：Hilbert 空间的几何结构是否能提供优于传统概率映射的归纳偏置？这一问题具有研究价值，但“量子模型的优势”并非单一命题。表达能力、样本效率、抗遗忘、数值稳定性和计算复杂度分别依赖不同假设，不能由其中一项直接推出其余各项。

量子持续学习已有理论与硬件实验，参数保护能够缓解所研究量子模型的灾难性遗忘。#c(1)#c(2) 算法稳定性与信息几何研究则提供了分析泛化和优化行为的工具。#c(5)#c(10)#c(11) 与此同时，系统基准显示，在若干经典分类任务中，强经典基线仍可能占优，纠缠也不必然改善结果。#c(19) 因此，研究重点应从模型名称转向可检验的机制。

本文围绕三个问题展开：第一，Born 读出与 softmax 的区别究竟发生在哪个层次；第二，文献中的持续学习和稳定性结果具备哪些迁移条件；第三，如何通过公平对照判断 QuantumGate 的收益是否需要量子线路结构。本文采用叙述性综述与实现分析，不构成穷尽所有文献的系统综述，也不将标准数学关系主张为新的理论发现。

= 研究范围与实现对象

== 证据类型与版本边界

本文将证据分为三类：公开论文给出的理论或实验结果；由概率映射与态几何得到的数学分析；结合当前路由实现提出的工程假设。除明确引用的内部实验记录外，本文没有新增训练结果。文献检索截止日期为 2026 年 10 月 29 日，预印本的结果按其公开版本理解，不能等同于已完成独立证明复核。

实现分析首先对应仓库 `intproduct/MOTE` 的固定提交 `c7d6690`，随后补充已保存于 `c726b4e` 的共享 Born 路由实现。实验主工作区为 `D:/files/MOTE`，目标分支为 `mote-quantum`。该版本划分用于避免把原始重上传线路的性质直接套到后续简化实现。

== 三类 Born 路由与经典结构对照

分块 `quantum` gate 使用 RY/CRY 实数演化，读取不同 qubit 的边缘概率，再由共享模块归一化为专家权重。由于边缘概率并非互斥完备事件，其输出不应直接称为一次联合 Born 测量的类别分布。

`compact_born` 以 $q = log_2 E$ 个 qubit 表示 $E$ 个专家，通过复振幅、相位旋转和数据重上传得到联合计算基概率。这里 $E$ 为 2 的幂，生产实现支持 2 至 64 个专家。`amplitude_born` 则先构造归一化实振幅，再施加实数旋转与输入相关的重上传。后续 `SharedBornRouter` 使用输入无关的共享正交矩阵，保留非线性联合振幅编码，并直接使用概率作为软权重。

MPS 和 residual MPO 提供经典张量结构对照，不能仅因其量子启发背景就将效果归因于量子硬件。上述 gate 都按 token 重新构造状态，没有跨 token 持续保存的量子记忆。其小规模态向量由经典 CPU/GPU 精确模拟，首先需要比较同等资源预算下的表现，而非援引量子硬件的复杂度分离。

= 概率表达与状态几何

== 固定读出的输出空间

设 $x in RR^d$ 为输入表征，$E$ 为专家数，$psi_theta(x) in CC^E$ 为归一化纯态。计算基读出的概率为
$ p_i(x; theta) = abs(psi_(theta,i)(x))^2, quad sum_(i=1)^E p_i = 1. $ <eq-born>
因此 $p$ 位于通常的概率单纯形
$ Delta^(E-1) = {p in RR^E : p_i >= 0, sum_i p_i = 1}. $ <eq-simplex>
任意 $p in Delta^(E-1)$ 都可由 $psi_i = sqrt(p_i)$ 实现；对单纯形内部任意严格正的概率向量，又有
$ p = op("softmax")(log p). $ <eq-softmax>
这里的态构造允许任意归一化态，并不保证某个受限线路能达到全部分布。softmax 的有限 logits 不能产生精确零概率，但当前实现采用概率平滑，因此这一边界差异并非主要的工程收益来源。

#finding([命题 1：固定测量不扩大类别概率的输出空间。], [完整态的 Born 读出与 softmax 都可以覆盖普通单纯形的内部。可能产生优势的是受限函数族、参数化及优化路径，而不是超出该单纯形的“非经典类别概率”。])

对非零实振幅 $a$，平方归一化可以改写为
$ p_i = a_i^2 / (sum_j a_j^2) = op("softmax")(2 log abs(a))_i. $ <eq-square>
这一等式是点态概率等价，并不意味着任意两个网络具有相同函数族或训练轨迹。因此平方振幅头是必要的经典对照，但不能据此直接断言所有量子线路都冗余。

== 平滑、温度与实际权重

设平滑常数为 $epsilon > 0$，共享模块的正增益与温度分别为 $g$ 和 $T$，则在 top-k 之前有
$ tilde(p)_i = (p_i + epsilon)/(1 + E epsilon), quad
  w_i = (tilde(p)_i^(g/T))/(sum_j tilde(p)_j^(g/T)). $ <eq-temperature>
这说明原始 Born 概率并不总是最终软权重。若不同方法使用不同的增益、温度或初始化熵，概率锐度的变化就可能被误认为线路结构的作用。共享 Born 直接概率读出的等价性，也需要核对相应的增益和温度条件。

#figure(
  block(width: 100%, inset: 10pt, stroke: 0.4pt + rgb("888888"))[
    #set math.equation(numbering: none)
    #align(center)[$ x arrow psi_theta(x) arrow p_i = abs(psi_i)^2 arrow w_i arrow op("top-k")(w) $]
  ],
  caption: [从输入到实际专家选择的分析层次。态距离、概率距离与离散选择的稳定性是不同量，不能互相替代。],
) <fig-pipeline>

== Fubini–Study 度量与可观测信息

将纯态振幅局部写为 $psi_i = sqrt(p_i) exp(upright("i") phi_i)$，其中全局相位无物理意义。相关状态几何应定义在投影态空间。Fubini–Study 线元可以分解为
$ dif s_("FS")^2 = 1/4 sum_i (dif p_i)^2/p_i
  + sum_i p_i (dif phi_i)^2 - (sum_i p_i dif phi_i)^2. $ <eq-fs>
第一项是类别概率 Fisher 几何的四分之一；相位协方差项刻画了固定计算基读出可能无法分辨的方向。经典测量 Fisher 信息与量子 Fisher 信息的关系已有标准统计几何理论。#c(12)#c(13) 对固定测量，一般有 $F^("C") ≼ F^("Q")$；不能假定一次固定测量能够利用全部态信息。

对于路由任务，这一区别有直接含义：保护量子态变化可能同时保护当前输出不可见的方向。若目标是保持专家选择，输出概率的 KL 或 Fisher 度量更直接；是否需要保护额外相位信息，应取决于后续操作能否使其可观测。

== 实振幅下的 Fisher 等价

考虑关于参数平滑变化的归一化实态 $psi(theta)$，并假设分析邻域内读出振幅均非零。由 $p_i = psi_i^2$ 和归一化条件可得
$ partial_a p_i = 2 psi_i partial_a psi_i, quad
  sum_i psi_i partial_a psi_i = 0. $ <eq-tangent>
输出概率的经典 Fisher 矩阵因此为
$ F^("C")_(a b) = sum_i ((partial_a p_i)(partial_b p_i))/p_i
  = 4 sum_i (partial_a psi_i)(partial_b psi_i)
  = F^("Q")_(a b). $ <eq-fisher>
最后一个等号使用纯态量子 Fisher 的表达式及归一化切向约束。该结论是对标准公式的应用，不是新增的普适优势定理。

#finding([推论：实振幅路由的自然梯度具有经典对应。], [对于上述条件成立的实振幅联合读出，态空间度量与输出概率度量一致。因此 `amplitude_born` 和共享实振幅 gate 的几何优化，应先与相应经典 Fisher 方法比较。概率平滑、温度/增益、边缘化及硬 top-k 之后，不能不加证明地继续使用该等式。])

复态反例说明了条件的重要性。令
$ psi(theta) = 1/sqrt(2) vec(1, exp(upright("i") theta)), quad
  F^("C") = 0, quad F^("Q") = 1. $ <eq-phase>
计算基概率恒为 $(1/2, 1/2)$，但态仍沿相位方向变化；若之后加入混合操作，相位可能再次影响读出。于是“状态变化大”与“当前路由变化大”不是同义命题。

== 范数保持与离散路由的边界

同一个固定酉矩阵保持输入态之间的距离。但重上传线路使用输入相关的算子，编码器的 Jacobian 也可能放大扰动，因此这一局部性质不自动成为端到端 Lipschitz 保证。共享 Born 的算子固定于输入，但其非线性编码、概率后处理与离散选择仍需单独分析。

#block(breakable: false)[
  设 logits 按降序排列，第 $k$ 与第 $k+1$ 个分量间隔为 $m_k = z_((k)) - z_((k+1)) > 0$。一个充分的选择集合不变条件为
  $ norm(delta z)_infinity < m_k / 2. $ <eq-margin>
]
当间隔接近零时，极小概率变化也能翻转专家选择。量子分类器的对抗脆弱性研究同样提示，不应将范数保持解释为普遍鲁棒性。#c(22)#c(23)

= 文献中的潜在优势及迁移条件

== 持续学习：结构与保护机制需要分离

灾难性遗忘指学习新任务后旧任务能力下降。量子持续学习工作利用参数重要性或局部几何信息保护旧任务；超导硬件实验也展示了所选任务上的保持效果。#c(1)#c(2) 这些结果同时说明量子模型自身会遗忘，不能据其名称推断天然的知识保持能力。

经典 EWC 方法以旧任务的重要参数为中心施加惩罚：#c(3)
$ cal(L)_("total") = cal(L)_("new") + lambda/2 sum_j
  F_j (theta_j - theta_j^star)^2. $ <eq-ewc>
其中 $F_j$ 是旧任务相关的重要性估计，不应不经核对就称为完整量子 Fisher。迁移到 QuantumGate 时，最低限度应比较“经典/Born 结构”与“无保护/同一种保护”的双因素设计。量子组独享正则化，会混淆结构收益与训练策略收益。

长期序列依赖则是另一个问题。酉循环网络通过受约束的状态递归缓解梯度问题，是经典模型也能利用的机制。#c(6) 当前 QuantumGate 没有跨 token 持久状态，因此不能直接继承该类长期记忆结论。

== 数据污染与机器遗忘

机器遗忘（unlearning）关注主动移除指定数据的训练影响，与保留旧任务的目标不同。近期研究在 MNIST 二分类与量子相分类等设置中报告了量子模型对污染数据的韧性及近似遗忘能力。#c(4) 这提供了值得复现的实验线索，但不等于大语言模型路由中的普适结论，也不等于严格的数据删除保证。

解释相关机制时，需要控制经典非线性基线。对于标量 logit $z(theta)$、标签 $y$ 及预测 $p = op("sigmoid")(z)$，二元交叉熵满足
$ (dif^2 cal(L))/(dif theta^2) = p(1-p) (z')^2 + (p-y) z''. $ <eq-hessian>
第二项来自非线性 logit 的曲率，经典正弦或余弦网络也具有它。针对该论文预印本附录的简化线性/余弦比较，这提示我们应加入周期网络，而非只以线性或普通 ReLU 模型作为唯一参照。该分析不替代论文整体实验，也不声称其所有结果都由这一项解释。

== 泛化、有效维度与优化几何

QNN 的算法稳定性研究给出了与可训练门数、步长、迭代过程、可观测量范数和损失假设相关的泛化分析。#c(5) 算法稳定性主要指训练数据变化对模型输出或损失的影响，不能与有限梯度、输入鲁棒性或硬件数值一致性混用。包含经典编码器、对数概率及硬 top-k 的混合路由需要重新核对理论假设。

少样本泛化界和有效维度研究提示，模型的可训练自由度及其利用方式可能比名义参数量更重要。#c(7)#c(9) 数据相关量子 Fisher 又进一步强调线路、样本与对称性之间的关系。#c(10) 但更高有效维度不总是更好的泛化，忽略经典编码器参数也会造成不公平的容量比较。QNN 对随机标签的拟合现象表明，量子结构并不自动阻止过拟合。#c(8)

量子自然梯度为沿态空间度量进行优化提供了方法。#c(11) 在小态空间上可以尝试精确或块近似计算，但需要与经典输出 Fisher、KL 信赖域及调优后的普通优化器比较。达到同一指标所需步数较少，不代表墙钟时间更短。

== 编码、频谱与结构先验

数据重上传使输入多次进入线路；编码分析表明，可获得的频率结构受编码算子与重复方式影响。#c(14)#c(15) 这支持以“任务与特征频谱是否匹配”解释收益，而非直接以 Hilbert 空间维数解释收益。当前受限线路的表达能力，也不能由其他架构的普适性结果直接保证。

对称性设计和经典张量网络同样说明，结构先验可以在合适任务中发挥作用。#c(16)#c(17) 但专家编号并没有天然的量子比特语义，二进制相邻或 Hamming 距离较小的专家也不一定功能相似。因此专家编号置换、耦合图改变及低秩交互对照，是判断结构收益是否稳健的必要步骤。

== 计算优势、负面结果与研究边界

量子学习的复杂度优势依赖访问模型、资源计量和任务结构。已有研究强调数据对于经典模型的帮助，以及特征几何与学习任务的匹配。#c(24) 利用量子实验访问条件的样本优势，不能直接迁移到经典 hidden states 的小型路由模拟。#c(25) 2026 年关于海量经典数据处理的预印本又提出不同的机器规模优势；本文仅核验其摘要与版本信息，未独立复核完整证明。#c(26)

广泛的小型分类基准、barren plateau 与量子核集中现象，分别提示基线、优化和特征可区分性的风险。#c(19)#c(20)#c(21) 这些结果不能否定所有 QML，但足以排除“更大态空间、更深线路、更多纠缠必然更好”的默认假设。已有量子混合专家工作也要求谨慎界定本项目的新颖性。#c(18)

= 对当前 QuantumGate 的研究假设

结合上述分析，可将后续研究归结为三个可证伪假设。

*H1：编码与受约束交互的任务匹配。* 在相同输入编码、输出锐度与资源预算下，某些 Born 结构可以形成更合适的输入到专家概率函数。如果平方振幅头或周期经典网络复现收益，则证据支持该参数化或频谱机制，而不支持线路本身不可替代。

*H2：可塑性与旧任务保持的改进。* 在相同保护策略和旧数据预算下，Born gate 可能在保持新任务学习能力的同时，减少旧任务退化。只有与经典基线的相同正则化对照后仍存在稳定差异，才构成结构相关抗遗忘证据。

*H3：输出相关几何改善优化或扰动响应。* 路由 Fisher 或 KL 约束可能减少无效的参数更新，但更低路由漂移必须同时对应可接受的任务质量与训练效率。仅因模型更新更小而“更稳定”，不能认定为优势。

最新内部路由稳定性实验已完成 18 项训练及对应核验。其报告指出，原 Born 与共享 Born 均未达到预注册的质量、延迟与切换率组合目标；共享 Born 的工程收益主要体现为相对原 Born 降低推理延迟，关键的无 U、无 CRY 对照仍缺失。该结果属于内部小规模实验，不是公开文献结论。它要求后续研究衔接现有负面与混合证据，不能从零假定候选结构已经占优。

= 验证框架

== 第一阶段：参数化与结构消融

首先冻结 backbone 与 experts，使用无泄漏的真实输入表征筛选机制。对照包含强 MLP/residual MLP、平方振幅归一化头、周期特征网络、低秩双线性模型，以及实振幅、复振幅和共享 Born 版本。统一输入归一化、编码秩、初始概率熵、温度/增益策略、top-k 与优化器搜索预算。分别报告同参数预算及同墙钟预算的结果。

对候选结构依次移除重上传、耦合与相位，并改变专家编号或耦合图。联合振幅编码可能已经生成非乘积态，因此“关闭 CRY”不等于“完全无纠缠”；后者需要乘积态编码和局部操作。固定正交变换若紧邻无约束线性层且中间没有非线性，可能被吸收到线性权重中。共享 Born 的非线性编码不满足这一简单吸收条件，应使用相应的经典正交结构作对照。

先在无容量丢弃下隔离 gate 效应，再恢复相同容量、batch 组成和 packing。冻结表征实验不能替代联合训练；它只能帮助选择值得进一步投入的机制。

== 第二阶段：持续学习与保护策略

采用数学、代码与通用任务等具有真实分布差异的任务序列，并平衡任务顺序。探索阶段可从 3 个种子开始，候选收益出现后至少扩到 5 个种子；最终重复次数应由效应大小与方差决定，而非将该数量视为显著性保证。

设 $A_(i,j)$ 为学习完任务 $i$ 后在任务 $j$ 上的表现，任务数为 $K$。可报告最终平均表现与历史任务遗忘量：
$ bar(A) = 1/K sum_(j=1)^K A_(K,j), quad
  cal(F) = 1/(K-1) sum_(j=1)^(K-1) (max_(j <= i <= K) A_(i,j) - A_(K,j)). $ <eq-forgetting>
同时报告新任务学习速度，比较相同新任务表现下的旧任务保持，避免把停止学习当作抗遗忘。

经典与 Born 两类结构均运行无保护和相同 EWC/旧域路由 KL 保护。旧样本数量、正则强度搜索范围和选择集保持一致。联合训练时应保存旧的原始输入并重新提取表征，只保留冻结 hidden states 可能掩盖 backbone 漂移；即使路由不变，expert 更新也可能损害旧任务输出。

== 第三阶段：稳定性与恢复的分层测量

数据污染、输入扰动、训练样本敏感性与清洁恢复应分别建组。可从 0%、5%、10%、20% 的可控错误标签或答案比例开始，保持干净验证集不变。正常长推理不应被自动当作噪声，截断错误也不能用来证明某种架构的固有缺陷。

输入层面记录概率 JS/KL、top-k 集合翻转和任务损失，并加入有语义意义的改写。数据敏感性实验应在共同初始化下替换少量训练样本。污染后的清洁恢复使用相同恢复预算，并与干净数据重训结果比较；普通恢复实验不应被标记为严格认证的 unlearning。

== 共同指标与统计解释

任务指标应包含准确率、最终答案交付率、代码可执行性及生成触顶率。路由指标包含熵、专家利用率、失活专家、容量丢弃、top-k 间隔和扰动后的选择变化。训练指标则包括梯度、编码器 Jacobian、经验 Fisher 的有效秩、吞吐、显存和达到目标质量的时间。

训练种子是重复单位，相关 token、层和计时重复不能代替独立训练重复。报告配对差值及区间，同时保留失败配置与选择过程。普通 Hessian 的谱受参数缩放影响，未经规范化的谱比较不能独立证明几何优势。

共享辅助损失还需核对数值语义。当前版本名为 z-loss 的实现使用 logsumexp，对 logits 的共同平移敏感；归一化 log 概率与任意经典 logits 可能因此承担不同约束。机制筛选应控制这些差异，避免将辅助目标差别误判为架构收益。

= 局限性与结论

本文综述并未复核全部论文证明，部分最新工作仍为预印本。所讨论公开实验通常规模较小，与大语言模型的混合任务训练存在明显差异。本文的数学关系描述固定测量和特定局部条件，不能直接保证包含可学习编码、概率平滑、硬选择及容量竞争的完整系统稳定性。

现有证据支持将 QuantumGate 视为一类有明确结构约束的概率参数化，并围绕编码、交互与优化度量展开机制研究。最有意义的目标是在公平资源预算下改善混合任务路由的学习与保持权衡，并给出经典强基线无法轻易解释的、可复现的原因。是否需要量子线路、复相位或耦合结构，应由对应消融决定。

固定 Born 读出并不提供超越普通类别概率空间的覆盖优势；它可能提供的价值存在于输入相关函数族及其训练几何。将这一结论与明确的负面对照结合，可以使后续研究更容易被验证，也能在量子结构未获优势时提炼出更便宜的经典实现。

#heading(numbering: none)[附录 A　数学核验与复现边界]

原报告配套 `geometry_checks.py` 验证概率等价、实态 Fisher 等价、相位反例及近并列 top-1 翻转。已保存的双精度结果中，实态 Fisher 矩阵最大差约为 $3.55 times 10^(-15)$；相位反例的经典 Fisher 约为零、量子 Fisher 为 1；总变差约 $2 times 10^(-8)$ 即可使近并列 top-1 交换。这些是数学一致性检查，不构成训练性能或量子优势证据。

代码分析对应 `quantum_gate.py`、`structured_routers.py`、`advanced_routers.py` 与 `experiments/routing_stability/routers.py`。内部实测依据为 `outputs/routing_stability_round1/REPORT_zh.md` 及其完成审计，本文不改写其原始记录。本文排版源文件自包含，编译不依赖在线 Typst 包或外置文献库。

#heading(numbering: none)[参考文献]
#reference(1)[Jiang, Lu, Deng. Quantum Continual Learning Overcoming Catastrophic Forgetting. Chinese Physics Letters, 2022. #link("https://doi.org/10.1088/0256-307X/39/5/050303")[doi:10.1088/0256-307X/39/5/050303].] <ref-1>
#reference(2)[Zhang et al. Experimental demonstration of quantum continual learning with superconducting qubits. npj Quantum Information, 2026. #link("https://www.nature.com/articles/s41534-025-01174-y")[doi:10.1038/s41534-025-01174-y]. 作者预印本：#link("https://arxiv.org/abs/2409.09729")[arXiv:2409.09729].] <ref-2>
#reference(3)[Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks. PNAS, 2017. #link("https://arxiv.org/abs/1612.00796")[arXiv:1612.00796].] <ref-3>
#reference(4)[Chen, Zhang. Superior resilience to poisoning and amenability to unlearning in quantum machine learning. Nature Communications, 2026. #link("https://www.nature.com/articles/s41467-026-70420-4")[doi:10.1038/s41467-026-70420-4]. 机制分析版本：#link("https://arxiv.org/html/2508.02422v1")[arXiv:2508.02422v1].] <ref-4>
#reference(5)[Yang, Xie, Xu. Stability and Generalization of Quantum Neural Networks. 2025, 预印本. #link("https://arxiv.org/abs/2501.12737")[arXiv:2501.12737].] <ref-5>
#reference(6)[Arjovsky, Shah, Bengio. Unitary Evolution Recurrent Neural Networks. ICML, 2016. #link("https://proceedings.mlr.press/v48/arjovsky16.html")[PMLR 48].] <ref-6>
#reference(7)[Caro et al. Generalization in quantum machine learning from few training data. Nature Communications, 2022. #link("https://arxiv.org/abs/2111.05292")[arXiv:2111.05292].] <ref-7>
#reference(8)[Gil-Fuster, Eisert, Bravo-Prieto. Understanding quantum machine learning also requires rethinking generalization. Nature Communications, 2024. #link("https://doi.org/10.1038/s41467-024-45882-z")[doi:10.1038/s41467-024-45882-z].] <ref-8>
#reference(9)[Abbas et al. The power of quantum neural networks. Nature Computational Science, 2021. #link("https://doi.org/10.1038/s43588-021-00084-1")[doi:10.1038/s43588-021-00084-1].] <ref-9>
#reference(10)[Haug, Kim. Generalization of Quantum Machine Learning Models Using Quantum Fisher Information Metric. Physical Review Letters, 2024. #link("https://doi.org/10.1103/PhysRevLett.133.050603")[doi:10.1103/PhysRevLett.133.050603].] <ref-10>
#reference(11)[Stokes, Izaac, Killoran, Carleo. Quantum Natural Gradient. Quantum, 2020. #link("https://quantum-journal.org/papers/q-2020-05-25-269/")[Quantum 4, 269].] <ref-11>
#reference(12)[Braunstein, Caves. Statistical distance and the geometry of quantum states. Physical Review Letters, 1994. #link("https://doi.org/10.1103/PhysRevLett.72.3439")[doi:10.1103/PhysRevLett.72.3439].] <ref-12>
#reference(13)[Barndorff-Nielsen, Gill. Fisher information in quantum statistics. 作者预印本. #link("https://arxiv.org/abs/quant-ph/9808009")[arXiv:quant-ph/9808009].] <ref-13>
#reference(14)[Pérez-Salinas et al. Data re-uploading for a universal quantum classifier. Quantum, 2020. #link("https://arxiv.org/abs/1907.02085")[arXiv:1907.02085].] <ref-14>
#reference(15)[Schuld, Sweke, Meyer. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A, 2021. #link("https://arxiv.org/abs/2008.08605")[arXiv:2008.08605].] <ref-15>
#reference(16)[Meyer et al. Exploiting symmetry in variational quantum machine learning. PRX Quantum, 2023. #link("https://doi.org/10.1103/PRXQuantum.4.010328")[doi:10.1103/PRXQuantum.4.010328].] <ref-16>
#reference(17)[Stoudenmire, Schwab. Supervised Learning with Tensor Networks. 2016. #link("https://arxiv.org/abs/1605.05775")[arXiv:1605.05775].] <ref-17>
#reference(18)[Nguyen et al. QMoE: A Quantum Mixture of Experts Framework for Scalable Quantum Neural Networks. 2025, 预印本. #link("https://arxiv.org/abs/2507.05190")[arXiv:2507.05190].] <ref-18>
#reference(19)[Bowles, Ahmed, Schuld. Better than classical? The subtle art of benchmarking quantum machine learning models. 2024, 预印本. #link("https://arxiv.org/abs/2403.07059")[arXiv:2403.07059].] <ref-19>
#reference(20)[Cerezo et al. Cost function dependent barren plateaus in shallow parametrized quantum circuits. Nature Communications, 2021. #link("https://doi.org/10.1038/s41467-021-21728-w")[doi:10.1038/s41467-021-21728-w].] <ref-20>
#reference(21)[Thanasilp et al. Exponential concentration in quantum kernel methods. Nature Communications, 2024. #link("https://arxiv.org/abs/2208.11060")[arXiv:2208.11060].] <ref-21>
#reference(22)[Liu, Wittek. Vulnerability of quantum classification to adversarial perturbations. Physical Review A, 2020. #link("https://arxiv.org/abs/1905.04286")[arXiv:1905.04286].] <ref-22>
#reference(23)[Lu, Duan, Deng. Quantum Adversarial Machine Learning. Physical Review Research, 2020. #link("https://arxiv.org/abs/2001.00030")[arXiv:2001.00030].] <ref-23>
#reference(24)[Huang et al. Power of data in quantum machine learning. Nature Communications, 2021. #link("https://doi.org/10.1038/s41467-021-22539-9")[doi:10.1038/s41467-021-22539-9].] <ref-24>
#reference(25)[Huang et al. Quantum advantage in learning from experiments. Science, 2022. #link("https://www.preskill.caltech.edu/pubs/preskill-2022-quantum-advantage.pdf")[作者公开全文].] <ref-25>
#reference(26)[Zhao et al. Exponential quantum advantage in processing massive classical data. 2026, 预印本, v2 更新于 2026-10-01. #link("https://arxiv.org/abs/2604.07639")[arXiv:2604.07639].] <ref-26>
