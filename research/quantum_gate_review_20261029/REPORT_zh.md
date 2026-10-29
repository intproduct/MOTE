# 量子机器学习优势向专家路由的迁移

QuantumGate 的信息几何、持续学习与验证框架

研究综述与机制分析 · 2026 年 10 月 29 日

## 摘要

QuantumGate 的研究价值集中在态编码形成的函数结构、概率映射诱导的优化几何，以及任务切换时对重要路由行为的保护。本文结合量子机器学习文献、信息几何推导与 MoTE 已完成的实验，分析这三条机制的实现路径。固定计算基读出的 Born 概率与 softmax 共享普通概率单纯形；结构差异由输入编码、演化方式和读出后的变换共同决定。归一化实态在局部非零振幅条件下具有相等的经典与量子 Fisher 信息，为路由自然梯度和持续学习正则化提供了直接的计算关系。公开持续学习研究表明，重要参数保护能够改善旧任务保持；编码研究则将线路表达能力与可学习频谱联系起来。内部实验显示，共享 Born 相对原 Born 降低整模型前向延迟约 24.5%，同时暴露出相对 half MLP 更高的 PPL 和输入舍入切换率。针对这些结果，本文提出以边界裕量、输入敏感度和专家输出差异为核心的稳定性分析，并给出编码消融、持续学习及输出敏感性评估的实验顺序。

  **关键词：** 量子机器学习；混合专家；Born 规则；Fisher 信息；持续学习；路由稳定性

## 引言

专家路由的作用是将输入表征转换为专家选择和组合权重。这个映射决定了哪些专家获得训练、不同任务如何共享计算，以及模型对输入扰动的响应。QuantumGate 将态编码、受约束演化和 Born 读出引入这一映射，形成了一组可以独立设计和测量的结构因素。

量子机器学习文献提供了三条与路由直接相关的线索。持续学习研究利用参数重要性保护旧任务功能；[1](#ref-1)[2](#ref-2) 编码与重上传研究分析输入如何形成可学习的频率结构；[14](#ref-14)[15](#ref-15) Fisher 几何研究刻画参数变化对状态或输出分布的影响。[10](#ref-10)[11](#ref-11) 三者分别对应任务保持、函数表达和优化路径，可以在同一套路由实验中形成相互衔接的解释。

本文的核心判断是：**QuantumGate 应围绕输入结构与路由几何的匹配进行设计，实验评价同时覆盖任务质量、知识保持、扰动响应和实际计算成本。** 下文先建立几何分析，再梳理公开证据，随后解释已有实验的收益与退化，并据此确定验证顺序。

本文采用截至 2026 年 10 月 29 日的一手文献、标准信息几何关系和 MoTE 内部实验记录。实现分析覆盖提交 `c7d6690` 的重上传路由及 `c726b4e` 保存的共享 Born 实现。内部实测来自 WikiText-2 上的六组路由、三个种子、共 18 项训练及配套评估。

## QuantumGate 的概率结构与训练几何

### 共同的输出空间与不同的函数参数化

设 $x\in\mathbb R^d$ 为输入表征，$E$ 为专家数，$\psi_\theta(x)\in\mathbb C^E$ 为归一化纯态。计算基读出的专家概率为

$$
p_i(x;\theta)=|\psi_{\theta,i}(x)|^2,\qquad \sum_{i=1}^{E}p_i=1.
$$

输出位于概率单纯形

$$
\Delta^{E-1}=\left\{p\in\mathbb{R}^{E}:p_i\geq0,\ \sum_i p_i=1\right\}.
$$

  任意类别分布都对应一个振幅构造 $\psi_i=\sqrt{p_i}$；任意严格正的类别分布也满足

$$
p=\operatorname{softmax}(\log p).
$$


因此，Born 读出与 softmax 在单纯形内部具有相同的概率覆盖范围。对当前采用概率平滑的实现，研究重点可以明确放在 $x\mapsto p_\theta(x)$ 的函数结构与学习动态上。

振幅参数化提供了一个具体的比较入口。对非零实振幅 $a$，有

$$
p_i=\frac{a_i^2}{\sum_j a_j^2}=\operatorname{softmax}(2\log|a|)_i.
$$

这个等式把平方读出对应到一组特殊 logits。它使实验可以分别考察平方归一化的作用和演化层的作用：先固定编码器比较两类读出，再加入旋转、耦合或重上传，测量每一项带来的增量。

受限线路的意义在于施加结构先验。参数共享、局部耦合和重复编码共同约束可达到的函数；当这些约束与任务所需的交互和频率一致时，模型可能以较少自由度学到有效的路由。输入编码由此成为连接 Hilbert 空间几何与实际任务的关键设计变量。

### 后处理决定实际的专家权重

当前实现包含概率平滑及共享增益、温度。设 $\epsilon>0$、增益为 $g$、温度为 $T$，在 top-k 之前有

$$
\widetilde p_i=\frac{p_i+\epsilon}{1+E\epsilon},\qquad w_i=\frac{\widetilde p_i^{g/T}}{\sum_j\widetilde p_j^{g/T}}.
$$

比值 $g/T$ 直接控制权重锐度：增大它会强化高概率专家，减小它会使权重接近均匀。概率底值同时改变低概率区域的灵敏度。因此，结构实验应统一初始熵、增益与温度策略，把权重锐度作为显式控制变量。

图 1　映射链：输入 → 归一化态 → Born 概率 → 实际权重 → top-k 专家集合。编码 Jacobian、态几何、概率后处理和选择边界分别控制不同阶段的响应。

### 状态几何与测量信息

将纯态局部表示为 $\psi_i=\sqrt{p_i}e^{\mathrm i\phi_i}$，去除全局相位后，Fubini–Study 线元为

$$
\mathrm{d}s_{\mathrm{FS}}^2=\frac14\sum_i\frac{(\mathrm{d}p_i)^2}{p_i}+\sum_i p_i(\mathrm{d}\phi_i)^2-\left(\sum_i p_i\mathrm{d}\phi_i\right)^2.
$$

该分解包含概率变化和相位变化两部分。固定计算基测量直接观测第一部分；相位通过后续混合操作进入概率读出。量子统计几何将这种测量信息损失写为 $F^{\mathrm C}\preceq F^{\mathrm Q}$。[12](#ref-12)[13](#ref-13)

对路由保护而言，度量的选择对应明确的目标。输出 Fisher 或 KL 衡量专家概率的变化，适合保持已有的专家分工；量子 Fisher 还追踪读出前的状态方向，适合研究相位与后续混合操作之间的协同关系。比较两种度量，可以判断额外状态信息是否改善后续学习。

### 实振幅提供可直接利用的 Fisher 等价

考虑归一化、平滑依赖参数且局部振幅非零的实态 $\psi(\theta)$。由 $p_i=\psi_i^2$ 及归一化条件得到

$$
\partial_a p_i=2\psi_i\partial_a\psi_i,\qquad\sum_i\psi_i\partial_a\psi_i=0.
$$

输出概率的经典 Fisher 矩阵于是满足

$$
F^{\mathrm C}_{ab}=\sum_i\frac{(\partial_a p_i)(\partial_b p_i)}{p_i}=4\sum_i(\partial_a\psi_i)(\partial_b\psi_i)=F^{\mathrm Q}_{ab}.
$$

归一化切向约束消去了纯态量子 Fisher 中沿状态自身方向的项，形成了上述等价。对实振幅联合读出，这意味着可以从概率导数或状态导数两条路径构造同一个局部度量。

**设计结论：优先研究输出相关的几何优化。** `amplitude_born` 和共享实振幅路由适合用输出 Fisher、KL 约束及相应自然梯度进行配对实验。概率平滑与温度改变读出 Jacobian，因此应基于实际权重重新计算度量，并将平滑前后的结果分别记录。

复振幅提供了相位维度的对照。对于

$$
\psi(\theta)=\frac1{\sqrt2}\begin{pmatrix}1\\e^{\mathrm i\theta}\end{pmatrix},\qquad F^{\mathrm C}=0,\qquad F^{\mathrm Q}=1,
$$

相位参数改变了状态，计算基概率保持 $(1/2, 1/2)$。这一例子具体展示了固定读出丢失的信息方向，也给出了相位消融的解释方式：测量相位参数经过最终混合层之后对概率的 Jacobian，即可识别它对当前任务的实际贡献。

### 从正交演化到路由敏感度

固定酉或正交矩阵保持态向量的欧氏距离，这使其适合作为受约束的特征混合算子。重上传结构中的算子还随输入变化，完整导数包含编码与算子两条路径：

$$
\frac{\mathrm d}{\mathrm dx}\big(U(x)a(x)\big)=\frac{\mathrm dU}{\mathrm dx}a(x)+U(x)\frac{\mathrm da}{\mathrm dx}.
$$

共享 Born 将 $U$ 固定为输入无关的可学习算子，因而消去第一项；剩余响应由编码器和读出控制。这个分解为效率与稳定性提供了同一个分析框架：共享算子减少逐 token 的算子构造，同时改变输入敏感度的来源。

  硬选择的稳定性取决于分数间隔。设第 $k$ 与第 $k+1$ 名间隔为 $m_k=z_{(k)}-z_{(k+1)}>0$，则

$$
\|\delta z\|_\infty<\frac{m_k}{2}
$$

  是 top-k 集合保持不变的充分条件。

在局部线性近似下，$\|\delta z\|_\infty$ 由输入扰动与路由 Jacobian 共同决定。提高间隔和降低 Jacobian 都能扩大选择的稳定区域。该关系将“路由更稳定”转化为两个可测量、可干预的训练目标。

## 公开研究提供的三条设计路线

### 持续学习：保护旧任务敏感的参数方向

量子持续学习研究表明，参数重要性保护能够改善旧任务保持。Jiang 等的工作及后续超导硬件实验都围绕这一机制展开。[1](#ref-1)[2](#ref-2) 新任务训练会改变同时服务于旧任务的参数，保护机制通过提高重要方向上的更新成本，降低任务间干扰。

经典 EWC 将这一思想写成加权二次惩罚：[3](#ref-3)

$$
\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{new}}+\frac\lambda2\sum_j F_j(\theta_j-\theta_j^\star)^2.
$$

$F_j$ 表示旧任务的重要性估计，较大的值意味着该方向的变化更容易损害旧功能。对 QuantumGate，可分别估计旧任务损失 Fisher 与旧域路由 Fisher：前者直接关联任务输出，后者关联专家选择。比较两者能揭示遗忘主要发生在任务表征、专家分工还是专家内部参数。

路由 KL 提供了函数层面的保护方式。保存旧域输入，在学习新域时约束同一输入的专家分布变化，可以将参数尺度的影响转化为输出变化的直接度量。实振幅 Fisher 等价又给出了局部二阶分析工具，使参数保护、概率保护和态几何能够在同一模型上比较。

长期序列记忆对应状态沿时间的传播。酉循环网络利用受约束的递归变换改善长距离梯度传播。[6](#ref-6) 当前 gate 按 token 独立构造状态，其记忆研究应集中于训练过程中参数与专家分工的保持。若后续引入跨 token 状态，递归稳定性将成为一个新增的设计维度。

### 编码与频谱：让结构对应任务的交互形式

数据重上传通过多次编码输入扩展可表达的频率结构；编码算子的谱、重复次数和可训练变换共同控制模型的函数族。[14](#ref-14)[15](#ref-15) 对路由而言，这一视角提示首先研究哪些输入特征需要周期响应或高阶交互，再选择相应的编码和演化深度。

对称性研究展示了任务结构与受约束模型之间的匹配价值。[16](#ref-16) 经典 MPS 监督学习则提供了低秩高阶特征的实现参照。[17](#ref-17) 两类研究都把性能来源指向结构先验：模型以特定方式共享或分解交互，使有限参数优先用于任务相关方向。

专家编号是实现中人为指定的离散坐标。若线路的耦合图偏好某些二进制邻接关系，这种偏好就会影响哪些专家之间更容易形成联合响应。随机置换专家编号、改变耦合图并保持非路由权重一致，可以测出结果对该结构先验的依赖。稳定跨置换的收益更适合推广到新的专家布局；强烈依赖特定布局的收益则提示需要显式学习专家组织。

### 泛化与优化：有效方向决定学习效率

QNN 的算法稳定性研究将数据变化引起的损失变化与门数、步长、训练过程及损失条件联系起来。[5](#ref-5) 少样本泛化研究进一步强调有效可训练自由度的作用。[7](#ref-7) 这些结果共同支持一个工程原则：统计编码器与线路的总可训练容量，并记录训练过程中实际活跃的参数方向。

有效维度与数据相关量子 Fisher 提供了测量这些方向的工具。[9](#ref-9)[10](#ref-10) 若大量参数对当前数据产生近乎相同的变化，优化将面临冗余；若有效方向覆盖任务所需变化，模型可以更充分地利用容量。应将 Fisher 的有效秩、条件数与验证损失配对观察，确定增加深度、秩或相位究竟扩展了什么功能。

量子自然梯度根据态空间度量调整更新方向。[11](#ref-11) 在当前小态空间上，精确或块近似度量的成本可以直接测量。候选方法的收益由达到目标质量的总耗时决定，包含度量估计、求解、前后向及额外存储。这样的核算能够选择实际有用的几何优化方案。

### 噪声数据：研究损失曲率与错误影响的恢复

近期数据污染与 unlearning 研究在 MNIST 二分类、量子相分类等设置中报告了量子模型的韧性和近似遗忘效果。[4](#ref-4) 这为 QuantumGate 增加了两个具体问题：错误样本是否推动路由发生过大的更新，以及去掉错误样本后，专家分工是否容易恢复。

对于标量 logit $z(\theta)$、标签 $y$ 和预测 $p=\operatorname{sigmoid}(z)$，交叉熵的曲率为

$$
\frac{\mathrm d^2\mathcal L}{\mathrm d\theta^2}=p(1-p)(z')^2+(p-y)z''.
$$

第一项由预测不确定性和一阶敏感度构成，第二项将预测残差与 logit 曲率联系起来。余弦等周期参数化会改变第二项，因此经典周期网络是定位机制的重要对照。通过共同污染样本、共同初始化和相同恢复预算，可以检验韧性收益与周期性、振幅约束或线路交互的关系。

## 当前实现与实测结果

### 读出结构决定对照方式

分块 `quantum` gate 先读取各 qubit 的边缘概率，再归一化为专家权重，适合研究边缘特征的组合。`compact_born` 使用 $q=\log_2 E$ 个 qubit 的联合读出，直接将基态概率映射到专家，生产实现覆盖 2 至 64 个专家且要求 $E$ 为 2 的幂。`amplitude_born` 采用实振幅编码与输入相关旋转，适合考察平方读出、重上传和耦合的增量作用。

`SharedBornRouter` 将演化合并为输入无关的正交矩阵，在推理时缓存该矩阵并以矩阵乘法作用于输入态。MPS 与 residual MPO 则提供经典张量结构参照。将这些实现按“编码—交互—读出”拆分，可以避免一次替换多个组件后难以定位收益来源。

### 正面结果：共享算子降低推理开销

已完成实验使用两层因果语言模型、WikiText-2 训练集拟合的 BPE4096、16 个专家、top-4、FP32 和 TITAN V。每个配置训练 4000 步，六组配置分别运行三个种子。batch16、context128 的整模型前向中，原 Born 平均延迟为 88.727 ms，共享 Born 为 66.998 ms；逐种子配对平均下降约 24.5%。

这一收益与实现中的计算路径一致。输入无关的 $U$ 可以在推理预热时构造并缓存，逐 token 的线路演化被合并为高效的矩阵乘法。共享 Born 的串行训练步耗时相对原 Born 下降约 4.8%，训练仍需更新并反向传播该算子，因此训练端保留了更多计算。

由此得到的工程方向是：优先识别并缓存输入无关的结构，将可复用算子与输入相关编码分离。该方向可以应用到实数正交路由及其他具有共享小矩阵的结构模型。

### 负面结果：共享 Born 的质量与舍入敏感度落后于 half MLP

当前设置下，half MLP 的 test PPL 为 $79.48\pm1.98$，共享 Born 为 $80.39\pm2.47$，数值为三个种子的均值与样本标准差。共享 Born 在三个种子上的 PPL 均更高，配对平均增加 1.135%。其 batch16、context128 前向相对 half MLP 平均慢约 6.9%。这组结果将后续优化目标明确为语言建模质量、推理成本和扰动敏感度的联合改善。

局部 BF16 输入舍入探针中，half MLP 的 top-4 集合切换率为 $0.1724\%$，共享 Born 为 $0.2406\%$。这里的运算主体保持 FP32，扰动来自 router 输入的 BF16 舍入后回转。共享 Born 同时具有较高的软概率熵和更多接近选择边界的决策：标准化 gap 低于 0.05 的比例约为 72%，half MLP 约为 57%。

这些观测给出了一个可检验解释：共享 Born 的权重更接近均匀，部分专家排序因而更容易受小扰动改变。跨架构 gap 使用的分数尺度也参与其中。下一步应联合记录实际分数扰动、选择间隔和编码 Jacobian，并用统一温度或初始熵对照，区分锐度变化与输入敏感度的贡献。

### 负面结果：减少切换次数后，输出尾部仍可能恶化

边界训练使 MLP 的局部 BF16 切换率按种子配对平均下降约 10.3%，共享 Born 下降约 9.1%。在完整模型传播测试中，MLP 的目标 token logprob 绝对误差大于 0.1 的事件由 42 增至 64，样本总数均为 49,152；三个种子的此类事件数均增加。这个结果指出，单独优化选择间隔遗漏了切换的后果。

对单层加权专家输出 $y(x)=\sum_i w_i(x)f_i(x)$，固定专家函数时，权重变化带来的输出差为

$$
\delta y=\sum_i\delta w_i f_i(x).
$$

在硬选择发生替换时，被替换专家的输出差异直接参与误差。两个功能相近专家的交换可能影响很小；一次涉及差异较大专家的交换可能产生较大偏移。多层网络再通过后续 Jacobian 传播该偏移，形成输出误差尾部。

因此，下一轮稳定性目标应同时测量“切换概率”和“切换影响”。可检验的改进是根据候选专家的输出距离或短程输出敏感度对边界损失加权，使训练重点落在后果较大的选择边界上。

## 研究假设与实验顺序

### 假设一：编码匹配带来任务质量收益

假设 H1 为：态编码与受约束交互匹配混合任务中的有效特征，从而在共同预算下提高验证质量。第一阶段冻结 backbone 与 experts，比较强 MLP、平方振幅头、周期特征网络、低秩双线性模型、实振幅 Born、复振幅 Born 和共享 Born。保持输入归一化、编码秩、初始熵、温度、top-k 与优化器搜索预算一致。

对表现较好的结构逐项移除重上传、相位和耦合。联合振幅编码自身可能形成非乘积态，因此无纠缠基线应采用乘积态编码和局部操作。固定正交矩阵紧邻无约束线性层时可以合并进线性权重；当前共享 Born 的非线性振幅编码保留了一个需要独立测量的交互层。相应对照应包括同编码器的可学习正交矩阵，以定位线路分解产生的额外约束。

先采用无容量丢弃设置隔离 gate 的作用，再恢复相同容量与 batch 组成。第一阶段的目标是筛选有效机制，第二阶段进行联合训练，测量 gate 与 backbone、experts 的共同适应。

### 假设二：几何保护改善新旧任务的学习权衡

假设 H2 为：输出相关的参数保护使 Born 路由在相同新任务表现下保留更多旧任务能力。使用数学、代码与通用任务组成的序列，平衡任务顺序；设置经典/Born 结构与无保护/相同保护的双因素实验。保护方法先采用旧域路由 KL 与经验 Fisher EWC，再比较态空间度量。

设 $A_{i,j}$ 为学习完任务 $i$ 后在任务 $j$ 上的分数，任务数 $K > 1$，记录最终平均分与遗忘量：

$$
\overline A=\frac1K\sum_{j=1}^{K}A_{K,j},\qquad\mathcal F=\frac1{K-1}\sum_{j=1}^{K-1}\left(\max_{j\leq i\leq K}A_{i,j}-A_{K,j}\right).
$$

同时绘制新任务学习速度与旧任务保留的权衡曲线。探索阶段可从三个种子开始，候选结果出现后扩展到至少五个种子，并根据效应大小和方差决定后续样本量。

旧样本预算、正则强度搜索和选模数据保持一致。联合训练时保存旧的原始输入并重新提取表征，分别观察 backbone 漂移、路由漂移和 expert 更新对旧任务的贡献。这样可以把整体遗忘分解到实际发生变化的组件。

### 假设三：输出敏感性约束降低有害路由扰动

假设 H3 为：结合边界裕量、编码敏感度及专家输出差异的目标，能够降低传播后的预测误差。先复现已有舍入探针，再加入逐坐标 RMS 缩放扰动与语义改写；同时记录局部选择变化和完整模型输出变化。

训练策略可按三步递增：第一步统一温度并控制初始熵；第二步比较边界损失与路由 Jacobian 正则化；第三步引入候选专家输出距离加权。每一步都在相同任务损失、负载均衡和计算预算下比较，并跟踪大误差事件数与误差尾部。

污染与清洁恢复作为独立实验组。以 0%、5%、10%、20% 的错误标签或答案构造共同污染数据，保持干净验证集固定；移除错误样本后给予相同恢复预算，并与干净数据重训结果比较。评估目标是错误影响的大小及恢复速度。

### 共同的测量与统计方法

任务指标包括准确率、最终答案交付率、代码可执行性和生成触顶率。路由指标包括熵、专家利用率、失活专家、容量丢弃、分数扰动、选择间隔和集合翻转。优化指标包括梯度、编码 Jacobian、Fisher 有效秩、吞吐、显存和达到目标质量的时间。

统计以训练种子为重复单位，报告配对差值及区间。token、层和计时重复用于描述一次训练内部的分布。跨模型几何比较采用共同的参数尺度或函数空间度量，使谱的变化对应可解释的输出变化。

共享辅助目标需要控制一致的数值语义。当前 z-loss 使用 logsumexp，归一化 log 概率与任意 logits 在共同平移下的行为不同。机制实验先关闭这项约束，再分别启用并报告增量，从而把结构效应与辅助损失效应分开测量。

## 讨论：失败机制与资源条件

### 结构复杂度的代价

小型分类系统基准表明，经典方法在多组设置中优于所比较的量子模型，部分模型移除纠缠后仍保持或提高表现。[19](#ref-19) 这类结果意味着任务相关的结构选择比单纯增加线路复杂度更重要：额外交互会占用参数和优化预算，其收益需要通过与数据结构的匹配体现。

梯度消失和量子核集中给出了两种具体失败机制。成本函数与线路结构可能使梯度随系统规模快速减小；[20](#ref-20) 高表达性、测量方式及噪声也可能压缩样本间可用的核差异。[21](#ref-21) 对当前小态空间，深度扫描应同时记录梯度信号、样本间距离和验证损失，使复杂度增加的成本能够被及时识别。

QNN 对随机标签的拟合能力说明其容量足以吸收训练噪声。[8](#ref-8) 泛化分析因此应结合验证集、容量控制与数据治理。输入对抗脆弱性的研究则说明，分类边界与高维几何能够产生敏感方向。[22](#ref-22)[23](#ref-23) 这进一步支持以实际 Jacobian 和选择边界作为路由鲁棒性的测量对象。

### 硬件访问模型决定可利用的资源优势

量子学习的资源优势依赖任务所允许的数据访问和测量方式。特征几何研究分析了任务与量子表示的匹配；[24](#ref-24) 量子实验学习利用相干处理与测量资源降低特定学习问题的样本成本。[25](#ref-25) 2026 年海量经典数据处理预印本提出基于 quantum oracle sketching 的机器规模优势，其摘要把资源分离归因于所设计的量子访问机制。[26](#ref-26)

当前 QuantumGate 使用经典 GPU 上的小态向量模拟，实际成本主要来自编码投影、算子应用、读出和路由后的专家计算。对这一部署条件，最有用的资源目标是同质量下的时间与显存，以及同时间下的任务表现。共享算子缓存已经给出了一个可复现的优化方向。

### 证据如何指导下一轮选择

当前内部结果来自单一小型语料、两层模型和三个训练种子，直接覆盖的是这一配置下的质量、前向成本和舍入响应。拓展到混合数学、代码与通用任务后，专家分工和域间干扰会成为新的变量，因此下一轮应优先引入任务切换与多域验证。

候选方法可按收益来源分类。平方振幅头复现的收益归于读出参数化；周期网络复现的收益归于编码频谱；相同保护策略下的共同改进归于持续学习目标；在这些对照之后保留的差异，再用耦合图、相位和正交约束消融解释。已有 QMoE 研究提供了量子专家与量子路由的相关先例，[18](#ref-18) 本项目可围绕经典语言模型中的结构归因、效率和知识保持形成具体贡献。

## 结论

QuantumGate 的研究应围绕三项机制展开：编码与任务交互的匹配、输出相关的几何优化，以及旧任务重要路由行为的保护。固定 Born 读出与 softmax 共享概率单纯形，使函数结构和训练路径成为最直接的比较对象。实振幅 Fisher 等价提供了几何优化的计算基础，持续学习文献则提供了参数保护与任务干扰的研究框架。

已有实验确定了两个优先问题。共享算子显著降低原 Born 的推理开销，同时其相对 half MLP 的质量与舍入敏感度需要改善。边界训练降低了部分选择切换，输出误差尾部的表现进一步要求将专家输出差异纳入目标。基于这些证据，后续工作依次开展编码消融、持续学习和输出敏感性约束，可以同时解释模型为何改善、为何退化，以及改进应作用于哪一个组件。

## 附录 A　数学核验与复现信息

`geometry_checks.py` 对概率等价、实态 Fisher 等价、相位例子及近并列 top-1 翻转进行双精度核验。保存结果中，实态 Fisher 矩阵最大差约为 $3.55\times10^{-15}$；相位例子的经典 Fisher 约为零、量子 Fisher 为 1；总变差约 $2\times10^{-8}$ 时出现近并列 top-1 交换。这些结果验证了本文用于解释路由机制的代数关系与数值例子。

内部实验数据来自 `outputs/routing_stability_round1/REPORT_zh.md` 及完成审计；模型实现位于 `quantum_gate.py`、`structured_routers.py`、`advanced_routers.py` 和 `experiments/routing_stability/routers.py`。参考文献 26 的使用范围为作者摘要与版本信息，其余文献的访问版本见配套文献索引。本文源文件包含排版和参考文献，使用本地 Typst 可直接编译。

## 参考文献

<a id="ref-1"></a>

1. Jiang, Lu, Deng. Quantum Continual Learning Overcoming Catastrophic Forgetting. Chinese Physics Letters, 2022. [doi:10.1088/0256-307X/39/5/050303](https://doi.org/10.1088/0256-307X/39/5/050303).

<a id="ref-2"></a>

2. Zhang et al. Experimental demonstration of quantum continual learning with superconducting qubits. npj Quantum Information, 2026. [doi:10.1038/s41534-025-01174-y](https://www.nature.com/articles/s41534-025-01174-y). 作者预印本：[arXiv:2409.09729](https://arxiv.org/abs/2409.09729).

<a id="ref-3"></a>

3. Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks. PNAS, 2017. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796).

<a id="ref-4"></a>

4. Chen, Zhang. Superior resilience to poisoning and amenability to unlearning in quantum machine learning. Nature Communications, 2026. [doi:10.1038/s41467-026-70420-4](https://www.nature.com/articles/s41467-026-70420-4). 机制分析版本：[arXiv:2508.02422v1](https://arxiv.org/html/2508.02422v1).

<a id="ref-5"></a>

5. Yang, Xie, Xu. Stability and Generalization of Quantum Neural Networks. 2025, 预印本. [arXiv:2501.12737](https://arxiv.org/abs/2501.12737).

<a id="ref-6"></a>

6. Arjovsky, Shah, Bengio. Unitary Evolution Recurrent Neural Networks. ICML, 2016. [PMLR 48](https://proceedings.mlr.press/v48/arjovsky16.html).

<a id="ref-7"></a>

7. Caro et al. Generalization in quantum machine learning from few training data. Nature Communications, 2022. [arXiv:2111.05292](https://arxiv.org/abs/2111.05292).

<a id="ref-8"></a>

8. Gil-Fuster, Eisert, Bravo-Prieto. Understanding quantum machine learning also requires rethinking generalization. Nature Communications, 2024. [doi:10.1038/s41467-024-45882-z](https://doi.org/10.1038/s41467-024-45882-z).

<a id="ref-9"></a>

9. Abbas et al. The power of quantum neural networks. Nature Computational Science, 2021. [doi:10.1038/s43588-021-00084-1](https://doi.org/10.1038/s43588-021-00084-1).

<a id="ref-10"></a>

10. Haug, Kim. Generalization of Quantum Machine Learning Models Using Quantum Fisher Information Metric. Physical Review Letters, 2024. [doi:10.1103/PhysRevLett.133.050603](https://doi.org/10.1103/PhysRevLett.133.050603).

<a id="ref-11"></a>

11. Stokes, Izaac, Killoran, Carleo. Quantum Natural Gradient. Quantum, 2020. [Quantum 4, 269](https://quantum-journal.org/papers/q-2020-05-25-269/).

<a id="ref-12"></a>

12. Braunstein, Caves. Statistical distance and the geometry of quantum states. Physical Review Letters, 1994. [doi:10.1103/PhysRevLett.72.3439](https://doi.org/10.1103/PhysRevLett.72.3439).

<a id="ref-13"></a>

13. Barndorff-Nielsen, Gill. Fisher information in quantum statistics. 作者预印本. [arXiv:quant-ph/9808009](https://arxiv.org/abs/quant-ph/9808009).

<a id="ref-14"></a>

14. Pérez-Salinas et al. Data re-uploading for a universal quantum classifier. Quantum, 2020. [arXiv:1907.02085](https://arxiv.org/abs/1907.02085).

<a id="ref-15"></a>

15. Schuld, Sweke, Meyer. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A, 2021. [arXiv:2008.08605](https://arxiv.org/abs/2008.08605).

<a id="ref-16"></a>

16. Meyer et al. Exploiting symmetry in variational quantum machine learning. PRX Quantum, 2023. [doi:10.1103/PRXQuantum.4.010328](https://doi.org/10.1103/PRXQuantum.4.010328).

<a id="ref-17"></a>

17. Stoudenmire, Schwab. Supervised Learning with Tensor Networks. 2016. [arXiv:1605.05775](https://arxiv.org/abs/1605.05775).

<a id="ref-18"></a>

18. Nguyen et al. QMoE: A Quantum Mixture of Experts Framework for Scalable Quantum Neural Networks. 2025, 预印本. [arXiv:2507.05190](https://arxiv.org/abs/2507.05190).

<a id="ref-19"></a>

19. Bowles, Ahmed, Schuld. Better than classical? The subtle art of benchmarking quantum machine learning models. 2024, 预印本. [arXiv:2403.07059](https://arxiv.org/abs/2403.07059).

<a id="ref-20"></a>

20. Cerezo et al. Cost function dependent barren plateaus in shallow parametrized quantum circuits. Nature Communications, 2021. [doi:10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w).

<a id="ref-21"></a>

21. Thanasilp et al. Exponential concentration in quantum kernel methods. Nature Communications, 2024. [arXiv:2208.11060](https://arxiv.org/abs/2208.11060).

<a id="ref-22"></a>

22. Liu, Wittek. Vulnerability of quantum classification to adversarial perturbations. Physical Review A, 2020. [arXiv:1905.04286](https://arxiv.org/abs/1905.04286).

<a id="ref-23"></a>

23. Lu, Duan, Deng. Quantum Adversarial Machine Learning. Physical Review Research, 2020. [arXiv:2001.00030](https://arxiv.org/abs/2001.00030).

<a id="ref-24"></a>

24. Huang et al. Power of data in quantum machine learning. Nature Communications, 2021. [doi:10.1038/s41467-021-22539-9](https://doi.org/10.1038/s41467-021-22539-9).

<a id="ref-25"></a>

25. Huang et al. Quantum advantage in learning from experiments. Science, 2022. [作者公开全文](https://www.preskill.caltech.edu/pubs/preskill-2022-quantum-advantage.pdf).

<a id="ref-26"></a>

26. Zhao et al. Exponential quantum advantage in processing massive classical data. 2026, 预印本, v2 更新于 2026-10-01. [arXiv:2604.07639](https://arxiv.org/abs/2604.07639).
