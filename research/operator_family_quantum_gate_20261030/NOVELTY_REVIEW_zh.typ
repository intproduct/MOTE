// Generated from the matching Markdown report. Edit the Markdown and rerun.
#set page(paper: "a4", margin: (x: 23mm, y: 22mm), numbering: "1")
#set text(font: ("Times New Roman", "KaiTi"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.75em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 19pt, fill: rgb("17365d"))
#show heading.where(level: 2): set text(size: 14pt, fill: rgb("17365d"))
#show heading.where(level: 3): set text(size: 12pt)
#show link: set text(fill: rgb("24578a"))

#heading(level: 1)[#text("QuantumGate 创新边界复核")]

#text("2026-10-30｜针对已有量子路由及相邻经典工作的补充核查")

#heading(level: 2)[#text("1. 结论与对上一份报告的修正")]

#strong[#text("量子路由的一般概念已有先例；当前 Born 路由原型与已有工作存在实质重合。")]#text(" “用量子线路产生专家概率”“量子 gate 加经典专家”“量子路由加 top-k”均不宜作为首次性主张。当前实现的编码、门集、稀疏执行和应用对象有差异，方法创新的强度需要由新的设计原则、算法和验证结果支撑。")

#text("本轮扩大检索后，还找到了反事实路由优化和 router-expert 几何耦合的直接先例。上一份报告对这些相邻工作的覆盖不足，不能据其提出的方向推定新颖性。“算子效用地图”和“固定专家、只训练 gate”仍是必要的诊断方法，应作为借鉴已有研究的实验基础。")

#text("现阶段更合适的定位是：在已有量子路由框架上，研究面向结构化稀疏算子组合的、具有明确任务效用与成本目标的新路由方法。其新颖性仍需与最接近方法逐项核对，性能需要新实验验证。")

#heading(level: 2)[#text("2. 原先两篇工作的重合程度")]

#heading(level: 3)[#text("QMoE：一般架构有重合，计算对象不同")]

#text("Nguyen 等的 QMoE 于 2025 年 7 月公开，路由与专家均采用量子线路，并以受控操作组织专家及聚合。它已覆盖可训练量子选择器与专家共同适应的总体思路；我们的专家主要是经典结构化算子。")#link("https://arxiv.org/html/2507.05190v1")[#text("QMoE v1")]#text("。")

#text("对应的创新边界：可以研究新的经典稀疏算子路由方法，不能把“量子机制选择专家”本身作为新概念。换用不同专家类型是一项架构差异，其研究价值取决于是否带来新的机制或解决新的瓶颈。")

#heading(level: 3)[#text("Hybrid QMoE：与我们的基本公式高度重合")]

#text("Heddad 与 Bouanane 于 2025 年 12 月公开的工作，明确描述输入量子编码、可训练幺正变换、计算基 Born 概率和经典线性专家加权组合。这与我们现有原型的基本概率生成流程高度重合。")#link("https://arxiv.org/html/2512.22296v1")[#text("Hybrid QMoE v1，第 III、IV 节")]#text("。")

#text("我们的低秩有界联合幅度编码、共享实值 RY/CRY、缓存、top-k 执行和 MoTE 张量算子字典属于具体设计差异。只更换旋转门、调整线路深度或换一个数据集，通常只能构成较小的增量。若能导出新的结构原则并在质量与实际成本上显示稳定收益，具体结构也可以承载较强的方法贡献。")

#text("需要区分论文的概念描述与已核对代码。其公开 Phase 2、Phase 3 notebook 实际采用 Pauli-Z 期望值、线性 adapter 和 softmax；这与正文直接 Born 概率读出不同。应分别复现公开实现和按正文构造的变体，标明来源，避免把差异混入对照。正文已经公开的 Born 路由思想仍构成相关先例。")#link("https://github.com/RH2004/QMoE")[#text("作者代码库")]#text("。")

#heading(level: 2)[#text("3. 本轮新增的直接与相邻先例")]

#heading(level: 3)[#text("量子 top-k 与负载控制也已有工作")]

#text("公开 QRoute 原型采用学习到的角度编码、VQC、计算基概率和 top-k 专家权重，测试对象是合成簇分类。读取 ToyMoE 源码发现它计算所有专家输出后再混合，不能把该原型直接当作已验证的稀疏 dispatch 性能基线。")#link("https://github.com/Quantum-Labor/qroute")[#text("QRoute")]#text("，")#link("https://github.com/Quantum-Labor/qroute/blob/main/qroute/training/toy_moe.py")[#text("ToyMoE 源码")]#text("。")

#text("QMoE+ 已列于 TMLR 2026 年 8 月论文清单，作者代码包含 top-k 量子路由、数据重上传专家、相干聚合及负载均衡，并对选中的样本执行专家。我们核对了期刊目录与公开代码；OpenReview PDF 本轮受浏览器验证阻挡，不能据搜索缓存冒充完整阅读最终论文。")#link("https://jmlr.org/tmlr/papers/")[#text("TMLR 论文清单")]#text("，")#link("https://github.com/HbjNiser/qmoe-plus")[#text("QMoE+ 作者代码")]#text("。")

#text("因此，“引入稀疏选择”或“加入负载均衡”也不足以独立承担核心创新。经典 GPU 的数值舍入与 batch 改变所导致的路由错配，同这些工作中的量子硬件噪声是不同问题，需要针对性的定义、基线和方法。")

#heading(level: 3)[#text("反事实效用与几何耦合已有直接研究")]

#text("2026 年的 When Are Experts Misrouted? 在冻结模型内比较等计算量候选路由，并提出 Expert Preference Optimization，只更新路由以利用更低损失的选择。它与上一份报告建议的效用反馈试验直接相关，应纳入方法对照。")#link("https://arxiv.org/html/2605.07260v1")[#text("论文，第 3 至 5 节")]#text("。")

#text("Routers Learn the Geometry of Their Experts 研究路由与专家梯度的几何耦合、负载均衡的影响，并给出在线聚类路由器。它已涉及我们讨论的前向选择与反向共同适应关系。")#link("https://arxiv.org/html/2605.12476v1")[#text("论文")]#text("。Geometric Routing Enables Causal Expert Control 还研究了几何路由、专家功能与干预证据。")#link("https://arxiv.org/html/2604.14434v1")[#text("论文")]#text("。")

#text("另一个必要的经典参照是 DSelect-k：它通过二进制编码构造可微稀疏选择器。对乘积态逐比特测量的概率可以写成 Bernoulli 因子的乘积，与其连续松弛单选择器具有可比较的代数结构；两者的整体训练与稀疏化规则仍需分别核对。这一比较是本报告的数学观察，提示“少量比特索引多个专家”本身并非量子独有。")#link("https://proceedings.neurips.cc/paper_files/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html")[#text("DSelect-k，NeurIPS 2021")]#text("。")

#heading(level: 2)[#text("4. 当前工作应如何定位")]

#strong[#text("已完成工作的直接价值是建立可训练原型、真实 GPU 评估流程和失败诊断。")]#text(" 在现有先例下，若论文主要内容只是可训练量子 gate、若干门集变体和小模型分数比较，方法创新的说服力偏弱。")

#text("具体结构不是唯一可贡献的层面。仍可研究新的路由目标、低成本信用分配、可解释的编码与算子关系、明确条件下的稳定性保证，以及达到实测质量成本优势的实现方法。每一项都需要给出与最近工作的具体增量；把已有概念组合起来并命名，不能自动建立创新性。")

#text("建议把候选贡献写成可失败的陈述：通过训练数据上的算子作用关系确定编码、耦合或读出约束，在固定专家与激活预算下，降低错误选择的代价，并改善组合泛化或数值扰动下的预测稳定性。")

#text("这一陈述要求回答：关系怎样估计、如何进入模型、为何优于一般正交或双线性路由、额外成本多少、哪些任务上失效。在当前核查的工作中，没有看到与这一完整方案相同的已验证实现；检索范围不足以据此宣称全球首次，具体算法也尚未形成。")

#heading(level: 2)[#text("5. “更好的量子路由”需要通过的验证")]

#text("第一，比较对象应包含已有量子路由。Hybrid QMoE 的公开实现与正文读出形式分别标注；QRoute 可作为角度编码加计算基读出的参考。它们需要移植到同一专家字典、top-k、共享初始化和数据协议中，移植产生的改变必须记录。")

#text("第二，比较对象应包含强经典机制。除 MLP，还需要同编码器平方读出、结构化正交或低秩二次模型，并按目标加入 DSelect-k、反事实效用优化或几何路由。让量子分支和经典分支接收相同的额外监督，才能判断收益来自监督还是量子参数化。")

#text("第三，先固定专家识别机制，再共同训练观察分工形成。替换 probe 与候选集遗憾采用已有反事实分析思想；我们要验证新增结构是否带来额外收益。全部方法遵守同一调参预算与独立测试集。")

#text("第四，同时衡量选择质量与效率。参数数目、路由稳定性、PPL 和时延分别统计。输入舍入后的低切换率若伴随错误固定路由或专家闲置，应记录为失败；小参数线路若需要更长训练，应报告完整成本。")

#text("第五，要求机制证据。按真实算子关系设计的耦合，需要优于相同规模随机图；可分态与耦合态需要匹配编码器的对照；量子形式的收益需要与等价或近似经典函数族比较。新的稳定性定理也应指出条件、紧致程度及经典方法能否达到同样界。")

#heading(level: 2)[#text("6. 下一步决策")]

#strong[#text("继续研究，调整定位，并把最近方法对照放在大规模训练之前。")]#text(" 第一项仍可做算子效用诊断，但明确它是评估工具。随后只保留一个有具体机制的新候选，写出与 Hybrid QMoE、QRoute、EPO 和经典几何路由的逐项差异。")

#text("若差异仅剩门型、深度和任务名称，应先继续设计；若能够清楚提出新的算子关系约束、训练规则或稳定性与成本目标，再进行小规模公平试验。若新增收益被强经典基线解释，就记录为经典归纳偏置的结果；若量子结构在匹配条件下仍有稳定收益，才围绕该机制组织主要贡献。")

#text("本轮没有启动训练、运行外部研究代码或更改冻结实验文件。对外部仓库的核查为只读源码检查，尚未完成独立复现。本补充文件应与前一份报告一起阅读，其中关于新颖性边界的判断以本次复核为准。")

