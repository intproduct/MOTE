// Generated from the matching Markdown report. Edit the Markdown and rerun.
#set page(paper: "a4", margin: (x: 23mm, y: 22mm), numbering: "1")
#set text(font: ("Noto Serif SC", "Cambria"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.75em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 19pt, fill: rgb("17365d"))
#show heading.where(level: 2): set text(size: 14pt, fill: rgb("17365d"))
#show heading.where(level: 3): set text(size: 12pt)
#show link: set text(fill: rgb("24578a"))

#heading(level: 1)[#text("一手文献索引")]

#text("检索截至 2026-10-29。以下来源用于证据定位，不表示所有理论证明都已独立复核。主报告中的迁移判断是结合本仓库代码作出的分析；尤其不能把论文的小型分类实验当成本项目语言模型的实验结果。期刊链接遇到访问限制时，使用所附作者预印本；不同版本可能有细节差异。")

#heading(level: 2)[#text("持续学习、遗忘和稳定性")]

#enum(start: 1, [#text("Jiang、Lu、Deng，")#emph[#text("Quantum Continual Learning Overcoming Catastrophic Forgetting")]#text("，2021 预印本、2022 Chinese Physics Letters。")#link("https://arxiv.org/abs/2108.02786")[#text("预印本")]#text(" / ")#link("https://doi.org/10.1088/0256-307X/39/5/050303")[#text("期刊")]#text("。量子持续学习和参数保护，不能据题名推断没有遗忘。")])
#enum(start: 2, [#text("Zhang 等，")#emph[#text("Experimental demonstration of quantum continual learning with superconducting qubits")]#text("，npj Quantum Information，2026-01-06。")#link("https://www.nature.com/articles/s41534-025-01174-y")[#text("期刊")]#text(" / ")#link("https://arxiv.org/html/2409.09729")[#text("作者全文")]#text("。DOI 含 2025，实际发表日期为 2026；预印本题名为 ")#emph[#text("Quantum continual learning on a programmable superconducting processor")]#text("。重点核对无 EWC 对照及 Fisher/EWC 方法。")])
#enum(start: 3, [#text("Kirkpatrick 等，")#emph[#text("Overcoming catastrophic forgetting in neural networks")]#text("，PNAS 2017。")#link("https://arxiv.org/abs/1612.00796")[#text("预印本")]#text("。EWC 的经典基线来源。")])
#enum(start: 4, [#text("Chen、Zhang，")#emph[#text("Superior resilience to poisoning and amenability to unlearning in quantum machine learning")]#text("，Nature Communications，2026-03-09。")#link("https://www.nature.com/articles/s41467-026-70420-4")[#text("期刊")]#text(" / ")#link("https://arxiv.org/html/2508.02422v1")[#text("作者全文 v1")]#text("。污染与近似 unlearning；主报告的余弦 logit 分析对应预印本附录，不能直接替代全部实验。")])
#enum(start: 5, [#text("Yang、Xie、Xu，")#emph[#text("Stability and Generalization of Quantum Neural Networks")]#text("，2025 预印本。")#link("https://arxiv.org/abs/2501.12737")[#text("摘要")]#text(" / ")#link("https://arxiv.org/html/2501.12737v2")[#text("全文 v2")]#text("。区分算法稳定性、输入鲁棒性和优化稳定性；核对平滑性、步长和门数条件。")])
#enum(start: 6, [#text("Arjovsky、Shah、Bengio，")#emph[#text("Unitary Evolution Recurrent Neural Networks")]#text("，ICML 2016。")#link("https://proceedings.mlr.press/v48/arjovsky16.html")[#text("会议论文")]#text("。经典酉递归网络中的长期依赖，不是无状态路由中的灾难性遗忘。")])

#heading(level: 2)[#text("泛化与信息几何")]

#enum(start: 7, [#text("Caro 等，")#emph[#text("Generalization in quantum machine learning from few training data")]#text("，Nature Communications 2022。")#link("https://arxiv.org/abs/2111.05292")[#text("预印本")]#text(" / ")#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC9395350/")[#text("开放全文")]#text("。门数和有效变化参数与泛化界；不是普遍量子优势证明。")])
#enum(start: 8, [#text("Gil-Fuster、Eisert、Bravo-Prieto，")#emph[#text("Understanding quantum machine learning also requires rethinking generalization")]#text("，Nature Communications 2024。")#link("https://arxiv.org/abs/2306.13461")[#text("预印本")]#text(" / ")#link("https://www.nature.com/articles/s41467-024-45882-z")[#text("期刊")]#text("。拟合随机标签等现象，提醒不能以模型类别保证泛化。")])
#enum(start: 9, [#text("Abbas 等，")#emph[#text("The power of quantum neural networks")]#text("，Nature Computational Science 2021。")#link("https://arxiv.org/abs/2011.00027")[#text("预印本")]#text(" / ")#link("https://doi.org/10.1038/s43588-021-00084-1")[#text("期刊")]#text("。有效维度与选定网络上的训练结果；需要强基线与预算控制。")])
#enum(start: 10, [#text("Haug、Kim，")#emph[#text("Generalization of Quantum Machine Learning Models Using Quantum Fisher Information Metric")]#text("，PRL 2024。")#link("https://arxiv.org/abs/2303.13462")[#text("预印本")]#text(" / ")#link("https://doi.org/10.1103/PhysRevLett.133.050603")[#text("期刊")]#text("。数据相关 QFIM、对称性和可学习结构。")])
#enum(start: 11, [#text("Stokes、Izaac、Killoran、Carleo，")#emph[#text("Quantum Natural Gradient")]#text("，Quantum 2020。")#link("https://arxiv.org/abs/1909.02108")[#text("预印本")]#text(" / ")#link("https://quantum-journal.org/papers/q-2020-05-25-269/")[#text("期刊")]#text("。态空间度量及块近似；迁移时须核对输出可观测性和成本。")])
#enum(start: 12, [#text("Braunstein、Caves，")#emph[#text("Statistical distance and the geometry of quantum states")]#text("，PRL 1994。")#link("https://doi.org/10.1103/PhysRevLett.72.3439")[#text("期刊")]#text(" / ")#link("https://painterlab.caltech.edu/wp-content/uploads/2019/06/statistical_distance_geometry_quantum_states.pdf")[#text("作者机构托管 PDF")]#text("。量子态统计距离与测量信息。")])
#enum(start: 13, [#text("Barndorff-Nielsen、Gill，")#emph[#text("Fisher information in quantum statistics")]#text("。")#link("https://arxiv.org/abs/quant-ph/9808009")[#text("作者预印本")]#text("。经典测量 Fisher 与量子 Fisher 的关系，固定测量及多参数情形的限制。")])

#heading(level: 2)[#text("编码与结构")]

#enum(start: 14, [#text("Pérez-Salinas 等，")#emph[#text("Data re-uploading for a universal quantum classifier")]#text("，Quantum 2020。")#link("https://arxiv.org/abs/1907.02085")[#text("预印本")]#text("。重上传构造；普适性有具体架构条件。")])
#enum(start: 15, [#text("Schuld、Sweke、Meyer，")#emph[#text("Effect of data encoding on the expressive power of variational quantum-machine-learning models")]#text("，PRA 2021。")#link("https://arxiv.org/abs/2008.08605")[#text("预印本")]#text(" / ")#link("https://arxiv.org/html/2008.08605v2")[#text("全文")]#text("。编码、频率与表达能力，是设计周期经典对照的依据。")])
#enum(start: 16, [#text("Meyer 等，")#emph[#text("Exploiting Symmetry in Variational Quantum Machine Learning")]#text("，PRX Quantum 2023。")#link("https://arxiv.org/abs/2205.06217")[#text("预印本")]#text(" / ")#link("https://doi.org/10.1103/PRXQuantum.4.010328")[#text("期刊")]#text("。几何 QML/对称性方向；需与实际数据结构匹配。")])
#enum(start: 17, [#text("Stoudenmire、Schwab，")#emph[#text("Supervised Learning with Tensor Networks")]#text("，2016。")#link("https://arxiv.org/abs/1605.05775")[#text("预印本")]#text("。经典张量网络构造高阶特征的对照。")])
#enum(start: 18, [#text("Nguyen 等，")#emph[#text("QMoE: A Quantum Mixture of Experts Framework for Scalable Quantum Neural Networks")]#text("，2025 预印本。")#link("https://arxiv.org/abs/2507.05190")[#text("论文")]#text("。量子专家与路由相关先例；与大语言模型 MoTE 的设定不同。新颖性论证还需独立系统检索。")])

#heading(level: 2)[#text("限制与负面结果")]

#enum(start: 19, [#text("Bowles、Ahmed、Schuld，")#emph[#text("Better than classical? The subtle art of benchmarking quantum machine learning models")]#text("，2024 预印本。")#link("https://arxiv.org/abs/2403.07059")[#text("论文")]#text("。广泛的小型分类基准和去纠缠对照；提示公平基线的重要性。")])
#enum(start: 20, [#text("Cerezo 等，")#emph[#text("Cost function dependent barren plateaus in shallow parametrized quantum circuits")]#text("，Nature Communications 2021。")#link("https://arxiv.org/abs/2001.00550")[#text("预印本")]#text(" / ")#link("https://doi.org/10.1038/s41467-021-21728-w")[#text("期刊")]#text("。全局/局部损失及线路条件影响梯度；不能直接将定理套用于 MoE 完整损失。")])
#enum(start: 21, [#text("Thanasilp 等，")#emph[#text("Exponential concentration in quantum kernel methods")]#text("，Nature Communications 2024。")#link("https://arxiv.org/abs/2208.11060")[#text("预印本")]#text("。表达性、测量与噪声导致核集中等风险。")])
#enum(start: 22, [#text("Liu、Wittek，")#emph[#text("Vulnerability of quantum classification to adversarial perturbations")]#text("，Physical Review A 2020。")#link("https://arxiv.org/abs/1905.04286")[#text("预印本")]#text("。高维量子分类的对抗脆弱性。")])
#enum(start: 23, [#text("Lu、Duan、Deng，")#emph[#text("Quantum Adversarial Machine Learning")]#text("，Physical Review Research 2020。")#link("https://arxiv.org/abs/2001.00030")[#text("预印本")]#text("。量子分类器也可能受对抗样本影响。")])

#heading(level: 2)[#text("计算优势及其访问条件")]

#enum(start: 24, [#text("Huang 等，")#emph[#text("Power of data in quantum machine learning")]#text("，Nature Communications 2021。")#link("https://arxiv.org/abs/2011.01938")[#text("预印本")]#text(" / ")#link("https://www.nature.com/articles/s41467-021-22539-9")[#text("期刊")]#text("。任务与特征几何的匹配，以及经典数据对经典算法的帮助。")])
#enum(start: 25, [#text("Huang 等，")#emph[#text("Quantum advantage in learning from experiments")]#text("，Science 2022。")#link("https://www.preskill.caltech.edu/pubs/preskill-2022-quantum-advantage.pdf")[#text("作者 PDF")]#text("。学习量子实验中的访问条件及测量资源，与 GPU 路由模拟不同。")])
#enum(start: 26, [#text("Zhao 等，")#emph[#text("Exponential quantum advantage in processing massive classical data")]#text("，2026 预印本，v2 为 2026-10-01。")#link("https://arxiv.org/abs/2604.07639")[#text("摘要与版本")]#text("。本次仅核验摘要、版本和作者发布信息，没有完整复核证明；不将其机器规模优势改写为本项目实际速度或准确率优势。")])

