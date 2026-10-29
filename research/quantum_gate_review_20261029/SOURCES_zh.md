# 一手文献索引

检索截至 2026-10-29。以下来源用于证据定位，不表示所有理论证明都已独立复核。主报告中的迁移判断是结合本仓库代码作出的分析；尤其不能把论文的小型分类实验当成本项目语言模型的实验结果。期刊链接遇到访问限制时，使用所附作者预印本；不同版本可能有细节差异。

## 持续学习、遗忘和稳定性

1. Jiang、Lu、Deng，*Quantum Continual Learning Overcoming Catastrophic Forgetting*，2021 预印本、2022 Chinese Physics Letters。[预印本](https://arxiv.org/abs/2108.02786) / [期刊](https://doi.org/10.1088/0256-307X/39/5/050303)。量子持续学习和参数保护，不能据题名推断没有遗忘。
2. Zhang 等，*Experimental demonstration of quantum continual learning with superconducting qubits*，npj Quantum Information，2026-01-06。[期刊](https://www.nature.com/articles/s41534-025-01174-y) / [作者全文](https://arxiv.org/html/2409.09729)。DOI 含 2025，实际发表日期为 2026；预印本题名为 *Quantum continual learning on a programmable superconducting processor*。重点核对无 EWC 对照及 Fisher/EWC 方法。
3. Kirkpatrick 等，*Overcoming catastrophic forgetting in neural networks*，PNAS 2017。[预印本](https://arxiv.org/abs/1612.00796)。EWC 的经典基线来源。
4. Chen、Zhang，*Superior resilience to poisoning and amenability to unlearning in quantum machine learning*，Nature Communications，2026-03-09。[期刊](https://www.nature.com/articles/s41467-026-70420-4) / [作者全文 v1](https://arxiv.org/html/2508.02422v1)。污染与近似 unlearning；主报告的余弦 logit 分析对应预印本附录，不能直接替代全部实验。
5. Yang、Xie、Xu，*Stability and Generalization of Quantum Neural Networks*，2025 预印本。[摘要](https://arxiv.org/abs/2501.12737) / [全文 v2](https://arxiv.org/html/2501.12737v2)。区分算法稳定性、输入鲁棒性和优化稳定性；核对平滑性、步长和门数条件。
6. Arjovsky、Shah、Bengio，*Unitary Evolution Recurrent Neural Networks*，ICML 2016。[会议论文](https://proceedings.mlr.press/v48/arjovsky16.html)。经典酉递归网络中的长期依赖，不是无状态路由中的灾难性遗忘。

## 泛化与信息几何

7. Caro 等，*Generalization in quantum machine learning from few training data*，Nature Communications 2022。[预印本](https://arxiv.org/abs/2111.05292) / [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9395350/)。门数和有效变化参数与泛化界；不是普遍量子优势证明。
8. Gil-Fuster、Eisert、Bravo-Prieto，*Understanding quantum machine learning also requires rethinking generalization*，Nature Communications 2024。[预印本](https://arxiv.org/abs/2306.13461) / [期刊](https://www.nature.com/articles/s41467-024-45882-z)。拟合随机标签等现象，提醒不能以模型类别保证泛化。
9. Abbas 等，*The power of quantum neural networks*，Nature Computational Science 2021。[预印本](https://arxiv.org/abs/2011.00027) / [期刊](https://doi.org/10.1038/s43588-021-00084-1)。有效维度与选定网络上的训练结果；需要强基线与预算控制。
10. Haug、Kim，*Generalization of Quantum Machine Learning Models Using Quantum Fisher Information Metric*，PRL 2024。[预印本](https://arxiv.org/abs/2303.13462) / [期刊](https://doi.org/10.1103/PhysRevLett.133.050603)。数据相关 QFIM、对称性和可学习结构。
11. Stokes、Izaac、Killoran、Carleo，*Quantum Natural Gradient*，Quantum 2020。[预印本](https://arxiv.org/abs/1909.02108) / [期刊](https://quantum-journal.org/papers/q-2020-05-25-269/)。态空间度量及块近似；迁移时须核对输出可观测性和成本。
12. Braunstein、Caves，*Statistical distance and the geometry of quantum states*，PRL 1994。[期刊](https://doi.org/10.1103/PhysRevLett.72.3439) / [作者机构托管 PDF](https://painterlab.caltech.edu/wp-content/uploads/2019/06/statistical_distance_geometry_quantum_states.pdf)。量子态统计距离与测量信息。
13. Barndorff-Nielsen、Gill，*Fisher information in quantum statistics*。[作者预印本](https://arxiv.org/abs/quant-ph/9808009)。经典测量 Fisher 与量子 Fisher 的关系，固定测量及多参数情形的限制。

## 编码与结构

14. Pérez-Salinas 等，*Data re-uploading for a universal quantum classifier*，Quantum 2020。[预印本](https://arxiv.org/abs/1907.02085)。重上传构造；普适性有具体架构条件。
15. Schuld、Sweke、Meyer，*Effect of data encoding on the expressive power of variational quantum-machine-learning models*，PRA 2021。[预印本](https://arxiv.org/abs/2008.08605) / [全文](https://arxiv.org/html/2008.08605v2)。编码、频率与表达能力，是设计周期经典对照的依据。
16. Meyer 等，*Exploiting Symmetry in Variational Quantum Machine Learning*，PRX Quantum 2023。[预印本](https://arxiv.org/abs/2205.06217) / [期刊](https://doi.org/10.1103/PRXQuantum.4.010328)。几何 QML/对称性方向；需与实际数据结构匹配。
17. Stoudenmire、Schwab，*Supervised Learning with Tensor Networks*，2016。[预印本](https://arxiv.org/abs/1605.05775)。经典张量网络构造高阶特征的对照。
18. Nguyen 等，*QMoE: A Quantum Mixture of Experts Framework for Scalable Quantum Neural Networks*，2025 预印本。[论文](https://arxiv.org/abs/2507.05190)。量子专家与路由相关先例；与大语言模型 MoTE 的设定不同。新颖性论证还需独立系统检索。

## 限制与负面结果

19. Bowles、Ahmed、Schuld，*Better than classical? The subtle art of benchmarking quantum machine learning models*，2024 预印本。[论文](https://arxiv.org/abs/2403.07059)。广泛的小型分类基准和去纠缠对照；提示公平基线的重要性。
20. Cerezo 等，*Cost function dependent barren plateaus in shallow parametrized quantum circuits*，Nature Communications 2021。[预印本](https://arxiv.org/abs/2001.00550) / [期刊](https://doi.org/10.1038/s41467-021-21728-w)。全局/局部损失及线路条件影响梯度；不能直接将定理套用于 MoE 完整损失。
21. Thanasilp 等，*Exponential concentration in quantum kernel methods*，Nature Communications 2024。[预印本](https://arxiv.org/abs/2208.11060)。表达性、测量与噪声导致核集中等风险。
22. Liu、Wittek，*Vulnerability of quantum classification to adversarial perturbations*，Physical Review A 2020。[预印本](https://arxiv.org/abs/1905.04286)。高维量子分类的对抗脆弱性。
23. Lu、Duan、Deng，*Quantum Adversarial Machine Learning*，Physical Review Research 2020。[预印本](https://arxiv.org/abs/2001.00030)。量子分类器也可能受对抗样本影响。

## 计算优势及其访问条件

24. Huang 等，*Power of data in quantum machine learning*，Nature Communications 2021。[预印本](https://arxiv.org/abs/2011.01938) / [期刊](https://www.nature.com/articles/s41467-021-22539-9)。任务与特征几何的匹配，以及经典数据对经典算法的帮助。
25. Huang 等，*Quantum advantage in learning from experiments*，Science 2022。[作者 PDF](https://www.preskill.caltech.edu/pubs/preskill-2022-quantum-advantage.pdf)。学习量子实验中的访问条件及测量资源，与 GPU 路由模拟不同。
26. Zhao 等，*Exponential quantum advantage in processing massive classical data*，2026 预印本，v2 为 2026-10-01。[摘要与版本](https://arxiv.org/abs/2604.07639)。本次仅核验摘要、版本和作者发布信息，没有完整复核证明；不将其机器规模优势改写为本项目实际速度或准确率优势。
