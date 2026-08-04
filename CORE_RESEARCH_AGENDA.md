## Core Research Questions

All development and experiments in this repository must ultimately serve at
least one of the following scientific questions:

1. Whether input-dependent gating gives MOTE a substantive and reproducible
   capability advantage over MixT under controlled conditions.
2. What functional and optimization role the MOTE gate plays, especially
   whether it turns a static tensor approximation into an input-conditioned
   family of tensor operators.
3. Whether hierarchical gating, combining coarse-grained MoE routing with
   fine-grained MOTE routing, is feasible and scientifically meaningful.

Before proposing or implementing a major experiment, state:
- which research question it addresses;
- which alternative explanation it rules out;
- which paper claim it supports.

Engineering metrics and implementation details are evidence, not the research
objective.

# MOTE 项目核心研究纲领

本项目研究的核心不是单纯提高某个模型在某项评测上的分数，而是研究输入依赖 gate 对张量压缩神经网络表达能力、能力恢复和条件计算结构的影响。

项目始终围绕以下三个科学问题展开。

## 科学问题一：MOTE gate 是否带来实质能力提升？

相比采用固定张量组合的 MixT，MOTE 引入输入依赖 gate，根据 token 或上下文动态组合不同张量块。

本项目首先要回答：

在模型基座、压缩范围、张量结构、训练数据、训练步数、参数规模和计算量尽可能可比的条件下，MOTE 是否能够比 MixT 更好地保持或恢复模型能力，尤其是数学推理和多步推理能力？

这一问题要求验证：

* MOTE 相对 MixT 的提升是否具有足够大的效应量；
* 提升是否能在不同随机种子、模型规模和训练配置中重复；
* 提升是否不能被初始化、额外参数、训练稳定性或评测波动解释；
* 提升是否能够从 GSM8K 扩展到其他数学和推理任务。

科学问题一是整个项目和论文成立的基础。

## 科学问题二：MOTE gate 为什么有效？

在确认 MOTE 相对 MixT 存在稳定提升后，本项目进一步研究 gate 的实际作用机制。

核心假设是：

MixT 表示一个对所有输入共享的静态张量近似，而 MOTE 通过输入依赖路由，将多个受限张量块组织成输入条件化的张量算子族，使模型能够针对不同 token、上下文和推理阶段采用不同的局部计算结构。

围绕这一假设，需要研究：

* 动态输入依赖权重是否优于固定或全局可学习权重；
* 不同张量块是否形成了可检测的功能专门化；
* 路由是否对数字、运算符、中间推理和自然语言 token 作出不同选择；
* gate 是否改善了 dense FFN 激活的逼近能力；
* gate 是否改变了专家之间的梯度分配、更新方向和优化冲突；
* 打乱、替换或冻结路由是否会因果性地破坏模型性能。

科学问题二决定本项目是一个单纯的工程改进，还是一个具有机制解释的科学工作。

## 科学问题三：多层级 gate 是否可行且有意义？

在 MOTE 的细粒度张量块路由基础上，本项目进一步研究其能否与粗粒度 MoE 专家路由结合。

目标结构包括：

* 外层 MoE gate 负责专家或专家组之间的粗粒度功能选择；
* 内层 MOTE gate 负责专家内部张量块之间的细粒度动态组合。

需要回答：

* 两层 gate 是否分别承担不同尺度的条件计算功能；
* 多层级路由是否优于具有相同参数量和计算量的扁平路由；
* MoE 和 MOTE 的收益是简单相加、相互替代，还是具有正向交互；
* 多层级 gate 是否能在保持能力的同时进一步改善参数效率或计算效率；
* 外层和内层路由是否形成可解释的粗粒度与细粒度分工。

科学问题三是对前两个问题的推广，用于判断 MOTE 是否能够成为更普遍的层级条件计算单元。

# 项目执行原则

所有训练实验、代码开发、消融分析和评测工作都必须至少直接服务于上述一个科学问题。

每项新工作开始前，应回答：

1. 它主要回答三个科学问题中的哪一个？
2. 它排除了哪一种替代解释？
3. 它将为论文中的哪项结论提供证据？
4. 如果不完成这项工作，核心结论是否仍然成立？

无法明确回答这些问题的工作，应被视为辅助性工程任务，而不是项目主线。

训练 loss、路由熵、专家负载、参数量、FLOPs、Pass@K 和单项评测分数都只是证据，不是项目本身的研究目标。不能因为某个局部指标容易测量，就让项目偏离三个核心科学问题。

# 论文核心叙事

本项目最终希望论证：

固定张量分解主要提供静态、全局共享的低秩近似，而输入依赖 gate 能够将多个受限张量块组织成动态的条件算子族，从而提高张量压缩模型对复杂推理能力的保持和恢复能力。进一步地，细粒度 MOTE 路由可以与粗粒度 MoE 路由结合，形成具有不同功能尺度的层级条件计算结构。
