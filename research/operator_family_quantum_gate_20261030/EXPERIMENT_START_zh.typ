// Generated from the matching Markdown report. Edit the Markdown and rerun.
#set page(paper: "a4", margin: (x: 23mm, y: 22mm), numbering: "1")
#set text(font: ("Times New Roman", "KaiTi"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.75em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 19pt, fill: rgb("17365d"))
#show heading.where(level: 2): set text(size: 14pt, fill: rgb("17365d"))
#show heading.where(level: 3): set text(size: 12pt)
#show link: set text(fill: rgb("24578a"))

#heading(level: 1)[#text("功能几何引导量子路由：研究启动与首轮实测")]

#text("2026-10-30；工作目录 D:/files/MOTE；分支 mote-quantum")

#heading(level: 2)[#text("1. 结论：功能代价有信号，静态量子排列尚未通过扩大训练条件")]

#strong[#text("首轮真实 GPU 诊断和 27 项路由训练已经完成；继续研究有依据，当前静态量子排列方案不进入大规模训练。")]#text(" 专家替换产生可测且可预测的输出影响。控制被替换专家及其路由权重后，训练集上的平均专家输出距离仍对验证集替换后的分布 KL 有预测力。把这一关系用于量子测量基与专家的静态映射后，三个种子的语言损失变化方向不一致。功能信息的价值与具体量子结构的有效性，现阶段得到的是两种不同强度的证据。")

#text("本阶段的研究对象是条件专家替换代价。直接测量有符号的 next-token NLL 变化和预测分布 KL，再区分哪些几何量能预测这些变化。静态平均几何、输入相关几何、局部输出 Fisher 和 gate 分数间隔分别作为解释变量。量子结构的潜在价值要通过结构消融与经典功能几何对照评估。")

#text("诊断阶段完成 9/9 个 checkpoint，共 221,184 次单专家替换。新代码的 17 项测试和九类候选的 GPU 流程检查通过。独立 pilot 完成 27/27 项冻结专家路由训练、27/27 项保存后 checkpoint 重放及输入舍入复测；缺失与核验问题清单均为空。本报告完成了首个研究循环，扩大验证和端到端共同训练是后续研究阶段。")

#heading(level: 2)[#text("2. 文献已经回答什么，以及本次定位")]

#text("Nguyen 等的 QMoE 已提出可训练量子路由选择量子专家，并讨论路由与专家共同适应；其分类实验比较线路门集与专家数量。它覆盖总体架构和联合训练思想。本轮要检验的“用实测算子替换代价约束测量映射”不是该文的实验变量。")#link("https://arxiv.org/html/2507.05190v1")[#text("QMoE")]#text("。")

#text("Hybrid QMoE 已提出输入编码、幺正演化、Born 测量与经典专家组合，并用非线性分类任务讨论表达能力。本文将其路由形式作为近邻方案。它的主要实验问题是路由表达能力，而本阶段研究的是用已有专家的功能代价设计测量基与专家的对应关系。")#link("https://arxiv.org/html/2512.22296v1")[#text("Hybrid QMoE")]#text("。")

#text("反事实路由分析与 EPO 已研究冻结模型中的路由失配，并通过路由器更新利用更优专家组合。本项目采用反事实效用作为诊断依据；诊断与冻结专家训练本身属于已有研究方法。")#link("https://arxiv.org/html/2605.07260v1")[#text("When Are Experts Misrouted?")]#text("。路由器与专家沿输入方向共同积累梯度、以及用质心路由捕获这种耦合，也已有直接研究。")#link("https://arxiv.org/html/2605.12476v1")[#text("Routers Learn the Geometry of Their Experts")]#text("。")

#text("新补充的 Coherent Overlap 区分了候选质量、路由上下文与交互效应，并以冻结路由的输出干预分析功能价值。该工作直接支持在分析中分开报告几何重合和功能收益；其研究对象并非本文候选的量子测量配置。")#link("https://arxiv.org/html/2607.28308v1")[#text("Beyond Geometric Complementarity")]#text("。")

#text("ERMoE 明确提出把路由与专家基底绑定，需要纳入相关工作。其正文公式把两个向量同时投影到同一个完整正交基，再计算余弦相似度；按这一精确条件，余弦值对基底不变。我们已用数值测试核对这一恒等式。")#link("https://arxiv.org/html/2511.10971v1")[#text("ERMoE，公式 4–5")]#text("。进一步核查官方仓库中的 ERMoE.py：公开的二维图像实现实际计算注意力矩阵在 patch 特征基下的对角能量比例，这一评分能够依赖基底，与正文余弦式是两个不同的机制；其 route 函数还先执行全部 ViT 专家，再作选择。因而，该实现需要单独核算前置成本，不能直接充当本任务的低成本、专家执行前路由基线。本轮采用定义明确的固定 MDS 坐标作为经典功能几何对照。")#link("https://github.com/Belis0811/ERMoE/blob/master/ERMoE.py")[#text("官方实现")]#text("。")

#text("路由切换率与输出损害也已有分离证据。一项量化路由重放研究在 OLMoE 上发现，仅重放变化的路由复现了较小比例的量化损失；该干预结论具有具体模型和量化器条件。这支持同时测量切换次数与实际损失，而不单独优化前者。")#link("https://arxiv.org/html/2608.26612v1")[#text("Launch-Bound and Substitutable")]#text("。")

#strong[#text("截至此次核查，上述近邻工作尚未完成本项目这条具体链路：专家替换代价估计、据此配置量子测量与专家映射、与获得相同功能信息的经典方案对照、验证预测质量与有害切换的权衡。")]#text(" 这是当前要检验的研究假设。新颖性评估将随具体方法进一步收敛，不把“功能几何”“量子路由”这些宽泛表述作为首创点。")

#heading(level: 2)[#text("3. 已完成的诊断协议与结果")]

#text("复用 routing_stability_round1/formal/756b04397b2d 中 mlp_half、amp_half、born_lite 的种子 17、29、43，共 9 个 best checkpoint。严格加载，读取前后校验 checkpoint 哈希；所有模型权重均冻结，状态哈希在诊断前后保持一致。")

#text("只干预最后一个 down 模块 blocks.1.down。该模块之后只有残差合并、LayerNorm 与词表投影，因而可以缓存所有专家输出并精确计算单 token 的最终预测变化。每个 checkpoint 在 train 和 validation 各抽取 64 个不重叠窗口，每个窗口取位置 31、63、95、127，共 512 个 token。所有模型采用相同窗口与位置。每个 token 枚举 4 个选中专家到 12 个未选中专家的 48 次替换，保持被替换专家的权重不变，其余三项不变。")

#text("逐个比较局部计算与完整模型强制路由重放，最大误差阈值预先设为 0.00005。检查还覆盖概率质量守恒、top-k 数量、局部输出重建、解析 LayerNorm JVP 与自动微分一致、Fisher 的局部 KL 近似，以及零干预恒等性。")

#text("三个种子的平均结果如下。相关系数在同一 token、同一个被移除专家内部计算，从而固定替换权重；随后对这些条件组求平均。")

- #text("half MLP：训练集平均输出距离对替换 KL 的相关系数 0.509；gate 分数间隔为 0.307；局部输出 Fisher 为 0.961。利用真实下一个 token 在 48 个替换及原路由中选择的诊断 oracle，平均 NLL 变化为 -0.285。")
- #text("原幅度 Born：对应相关系数依次为 0.419、0.180、0.955；诊断 oracle 的平均 NLL 变化为 -0.317。")
- #text("共享 Born：对应相关系数依次为 0.574、0.428、0.972；诊断 oracle 的平均 NLL 变化为 -0.231。")

#text("这些结果说明：单纯的分数间隔不足以刻画替换后果；功能几何在跨 train/validation 的条件下仍保留信息。Fisher 衡量的是输出分布局部变化，具有预测标签的有符号 NLL 变化另行记录。静态全局几何的相关性弱于输入相关 Fisher，提示后续要检验静态量子基配置是否过于粗糙。")

#strong[#text("诊断 oracle 使用真实目标标签逐 token 选最优替换，是模型内部可用空间的探索指标。")]#text(" 可部署 gate 在推理时只能看到当前特征，能否从训练数据学到这一效用规律由下一阶段检验。窗口 bootstrap 的区间反映固定模型和该验证样本下的变异；三种子的变化单独呈现。既有 WikiText 验证与测试结果已在历史研究中观察过，本阶段不读取 test，也不将该验证集称为全新盲测。")

#heading(level: 2)[#text("4. 已部署的路由训练 pilot")]

#text("每个种子统一采用已有 half MLP 模型作为冻结骨干及专家池。只训练最后一个 down 模块的路由器；不同候选不再使用各自训练过的专家，以隔离路由结构差异。训练缓存来自 1,024 个 train 窗口，共 4,096 个 token；验证来自 128 个 validation 窗口，共 512 个 token。新采样种子固定为 20261031。")

#text("九类候选各运行种子 17、29、43，共 27 项：普通 MLP；基于功能距离 MDS 坐标的经典路由；坐标随机打乱的同结构经典对照；当前共享幅度 Born；自然专家编号的角编码 Born；随机编号的角编码 Born；功能编号的角编码 Born；功能编号的无纠缠乘积态；量子期望值加经典线性层及 softmax 的读出对照。")

#text("角编码候选采用 4 qubit、16 个基态、低秩输入投影、RY 编码及两组可训练 RY，中间为相邻 CZ。末尾紧接计算基测量的额外 CZ 不改变 Born 概率，因此不添加这一无效操作。期望值读出对照复用同一线路，隔离读出选择；它是受公开方案启发的控制实现，尚不是原论文整个实验的逐项复现。无纠缠方案与经典乘积 Bernoulli 分布的代数等价已经测试。")

#text("功能编号只用训练集的平均专家输出平方距离，通过固定预算的交换搜索降低超立方体相邻基态间的平均功能距离。相邻基态是单 qubit RY 直接混合的对象。该设计假设：让容易被这种操作混合的基态对应功能接近的专家，可能降低不必要的输出变化。训练可否利用这种配置、top-k 边界是否遵循同一关系，都由随机编号和无纠缠对照检验。")

#text("经典功能几何候选通过同一距离矩阵的 MDS 坐标构造固定读出，让经典和量子方案均可使用功能信息。所有方法采用 rank 8 输入瓶颈，报告实际参数量；相同 rank 并不等于参数逐个相同。推理阶段的映射或固定坐标均可预计算，只需要输入特征进行一次路由。")

#text("每项训练 800 步、batch 128、AdamW 学习率 0.001；前 400 步拟合原路由概率，后 400 步优化精确 next-token CE 加 0.05 倍原路由 KL。所有候选采用相同采样序列。训练使用离线缓存的全部专家输出提供同样的稠密反事实梯度；这与原始稀疏分支的梯度路径不同，属于本轮明确控制的训练协议。该方案受到稠密反向路由研究启发。")#link("https://proceedings.mlr.press/v262/panda24a.html")[#text("Dense Backpropagation Improves Routing")]#text("。")

#text("固定使用最后一步 checkpoint，不用验证集挑最佳步数。记录训练及验证 CE/PPL、路由负载与未用专家、梯度有限值、参数更新、原始逐 token NLL、采样序列、checkpoint、优化器和随机状态。保留完整原始结果，尚未实现的续训接口不会自动使用保存状态。")

#heading(level: 2)[#text("5. 判断标准、成本和后续分支")]

#text("诊断阶段预先采用“至少两个种子的窗口区间显示 oracle 平均改善超过 0.02 nat”作为小规模 pilot 的信号门槛，三类旧 gate 均为三个种子通过。该门槛只允许继续测量可学习性。")

#text("训练 pilot 的扩展条件已经写入冻结配置：功能排列在三个种子上均优于相同角编码的随机排列；相对经典功能几何对照 CE 差不超过 0.01 nat；同时改善 BF16 输入 roundtrip 下的有害事件。若条件不满足，保留负结果，判断是排列假设无效、静态距离不足还是输入编码表达受限，再决定修订方向。不因单个种子或更低切换率直接扩大训练。")

#text("输入 roundtrip 只将 router 输入转换为低精度再回到 FP32；原生低精度内核、跨推理引擎与分布式条件需要后续独立协议。pilot 只改变最后一层路由，其 CE/PPL 目前是固定采样点上的指标；完整 validation 重评估、真正的稀疏执行效率、端到端专家共适应和新测试任务属于下一轮确认工作。")

#text("本机 TITAN V 同时承载其他桌面程序及 llama-server。实验设置单进程、单 CPU 线程、CUDA 分配上限 35%，不结束其他进程。记录的运行时间用于调度估算，不能用于孤立硬件效率结论。后续方法比较需要在可控资源条件下串行计时，并计入离线功能估计与训练成本。")

#heading(level: 2)[#text("6. 文件与复现入口")]

#text("实现目录：experiments/operator_geometry。诊断配置：configs/diagnostic_v1.json；训练配置：configs/router_pilot_v1.json。runner.py 执行冻结干预；analyze.py 进行原始数据校验及控制替换质量后的分析；routers.py 定义控制候选与训练集几何；pilot.py 执行路由器训练。tests/test_operator_geometry.py 包含数学及流程测试。")

#text("输出根目录：outputs/operator_geometry_round1。诊断组：diagnostic/2c26c073e9a5。每个 run 包含 train/validation 缓存、全部替换 NPZ、训练集距离、身份检查和结果。diagnostic_summary.csv 与 DIAGNOSTIC_AUDIT.json 对应本报告首轮实测。训练组按配置哈希独立创建，源码快照、原始曲线和失败现场均保留。")

#text("所有入口使用本工作区 .venv-titan/python.exe，通过 fitmotn.experiments.operator_geometry 的模块路径执行。锁文件防止本任务重复启动；发现已有不完整组或 run 时直接停止，不覆盖历史数据。旧分类、自回归、路由稳定性源码与配置保持原样。")

#heading(level: 2)[#text("7. 27 项训练的实测结果与决策")]

#strong[#text("静态功能排列未达到预设的三种子一致改善标准。")]#text(" 功能排列减去随机排列的配对验证 CE 差，在种子 17、29、43 上依次为 -0.00807、-0.00281、+0.01405 nat/token。对应的窗口 bootstrap 95% 区间依次为 [-0.02876, 0.01303]、[-0.02040, 0.01586]、[-0.00306, 0.03045]，均跨过零。三个种子平均差为 +0.00106。原始采样验证集上，功能角编码也比本轮普通 MLP 路由平均高 0.00893 nat/token。")

#text("以下为固定第 800 步的三个种子均值 ± 样本标准差。CE 单位为 nat/token；PPL 为每个种子 sampled BPE PPL 的算术均值及标准差。每个种子只有 512 个验证 token，不能将其视作完整 validation 评估。参数为实际可训练 router 参数量。")

- #text("普通 MLP：10,384 参数；CE 4.39570 ± 0.03495；PPL 81.13 ± 2.83。")
- #text("经典功能几何：10,276 参数；CE 4.40901 ± 0.03542；PPL 82.22 ± 2.91。")
- #text("经典随机几何：10,276 参数；CE 4.43459 ± 0.05276；PPL 84.40 ± 4.50。")
- #text("共享幅度 Born：10,375 参数；CE 4.41378 ± 0.02824；PPL 82.60 ± 2.34。")
- #text("自然排列角编码 Born：10,284 参数；CE 4.40250 ± 0.03652；PPL 81.69 ± 3.00。")
- #text("随机排列角编码 Born：10,284 参数；CE 4.40357 ± 0.04498；PPL 81.80 ± 3.65。")
- #text("功能排列角编码 Born：10,284 参数；CE 4.40463 ± 0.03562；PPL 81.86 ± 2.91。")
- #text("功能排列乘积态：10,284 参数；CE 4.40645 ± 0.03926；PPL 82.02 ± 3.22。")
- #text("期望值与 adapter：10,364 参数；CE 4.39944 ± 0.03963；PPL 81.45 ± 3.25。")

#strong[#text("经典对照提供了功能结构有用的初步证据。")]#text(" 经典功能几何相对随机标签几何在三个种子上均较好，CE 差为 -0.00489、-0.05081、-0.02105，平均 -0.02558。只有种子 29 的本次窗口区间完全低于零。该对照说明专家身份与几何关系的对应可以影响结果，值得保留为后续必须战胜的基线。")

#strong[#text("当前候选与原有路由之间仍存在适配差距。")]#text(" 三个种子原始 router 的平均 sampled CE 是 4.39118；九类新候选的平均 CE 均高于它。既有专家池与原始路由已经共同训练，新候选只在 4,096 个缓存 token 上接受短训练，因此该结果同时约束了新结构和当前适配协议。曲线显示种子 17 的多种候选在转入 CE 训练后验证损失上升，种子 43 的部分候选仍在改善。下一轮需要用训练集内部划分诊断泛化和训练预算，固定新的评估协议后再比较，避免按本次验证曲线挑步数。")

#text("功能角编码相对无 CZ 的乘积态，三个 CE 差为 -0.00076、-0.00602、+0.00132；各区间跨零。相对经典功能几何的差为 -0.00496、-0.00387、-0.00430；同样没有区间完全低于零。现有数值支持继续比较这几种结构，尚未形成稳定的耦合结构收益。")

#strong[#text("稳定性事件太少，需更大的固定测量集。")]#text(" 合并三种子共 1,536 个 token，功能角编码有 5 次 top-k 集合变化、0 次 NLL 增加超过 0.1 的事件；随机角编码为 2 次变化、1 次有害事件。两者都出现了稀有事件，小样本中的零次观测不能确定较低的总体风险。功能角编码的未用专家数为 0、0、1，负载 CV 均值为 0.526；未出现此前固定 top-4 的退化。共享 Born 和期望值 adapter 的单种子未用专家最多为 4，作为负载问题保留记录。")

#strong[#text("本轮几何机制的作用范围较弱。")]#text(" 通过静态排列优化，超立方体边上的训练输出距离相对 128 个随机排列的平均值仅降低 1.92%、2.01%、2.26%。RY 对计算基的局部混合关系并不自动等同于输入扰动引起的 top-k 边界交换；这一对应需要直接测量。功能距离预测的是替换影响大小，改善标签 NLL 还需要带方向的效用信息。这两处是下一阶段优先检验的机制。")

#text("下一步先开展三个有明确判据的工作：测量真实边界交换是否集中在功能近邻；用训练集构建输入条件化的低秩功能代价预测器，并检验跨样本泛化；再把同一预测信息分别用于经典路由和量子结构。只有输入条件化信息能稳定降低候选选择遗憾，再考虑它如何约束测量或耦合。前两步用于验证设计依据，第三步才检验量子参数化的额外价值。当前单语料、末层冻结实验也尚未检验跨任务算子族形成，后续需要任务与域迁移对照。")

#heading(level: 2)[#text("8. 核验与结果索引")]

#text("27 个 run 的原始文件哈希、800 步有限值记录、非零梯度、参数更新和固定末步选择均通过检查。同一种子下九个候选使用相同 minibatch 索引；三种角编码排列使用相同可训练参数初始化。原专家 checkpoint、冻结源码及快照哈希一致；训练器在每个 run 后检查冻结模型状态。独立载入 27 个末步 checkpoint，重放逐 token NLL 和输入舍入事件，全部通过预设容差。新候选未访问 test 集，原始结果均保留。")

#text("训练组为 outputs/operator_geometry_round1/router_pilot/da5325087d6b。review_v1/AUDIT.json 记录 large_training_eligible=false；runs.csv、aggregate.csv、paired.csv 和 checkpoint_replay.json 分别提供逐项、三种子、配对及重放证据。review_pilot.py 是独立核验入口；plot_review.py 生成三张结果图。图中的跨种子标准差和单 checkpoint 窗口置信区间分别标注。")

#text("图 1 展示全部候选相对原路由的采样验证 CE；图 2 展示训练阶段切换前后的曲线；图 3 展示逐种子配对差和窗口 bootstrap 区间。三图已经逐张检查。PDF 附图与 Markdown 的 figures 目录使用相同文件。")


#pagebreak()
#set page(width: 297mm, height: 210mm, margin: 18mm)
#heading(level: 2)[附图 1]
#figure(image("operator_pilot_figures/validation_ce.png", width: 100%), caption: [#text("图 1：全部候选相对原路由的采样验证 CE；点为种子，横线为均值 ± 种子标准差。")], numbering: none)

#pagebreak()
#heading(level: 2)[附图 2]
#figure(image("operator_pilot_figures/curves.png", width: 100%), caption: [#text("图 2：验证曲线；第 400 步之后从模仿路由转为 CE 与教师 KL 联合训练。")], numbering: none)

#pagebreak()
#heading(level: 2)[附图 3]
#figure(image("operator_pilot_figures/paired_ce.png", width: 100%), caption: [#text("图 3：配对 CE 差及固定 checkpoint 的窗口 bootstrap 95% 区间；负值表示前者更好。")], numbering: none)
