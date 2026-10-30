#set page(paper: "a4", margin: (x: 22mm, y: 19mm), numbering: "1")

#set text(font: ("Times New Roman", "KaiTi"), size: 10.5pt, lang: "zh")

#set par(justify: true, leading: 0.65em)

#set heading(numbering: none)

#show heading.where(level: 1): set text(size: 19pt, fill: rgb("17365d"))

#show heading.where(level: 2): set text(size: 13pt, fill: rgb("17365d"))

#show math.equation: set text(font: "New Computer Modern Math", size: 10.5pt)

#heading(level: 1)[#text("补充材料：协议、逐种子数据与核验")]

#text("对应《MoTE 功能几何与 QuantumGate：阶段审校报告》。主报告负责结论与机制，本补充负责可复核的数值。完整原始结果以本地 CSV／JSON 和 checkpoint 为准。")

#heading(level: 2)[#text("A．协议与样本")]

#text("三个原始 half-MLP 语言模型 checkpoint 对应种子 17／29／43。冻结最后一个 MoTE down 模块及其上游模型，观察 16 个专家、top-4 选择。每个上下文长度 128，只评估位置 31、63、95、127。训练 token 数组前 75% 与后 25% 分别用于拟合和内部开发，窗口和目标不跨分割边界。每种子拟合 32,768 个位置，内部开发与 validation 各 4,096 个位置。")

#text("效用教师为 realized-label CE 对隐藏状态梯度与专家输出的内积，经专家维度中心化。有限干预从原始已选四专家中移除一个，把其固定质量交给十二个未选专家之一，共 48 个候选。阈值在内部开发集选择，包括保留原始路由；精确重加权的其他协议另行评价。")

#text("非线性模型固定 4,000 步、batch 256、AdamW 学习率 0.001、weight decay 0.0001、梯度裁剪 1。每 200 步在内部开发集选择最低 MSE checkpoint。每个种子所有模型共用采样顺序。确定性正向控制继承相同初始化和采样。区间为每种子按窗口重采样，保留同窗口位置与候选之间的配对；未做多重比较修正。")

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 5, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("seed")]],
[#strong[#text("dev CE")]],
[#strong[#text("dev BPE PPL")]],
[#strong[#text("validation CE")]],
[#strong[#text("validation BPE PPL")]],
[#text("17")],
[#text("4.0954")],
[#text("60.0605")],
[#text("4.3481")],
[#text("77.3282")],
[#text("29")],
[#text("4.0651")],
[#text("58.2732")],
[#text("4.3278")],
[#text("75.7764")],
[#text("43")],
[#text("4.0497")],
[#text("57.3826")],
[#text("4.3341")],
[#text("76.2587")])]

#text("以上是抽样位置的基线均值。PPL=exp(平均 CE)，词表为训练集内 BPE；该表的度量单位与完整 word-level perplexity 不同。多数校准策略保留原始路由，因此抽样 CE/PPL 与基线相同。")

#pagebreak()

#heading(level: 2)[#text("B．效用学习与实际干预")]

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 5, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("方法")]],
[#strong[#text("参数")]],
[#strong[#text("dev R²：17")]],
[#strong[#text("29")]],
[#strong[#text("43")]],
[#text("mlp")],
[#text("10385")],
[#text("-0.001778")],
[#text("-0.002074")],
[#text("-0.001675")],
[#text("spectral_functional")],
[#text("10277")],
[#text("-0.001735")],
[#text("-0.001763")],
[#text("-0.001858")],
[#text("spectral_random")],
[#text("10277")],
[#text("-0.001949")],
[#text("-0.000623")],
[#text("-0.001150")],
[#text("shared_born")],
[#text("10376")],
[#text("-0.001934")],
[#text("-0.001729")],
[#text("-0.001669")],
[#text("angle_natural")],
[#text("10285")],
[#text("-0.001227")],
[#text("-0.001191")],
[#text("-0.001229")],
[#text("angle_random")],
[#text("10285")],
[#text("-0.001213")],
[#text("-0.002271")],
[#text("-0.002151")],
[#text("angle_functional")],
[#text("10285")],
[#text("-0.001300")],
[#text("-0.001014")],
[#text("-0.002041")],
[#text("product_functional")],
[#text("10285")],
[#text("-0.001281")],
[#text("-0.001054")],
[#text("-0.001968")],
[#text("exp_adapter")],
[#text("10365")],
[#text("-0.000995")],
[#text("-0.000966")],
[#text("-0.000191")])]

#text("R² 以训练集专家均值作为常数对照。全部非线性候选在内部开发集与 validation 上均为负。除 exp_adapter 的 seed17 在第 400 步选中，其余均在第 200 步选中；训练更久没有改善内部开发 MSE。选中 checkpoint 的训练集 R² 多为很小的正值。")

#heading(level: 2)[#text("期望值适配器的 validation 干预：每种子仅一次")]

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 4, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("seed")]],
[#strong[#text("ΔCE")]],
[#strong[#text("95% 配对区间")]],
[#strong[#text("有害次数 ΔCE>0.1")]],
[#text("17")],
[#text("+0.0000316")],
[#text("[+0.0000000, +0.0000948]")],
[#text("1")],
[#text("29")],
[#text("-0.0000480")],
[#text("[-0.0001439, +0.0000000]")],
[#text("0")],
[#text("43")],
[#text("-0.0000843")],
[#text("[-0.0002528, +0.0000000]")],
[#text("0")])]

#text("其他八类非线性模型全部保留原始路由，ΔCE 与配对区间为零。相对于允许不替换的标签 oracle，三个种子的 validation 选择遗憾约为 0.285／0.293／0.273 nat。标签 oracle 使用推理时不可见的目标标签，作为反事实参照。")

#heading(level: 2)[#text("强线性控制")]

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 4, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("秩")]],
[#strong[#text("dev R²：17")]],
[#strong[#text("29")]],
[#strong[#text("43")]],
[#text("4")],
[#text("-0.000039")],
[#text("0.000001")],
[#text("0.000278")],
[#text("8")],
[#text("-0.000031")],
[#text("0.000021")],
[#text("0.000077")],
[#text("16")],
[#text("-0.000027")],
[#text("0.000049")],
[#text("0.000069")])]

#text("秩与正则候选均预先固定，正则只用内部开发集选择。个别校准阈值在 validation 产生少量有害替换；逐次行动和区间保存在 utility_and_ridge.csv。")

#pagebreak()

#heading(level: 2)[#text("C．确定性原 gate 教师：逐种子结果")]

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 5, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("方法")]],
[#strong[#text("seed")]],
[#strong[#text("dev R²")]],
[#strong[#text("validation R²")]],
[#strong[#text("validation top-4")]],
[#text("mlp")],
[#text("17")],
[#text("0.9756")],
[#text("0.9761")],
[#text("39.87%")],
[#text("mlp")],
[#text("29")],
[#text("0.9871")],
[#text("0.9877")],
[#text("46.92%")],
[#text("mlp")],
[#text("43")],
[#text("0.9841")],
[#text("0.9845")],
[#text("48.00%")],
[#text("angle_functional")],
[#text("17")],
[#text("0.3816")],
[#text("0.3848")],
[#text("1.51%")],
[#text("angle_functional")],
[#text("29")],
[#text("0.3011")],
[#text("0.3034")],
[#text("0.17%")],
[#text("angle_functional")],
[#text("43")],
[#text("0.4175")],
[#text("0.4217")],
[#text("7.32%")],
[#text("product_functional")],
[#text("17")],
[#text("0.2404")],
[#text("0.2304")],
[#text("1.56%")],
[#text("product_functional")],
[#text("29")],
[#text("0.2971")],
[#text("0.2990")],
[#text("0.12%")],
[#text("product_functional")],
[#text("43")],
[#text("0.1955")],
[#text("0.1971")],
[#text("0.02%")])]

#text("validation 三种子的 R² 均值±样本 SD：MLP 0.9828±0.0060，四角 Born 0.3700±0.0605，乘积 Born 0.2422±0.0520。集合一致率均值±SD 为 44.93%±4.42%、3.00%±3.80%、0.57%±0.86%。三种结构的训练参数分别为 10,385／10,285／10,285。")

#heading(level: 2)[#text("相对 MLP 的 validation 配对 R² 差")]

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 4, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("方法")]],
[#strong[#text("seed")]],
[#strong[#text("差值")]],
[#strong[#text("窗口 95% 区间")]],
[#text("angle_functional")],
[#text("17")],
[#text("-0.5913")],
[#text("[-0.6232, -0.5602]")],
[#text("product_functional")],
[#text("17")],
[#text("-0.7457")],
[#text("[-0.7865, -0.7047]")],
[#text("angle_functional")],
[#text("29")],
[#text("-0.6843")],
[#text("[-0.7141, -0.6557]")],
[#text("product_functional")],
[#text("29")],
[#text("-0.6887")],
[#text("[-0.7174, -0.6615]")],
[#text("angle_functional")],
[#text("43")],
[#text("-0.5627")],
[#text("[-0.5795, -0.5471]")],
[#text("product_functional")],
[#text("43")],
[#text("-0.7873")],
[#text("[-0.8125, -0.7656]")])]

#text("区间按窗口重采样误差差值，并用固定训练均值基线的 MSE 分母归一化。top-4 的逐种子配对差及区间在 positive_paired.csv。此教师校准的输出为中心化负对数概率；全局可学习缩放参与拟合。")

#pagebreak()

#heading(level: 2)[#text("D．功能风险与专家对增量")]

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 5, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("seed／split")]],
[#strong[#text("平均 KL")]],
[#strong[#text("Fisher ρ")]],
[#strong[#text("Fisher RMSE")]],
[#strong[#text("绝对误差 P99")]],
[#text("17/dev")],
[#text("0.0239")],
[#text("0.9602")],
[#text("0.0301")],
[#text("0.0940")],
[#text("17/validation")],
[#text("0.0237")],
[#text("0.9622")],
[#text("0.0317")],
[#text("0.0919")],
[#text("29/dev")],
[#text("0.0216")],
[#text("0.9631")],
[#text("0.0239")],
[#text("0.0879")],
[#text("29/validation")],
[#text("0.0221")],
[#text("0.9648")],
[#text("0.0234")],
[#text("0.0848")],
[#text("43/dev")],
[#text("0.0195")],
[#text("0.9618")],
[#text("0.0271")],
[#text("0.1215")],
[#text("43/validation")],
[#text("0.0190")],
[#text("0.9648")],
[#text("0.0271")],
[#text("0.1166")])]

#text("每行 4,096 个位置×48 个候选。平均 KL 和误差使用全部候选；ρ 先在固定源专家的十二个目标内计算再平均。相关度测量排序，误差测量有限步近似偏差；两者需要一起解释。")

#heading(level: 2)[#text("内部开发集：完整距离减能量对照的配对 ρ 差")]

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 4, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("尺度／gauge")]],
[#strong[#text("seed")]],
[#strong[#text("差值")]],
[#strong[#text("95% 区间")]],
[#text("raw_local")],
[#text("17")],
[#text("+0.02580")],
[#text("[+0.02391, +0.02785]")],
[#text("raw_global")],
[#text("17")],
[#text("-0.00134")],
[#text("[-0.00225, -0.00048]")],
[#text("centered_local")],
[#text("17")],
[#text("+0.02580")],
[#text("[+0.02388, +0.02770]")],
[#text("centered_global")],
[#text("17")],
[#text("-0.00136")],
[#text("[-0.00220, -0.00058]")],
[#text("raw_local")],
[#text("29")],
[#text("+0.03444")],
[#text("[+0.03233, +0.03654]")],
[#text("raw_global")],
[#text("29")],
[#text("+0.00248")],
[#text("[+0.00162, +0.00341]")],
[#text("centered_local")],
[#text("29")],
[#text("+0.03396")],
[#text("[+0.03188, +0.03600]")],
[#text("centered_global")],
[#text("29")],
[#text("+0.00341")],
[#text("[+0.00256, +0.00432]")],
[#text("raw_local")],
[#text("43")],
[#text("+0.03175")],
[#text("[+0.02963, +0.03380]")],
[#text("raw_global")],
[#text("43")],
[#text("+0.00222")],
[#text("[+0.00122, +0.00323]")],
[#text("centered_local")],
[#text("43")],
[#text("+0.03067")],
[#text("[+0.02868, +0.03258]")],
[#text("centered_global")],
[#text("43")],
[#text("+0.00078")],
[#text("[-0.00013, +0.00172]")])]

#text("raw 为原始专家输出，centered 为减去同输入所有专家均值后的输出。全距离对公共输出平移不变；能量与交叉项的分解依赖这个 gauge。两个 gauge 都显示局部增量；静态全局增量更小，部分区间触零或为负。")

#pagebreak()

#heading(level: 2)[#text("E．输入条件化几何与数值边界")]

#figure(image("stage_figures/geometry.png", width: 100%, height: 65mm, fit: "contain"), caption: [#text("提高线性秩改善输出草图拟合；预测的专家距离排序仍低于训练集全局距离。内部开发集，三个种子。")])

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 4, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("风险代理：dev ρ")]],
[#strong[#text("seed17")]],
[#strong[#text("seed29")]],
[#strong[#text("seed43")]],
[#text("输入草图欧氏距离")],
[#text("0.3080")],
[#text("0.2558")],
[#text("0.2710")],
[#text("无锚射线")],
[#text("0.0236")],
[#text("-0.0046")],
[#text("-0.0112")],
[#text("锚点×4射线")],
[#text("0.3073")],
[#text("0.2551")],
[#text("0.2702")],
[#text("训练全局距离")],
[#text("0.5286")],
[#text("0.4632")],
[#text("0.4883")])]

#text("固定锚点由训练集范数分布的第 95 百分位确定，锚点倍数预先指定；对应同信息经典核直接计算。原始八维随机草图是压缩参照，可能含估计噪声，不能当作任意预测器的上界。")

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 4, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("输入往返条件")]],
[#strong[#text("集合变化")]],
[#strong[#text("固定4/5跨界")]],
[#strong[#text("一阶 TP／FP／FN")]],
[#text("bf16_input_roundtrip")],
[#text("295")],
[#text("293")],
[#text("293/0/0")],
[#text("fp16_input_roundtrip")],
[#text("30")],
[#text("30")],
[#text("30/1/0")])]

#text("计数汇合了 24 个相关模型／拆分重放；每组 4,096 位置，不能视为同等数量的独立试验。BF16 两个漏出固定边界的案例分别为原排名 3→5 与 4→6。FP16 临界案例的 batch／FP64 复核保留在 boundary_exceptions_v1。")

#pagebreak()

#heading(level: 2)[#text("F．审计、预算与可复核路径")]

#text("独立阶段审计通过：27 个效用模型、9 个教师校准模型、18 个线性控制、6 组风险数组和 24 组边界结果齐全。风险数组的源／目标／窗口索引与原始有限干预一一对应，精确 NLL 重放一致；当前源文件、快照和冻结 checkpoint 哈希一致。缺失清单与失败清单为空。开发阶段的 smoke 和旧失败 pilot 独立保留，不计入这些正式数量。")

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 3, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("正向控制")]],
[#strong[#text("参数")]],
[#strong[#text("共享 GPU 秒／run（均值±SD）")]],
[#text("MLP")],
[#text("10385")],
[#text("31.69 ± 0.75")],
[#text("Four-angle Born")],
[#text("10285")],
[#text("156.54 ± 1.52")],
[#text("Product Born")],
[#text("10285")],
[#text("155.10 ± 2.93")])]

#text("此耗时包含训练与开发集 checkpoint 选择，记录于共享 TITAN V 环境；独立推理延迟、吞吐和显存对照将按串行协议另测。新 gate 的参数预算应计入它是否附加在原 gate 之后；效用替换策略若部署，需要原 gate 加预测器，而教师蒸馏替换 gate 的预算不同。")

#text("以下路径均相对于 D:/files/MOTE。完整数字保存在 outputs/operator_geometry_round2/reviews/stage_20261030_v1：utility_and_ridge.csv、positive_control.csv、positive_paired.csv、risk_metrics.csv、risk_paired.csv、geometry.csv、boundary.csv、sampled_ce_ppl.csv、utility_paired_vs_mlp.csv、SUMMARY.json 和 AUDIT.json。")

#block(width: 100%)[#set text(size: 9pt)
#table(columns: 2, inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), [#strong[#text("原始证据组")]],
[#strong[#text("路径（outputs/operator_geometry_round2/ 下）")]],
[#text("缓存与有限替换")],
[#text("data/86a2ba9f5aa2")],
[#text("效用／教师模型")],
[#text("utility/3a0386d9a187；positive/d4ee19e23787")],
[#text("强线性／几何")],
[#text("ridge/01c1341f7586；geometry/input_geometry_v1")],
[#text("能量／风险")],
[#text("energy/decomposition_v1；risk/3a17e2fefbb6")],
[#text("边界／例外")],
[#text("boundary_gradient/input_sensitivity_v1；reviews/boundary_exceptions_v1")],
[#text("数学检验")],
[#text("theory/real_state_top4_v1；theory/metric_bridge_v1")])]

#text("数学推导另见 GEOMETRY_METRIC_NOTE_zh.md／.typ／.pdf。该笔记保留了较早的进度截面；本阶段完成状态以本报告和 STAGE_COMPLETION_AUDIT.json 为准。后续独立结构校准位于 outputs/operator_geometry_round3，方案在 CAPACITY_PLAN.json，预算顺延记录在 round2/EXTENSION_STATE.json。")
