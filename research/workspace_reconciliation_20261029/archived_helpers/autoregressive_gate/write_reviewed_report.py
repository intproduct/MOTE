"""Finalize the measured Chinese interpretation; does not change training artifacts."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
raw=json.loads((ROOT/'raw_summary.json').read_text())
review=json.loads((ROOT/'REVIEW_METRICS.json').read_text())
draft=(ROOT/'review_history/automatic_20261028_1927/REPORT_zh.md').read_text(encoding='utf-8')
labels={'mlp':'MLP full','mlp_half':'MLP half','mpo_c2_full':'MPO full (χ=2)',
        'mpo_c1_half':'MPO half (χ=1)','amp_d1_full':'Born full','amp_d1_half':'Born half'}
a={x['variant']:x for x in raw['aggregate']}
rows={(x['variant'],x['seed']):x for x in raw['runs']}
report=draft.replace('**状态：训练与自动汇总完成，实测解释及图表人工复核待完成。**',
'''**状态：18/18 真实 GPU 训练、18/18 最佳 checkpoint 完整重评估、18/18 串行计时与报告复核全部完成。失败、缺失、未完成均为 0；两个独立配置的 pilot 各 6/6 通过，177 项相关测试通过。**

本轮在相同训练 token 下，半预算 MLP 的平均测试 PPL 最低（79.48），半预算幅度 Born 接近（79.84）；全预算 MLP 为 80.67。半预算 Born 相对全预算 MLP 平均降低 0.82 PPL，但相对半预算 MLP 高 0.36，不能据此声称 Born 特有优势。MPO 两个预算的平均 PPL 均约 81.2。

MLP 的训练效率最好。相对全预算 MLP，MPO 的串行每步耗时约增加 13–14%，Born 约增加 51%。半预算 Born 的路由参数减少 52.53%，整个模型参数仅减少 1.58%。这些结论只描述本轮小模型、语料和实现；三种子的配对区间仍宽。
''')
report=report.replace('formal\\9b3af7494afc','formal/9b3af7494afc')
report=report.replace('## 1. 协议和数据', '''## 1. 协议和数据

基础 Git commit：`c7d6690dfb13f2bf0a00ae145020ba7b0bb222fd`，分支 `mote-quantum`。本轮自回归模块为该 commit 上的未提交新增代码；本轮新增模块、配置与测试的源码快照在 `source_snapshot/`，既有路由等代码由上述基础 commit 提供。11 个训练依赖文件的实际 SHA256 在正式组 `frozen_sources.json`，18 份 manifest 记录相同哈希，最终复核与当前文件一致。仅记录基础 commit 不足以复现实验，须结合快照和哈希。

环境：NVIDIA TITAN V 12 GiB，驱动 582.28，PyTorch 2.6.0+cu118 / CUDA runtime 11.8，Python 3.11.18，NumPy 2.2.6，tokenizers 0.22.2，transformers 4.57.6，datasets 3.6.0。FP32、TF32 关闭、单 GPU worker、PyTorch CPU 线程数 1；完整依赖在 `dependencies.txt`。nvidia-smi 所示 CUDA 13.0 是驱动支持上限。

正式流水线从 2026-10-28 12:45:28 到 19:27:08（Asia/Shanghai），约 6 小时 42 分钟，包括训练、审计、重评估和自动制图；报告复核随后完成。18 组训练步耗时合计约 4.99 小时，逐 run 壁钟合计约 6.29 小时，另有统一重评估与汇总开销。保留了原先 5–6 小时估算及之后的 6.5–7 小时修正记录。
''')
report=report.replace('- 每个 run 4000 步，batch 32，共 16,384,000 训练 token；随机窗口有放回采样。',
'''- 每个 run 4000 步，batch 32，共 16,384,000 训练 token；随机窗口有放回采样，约为训练语料 token 数的 5.31 倍，并非完整顺序遍历 5.31 个 epoch。
- AdamW，峰值学习率 6e-4，warmup 200，cosine 衰减至 0.1 倍，weight decay 0.01，梯度裁剪 1.0，无 dropout；辅助路由 loss 系数 0.01，CE/PPL 不包含辅助 loss。各门共用这些超参数，未逐门调参。
- 保留仓库既有 hard top-4 + soft straight-through 梯度规则，不丢弃 token；只替换 MoTE 内的专家选择器，语言模型词表输出仍为共享的标准 softmax 交叉熵。''')
report=report.replace('- PPL=exp(总 NLL / 有效目标 token 数)。评估滑窗 context128/stride64，每个目标只计一次，padding不参与loss/路由。',
'''- PPL=exp(总 NLL / 有效目标 token 数)。评估滑窗 context128/stride64，除整个序列开头的 EOS 外，每个目标只计一次，padding 不参与 loss/路由。完整 validation 计 322,802 个目标，test 计 368,043 个目标。三种子 PPL 均值是逐 run 的 exp(CE) 再求均值，而非 exp(平均 CE)。
- 全部 18 个最佳 checkpoint 的 validation CE 重算值与保存值完全相同；test 只在全部正式训练结束后统一评估。''')
method='''
### 本轮比较的两种新 gate

MPO 是经典的非线性残差张量网络：每个 token 的特征经 LayerNorm、瓶颈线性投影与 tanh，再投影为 16 个特征，对应 4 个二值物理索引。在 depth=2 的层中依次执行 `state += scale * tanh(MPO(state))`，最后用 16→16 线性层读出专家 logits。每个 MPO 核为 `[χ左, 2输出, 2输入, χ右]`，两端 χ=1；实现先将核收缩成共享的 16×16 算子，再乘每个 token 的状态，没有截断状态 MPS。全/半预算分别使用 χ=2/1，并配合输入投影宽度匹配预算。χ=1 的单个 MPO 因子化，但整个含投影、非线性与残差的路由器不能简单等同于完全独立的四比特模型。

幅度 Born 先取 `code = tanh(P LayerNorm(x))`，生成 `amplitude = 1 + 0.5*tanh(V code)`，经 L2 归一化得到 16 维联合实数状态。depth=1 的演化包含四条线上数据依赖角 `π*tanh(A code)` 加可训练偏置的 RY，以及相邻对 (0,1)、(2,3) 上双向的可训练 CRY。演化为实数幺正变换，平方幅度并归一化得到 16 个专家的 Born 概率，随后以 ε=1e-6 平滑并输出 log 概率。公共 GateRouter 还会乘可训练正增益 g，再做温度 T=1 的 softmax，因此最终软路由概率正比于平滑 Born 概率的 `g/T` 次方，并非始终严格等于原始平方幅度概率；之后使用相同的 hard top-4 / soft STE 规则。它使用经典 GPU 模拟；联合幅度输入投影中的 rank 是经典瓶颈宽度，不是 MPS 的 χ。本轮没有直接运行量子硬件，也没有使用上一轮的 compact_born 或辅助状态 ResidualMPSGate。

全预算 MPO/Born 分别为全预算 MLP 路由参数的 101.44%/100.36%；半预算 MLP/MPO/Born 分别为 52.51%/49.45%/47.47%，整数宽度限制使预算近似匹配。depth 与 χ 同时随候选配置变化，因此全/半 MPO 的差异也不能单独归因于 χ。具体实现见 `advanced_routers.py`，本轮组装见 `experiments/autoregressive_gate/model.py` 与配置文件。
'''
# Both MPO variants actually share depth=2; only chi and projection width differ.
method=method.replace('depth 与 χ 同时随候选配置变化，因此全/半 MPO 的差异也不能单独归因于 χ。',
                      '两个 MPO 都是 depth=2，但 χ 与输入投影宽度同时变化，因此其差异不能单独归因于 χ。')
report=report.replace('## 2. 最终测试集结果',method+'\n## 2. 最终测试集结果')
seed_table=['### 逐种子结果与方差','',
 '| Gate | seed17 PPL | seed29 PPL | seed43 PPL | CE样本方差 | PPL样本方差 | Next-token accuracy均值±SD |',
 '|---|---:|---:|---:|---:|---:|---:|']
for v in raw['config']['gates']:
    x=a[v]
    seed_table.append(f"| {labels[v]} | "+' | '.join(f"{rows[v,s]['test_ppl']:.4f}" for s in [17,29,43])+
                     f" | {x['test_ce_std']**2:.8f} | {x['test_ppl_std']**2:.5f} | {100*x['test_accuracy_mean']:.3f}% ± {100*x['test_accuracy_std']:.3f} pp |")
seed_table+=['','方差与标准差均按 n−1 计算；置信区间讨论使用同种子配对差，不能仅凭两组独立误差条重叠判断显著性。全部最佳步数均为 4000，标准差为 0。','']
report=report.replace('## 3. 同种子配对差值','\n'.join(seed_table)+'\n## 3. 同种子配对差值')
ppl_table=['### 配对 ΔPPL 的逐种子值','',
 '| Candidate | 对照 | seed17 | seed29 | seed43 | 均值±SD |','|---|---|---:|---:|---:|---:|']
for p in raw['paired_summary']:
    trip=[next(x for x in raw['paired'] if x['variant']==p['variant'] and x['baseline']==p['baseline'] and x['seed']==s)['delta_ppl'] for s in [17,29,43]]
    ppl_table.append(f"| {labels[p['variant']]} | {labels[p['baseline']]} | "+' | '.join(f'{d:+.3f}' for d in trip)+f" | {p['delta_ppl_mean']:+.3f} ± {p['delta_ppl_std']:.3f} |")
report=report.replace('## 4. 效率与路由','\n'.join(ppl_table)+'\n\n## 4. 效率与路由')
start,end=report.index('## 4. 效率与路由'),report.index('## 5. 实测解释')
eff=['## 4. 效率与路由','','以下均值±样本SD，n=3。训练显存和串行基准显存分别列出，均为 PyTorch peak allocated MiB，不包含整个桌面占用或 CUDA reserved memory。','',
 '| Gate | 训练步秒数 | run壁钟秒数 | 全程训练token/s | 训练峰值MiB |','|---|---:|---:|---:|---:|']
for v in raw['config']['gates']:
    x=a[v]
    eff.append(f'| {labels[v]} | '+' | '.join(f"{x[k+'_mean']:.1f} ± {x[k+'_std']:.1f}" for k in ['training_seconds','wall_seconds','training_tokens_per_second','peak_allocated_mb'])+' |')
eff+=['','| Gate | 串行每步ms | 串行token/s | 串行峰值MiB |','|---|---:|---:|---:|']
for v in raw['config']['gates']:
    x=a[v]
    eff.append(f"| {labels[v]} | {1000*x['serial_step_seconds_mean']:.2f} ± {1000*x['serial_step_seconds_std']:.2f} | {x['serial_tokens_per_second_mean']:.0f} ± {x['serial_tokens_per_second_std']:.0f} | {x['serial_peak_allocated_mb_mean']:.1f} ± {x['serial_peak_allocated_mb_std']:.1f} |")
eff+=['','训练步秒数排除验证和保存；run 壁钟包含它们。串行基准每个 checkpoint 预热 5 步、测量 20 步，包含 host batch transfer、forward、aux、backward、clip、AdamW；新优化器状态和权重副本在测量后丢弃，未改动 checkpoint。它衡量端到端训练吞吐，不是推理速度。GPU 同时承担 Windows 桌面，短基准存在系统时序噪声；未以同门的全/半预算之间 1–2% 小差异推导速度优势。','',
 '| Gate | soft路由熵/nat | hard负载CV | 未用专家/模块 |','|---|---:|---:|---:|']
for v in raw['config']['gates']:
    x=a[v]
    eff.append(f'| {labels[v]} | '+' | '.join(f"{x[k+'_mean']:.4f} ± {x[k+'_std']:.4f}" for k in ['entropy','load_cv','unused_experts'])+' |')
eff+=['','路由指标先在每个 run 的四个模块求均值，再计算三种子统计。test 的每个模块均没有完全未使用的专家；负载向量之和为 top-k=4，而非 1。滑窗重复上下文计入路由统计（每模块 736,000 次 token 访问），NLL 的 368,043 个目标不会重复。熵是每个 token 的 soft 分布熵，CV 是全测试过程的 hard 专家使用量不均衡度，两者不要求同步变化。','']
report=report[:start]+'\n'.join(eff)+'\n'+report[end:]
interpret='''## 5. 实测解释

**相同 token 下，尚未发现新 gate 的稳定性能优势。** 六组都学习到了下一词统计规律，validation CE 从约 8.37 降至约 4.37–4.39。全预算 MPO 与 Born 的平均测试 PPL 分别为 81.23 和 82.17，对照 MLP 为 80.67；这轮全预算更复杂的路由没有换来更好的平均效果。半预算 Born 为 79.84，比全预算 MLP 低 0.82，但有两个种子变好、一个变差，配对 ΔCE 的 95% t 区间为 [−0.04746, +0.02735]。所有相对 full/half MLP 的配对 CE 区间都跨 0；样本很少，这同时不能证明优势，也不能证明严格等价或非劣性。

**“比 full MLP 参数少、分数相近”在本轮成立，但需要 half MLP 这个对照。** 半预算 Born 使用 24,188 个路由参数，比 full MLP 少 52.53%，其整个模型只少 26,768 个参数（1.58%）。half MLP 已用 26,758 个路由参数达到平均 PPL 79.48，仍优于 Born half 的 79.84。Born half 对 half MLP 的逐种子 ΔPPL 为 −1.083、+1.890、+0.275，平均 +0.360；路由进一步节省 9.60%，总模型仅再省 0.1535%，每步耗时约增加 51.3%。因此更准确的结论是存在一个较小路由器的性能/开销折中候选，不能把相对 full MLP 的改善归因于 Born 或纠缠结构。

**MPO 的参数效率尚未转成整体优势。** 半预算 MPO 以 full MLP 49.45% 的路由参数达到平均 PPL 81.17，差 +0.50；对 half MLP 差 +1.69，三个种子中两个更差。full/half MPO 的均值接近（81.23/81.17），本轮增加 χ 与投影宽度没有带来可见收益。但这是联合配置比较，不能解释为 χ 永远无用。MPO 的三种子 PPL 方差比 MLP 小；只有 3 点，不能据此认定普遍更稳定。

**按时间计算，MLP 仍有明确的本机实现优势。** 完成相同 4000 步，MLP 的训练步耗时约 838–840 秒，MPO 约 954 秒，Born 约 1193–1206 秒。独立串行基准也一致：MLP 约 20.5k token/s，MPO 18.0–18.2k，Born 13.6k。Born 的每步耗时比 full MLP 增加约 51%，吞吐约低 34%，这两个百分比不是同一分母。减半路由参数没有显著减少本模型的总计算/显存；所有门训练峰值约 989–1003 MiB，主要模型结构未变。当前实现的多步小算子、张量收缩和模拟电路可能带来调度/访存开销，但本轮没有做内核 profiler，因此不能确认具体耗时归因，也不意味着其理论复杂度必然更差。

时间图的横坐标是同一训练步处的三种子平均训练耗时，纵坐标是该步的平均 validation CE；连接线用于阅读趋势，并非在每个相同秒数重新评估。为补充一个直接可复核的时间参照，以下列出**事后描述性**阈值：第一次已记录的完整 validation PPL ≤90（每500步评估，未做插值，不是精确首次达到时间）。这项阈值没有用于选模型或修改训练。

| Gate | 首次记录达到PPL≤90的步数（17/29/43） | 训练秒数均值±SD |
|---|---|---:|
'''
for d in review['derived']:
    interpret+=f"| {labels[d['variant']]} | {' / '.join(map(str,d['first_observed_ppl90_steps']))} | {d['first_observed_ppl90_seconds_mean']:.1f} ± {d['first_observed_ppl90_seconds_std']:.1f} |\n"
interpret+='''
**训练数值稳定，选模点仍在预算末端。** 全部 72,000 个训练步的总梯度范数有限且路由总梯度非零；每个 run 的全部可训练路由张量都有更新。保存的 best/last 共 36 个权重文件在 CPU 上逐张量检查有限值，重评估一致。没有 OOM、失败重试、隐藏异常或删除失败结果。本轮 18 个最佳 checkpoint 全在第4000步，后期验证 CE 仍有改善，说明结论是固定预算下的结果，不是各门充分收敛后的极限能力比较。

**路由更均衡不保证语言效果更好。** Born full 的平均 hard负载 CV 最低（0.373），但其平均 PPL 最高（82.17）；MPO 的 CV 约 0.782，也未出现全测试集完全未使用的专家。现有指标没有显示数值崩溃或整体专家失活，但不能排除局部 token/小批次的负载偏斜。早期或单批次的 unused 数不等价于最终全测试集专家失活。

本轮没有 matched-rank 的无 CRY/无演化对照，也没有只改变 MPO χ 的控制实验。联合幅度输入本身可能已经跨物理索引相关；depth1 的 CRY 连接不覆盖所有分区。因此不能从这些结果识别纠缠增益，更不能声称量子优势。若继续研究，应先固定投影宽度/预算，再独立变化 χ、depth 或关闭 CRY，并保留 half MLP；后续参数选择只能用 validation 或新语料，不能反复根据本次 test 调优后把它当成未见测试集。本轮没有据此追加训练。

'''
start,end=report.index('## 5. 实测解释'),report.index('## 6. 图表')
report=report[:start]+interpret+report[end:]
report=report.replace('## 6. 图表','''## 6. 图表

五张图已逐张检查标签、坐标、误差条与原始数据。学习曲线同时提供完整轨迹和后期放大面板，缩放已标注；阴影是种子样本 SD，不是置信区间。参数图使用同一物理横轴，未做断轴；配对图使用未校正 95% t 区间。
''')
report=report.replace('所有 checkpoint 均用相同三个固定提示生成，贪心解码最多64个新token；完整原文与token ID见 generation_samples.json，没有按流畅度筛选。样例用于检查能否自回归运行，不作为统计排名指标。',
'''所有 checkpoint 均用相同三个固定提示生成，贪心解码最多64个新token；54 条完整原文与 token ID 见 `generation_samples.json`，没有按流畅度筛选。逐条检查发现，各组均存在明显重复、主题漂移、WikiText 标题格式堆叠和不连贯内容。例：seed17、同一提示 “The city is located” 下，MLP full 和 MPO half 都反复生成 “in the city of the city”，Born full 反复生成 “the city’s north of the city” 一类模式；原始标点与拼写以 JSON 为准。这些样例说明自回归链路可运行，但尚不是高质量自由生成模型。

测试 CE/PPL 衡量真实上下文条件下的下一词预测；自由生成以自己的历史输出作上下文，两者不能互相替代。约 23–24% 的 next-token accuracy 也不能解释为回答准确率。这里未做温度/top-p 搜索、指令微调或人工评判排名。小模型、短上下文、有限训练和贪心策略可能共同影响生成质量，当前设计无法把其原因单独归于 gate。''')
report=report.replace('- curves.csv / routing.csv / raw_summary.json / generation_samples.json。',
'''- curves.csv / routing.csv / raw_summary.json / generation_samples.json：原始测量保留。
- sample_variances.csv / validation_thresholds.csv / REVIEW_METRICS.json：事后描述性补充，未改变原始测量。
- REVIEW_AUDIT.json / COMPLETION_AUDIT.json：独立数值、文件、矩阵和交付核验。
- review_history/automatic_20261028_1927/：自动报告、初版图与 analysis.py 的复核前版本。
- final_review.py / write_reviewed_report.py：本次只读核验与报告构建脚本；VISUAL_REVIEW.json 记录逐图检查。''')
report+='''
## 9. 完成情况与未解决的限制

训练 18/18、完整 validation/test 重评估 18/18、串行基准 18/18、生成样例 54/54，全部完成；缺失/失败/未完成清单为空。无额外全量重训，无覆盖旧 run。报告修订只增加解释、样本方差和阈值统计，绘图修改只增加后期放大视图、提高颜色可读性和调整标签位置，训练代码/配置与测量结果未改。独立核验脚本初版使用了不同的 JSON 空格序列化方式而触发 hash 断言；已改为与冻结 runner 一致的格式后核验通过，这是复核脚本问题，原配置与训练文件无差异。

仍未解决的是研究范围：仅 3 个种子和一个小英文语料、所有最优点仍在训练终点、生成质量有限、无独立纠缠消融、无内核性能剖析，以及串行短基准受桌面系统噪声影响。没有未修复的执行错误。`last.pt` 不包含 optimizer/RNG，不能直接用它声称无缝续训。模型、数据和训练产物保留在 outputs 中，不加入 Git。
'''
assert 'REVIEW_REQUIRED' not in report and '待复核填写' not in report
(ROOT/'REPORT_zh.md').write_text(report,encoding='utf-8')
print('Reviewed Chinese report written:',len(report),'characters')
