"""Conclusion-first capacity/geometry report and separate numerical supplement."""
from pathlib import Path
import json,csv,ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
BASE=REPO/'outputs/operator_geometry_round3';FIG=HERE/'capacity_figures';FIG.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
new=read(BASE/'capacity/481aeb0bc8dd/raw_summary.json');old=read(REPO/'outputs/operator_geometry_round2/positive/d4ee19e23787/raw_summary.json')
rows=old+new;s=read(BASE/'replay_v2/ff4f146e8d29/summary.json');audit=read(BASE/'reviews/sparse_replay_v2/AUDIT.json')
methods=['mlp','affine_born','angle_functional','reencode_cz','reencode_product'];names=['MLP','Affine Born','Four-angle','Re-encode CZ','Product']
labels=dict(zip(methods,names));labels['product_functional']='Old product'
colors=['#355d89','#397e72','#a98b64','#bd6751','#7c77a3']
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,3,figsize=(10.3,3.0),layout='constrained')
for ax,key,title in zip(axes,['r2_vs_constant','top4_set_agreement','mean_kl'],['Teacher-score fit','Exact top-4 agreement','Frozen-model output change']):
    for j,m in enumerate(methods):
        v=[r['splits']['dev'][key] for r in rows if r['method']==m] if key!='mean_kl' else [r['quality'][key] for r in s['rows'] if r['method']==m and r['split']=='dev' and r['mode']=='calibrated']
        ax.bar(j,np.mean(v),width=.6,color=colors[j],alpha=.75);ax.scatter(j+np.array([-.1,0,.1]),v,s=15,color='#172b40',zorder=3)
    ax.set_xticks(range(5),names,rotation=30,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('R2 (higher is better)');axes[1].set_ylabel('Fraction');axes[2].set_ylabel('KL (lower is better)')
fig.savefig(FIG/'fidelity.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9.5,3.0),layout='constrained')
for j,m in enumerate(methods):
    for mode,dx,c in [('native',-.16,'#9aafc4'),('calibrated',.16,'#355d89')]:
        v=[r['quality']['mean_kl'] for r in s['rows'] if r['method']==m and r['split']=='dev' and r['mode']==mode]
        axes[0].bar(j+dx,np.mean(v),width=.3,color=c,label=mode if j==0 else None)
    t=[r['median_ms'] for r in s['timings'] if r['method']==m and r['mode']=='native' and r['rows']==4096]
    axes[1].bar(j,np.mean(t),width=.6,color=colors[j],alpha=.8);axes[1].scatter(j+np.array([-.1,0,.1]),t,s=18,color='#172b40')
for ax in axes:ax.set_xticks(range(5),names,rotation=25,ha='right');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Exact output KL');axes[0].legend(frameon=False);axes[1].set_ylabel('Router + top-4 time (ms / 4096 rows)')
fig.savefig(FIG/'calibration_time.png');plt.close(fig)

# Reuse only the document serializer's class definition; do not execute the old report builder.
tree=ast.parse((HERE/'build_stage_review.py').read_text(encoding='utf8'));node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Document')
src=ast.unparse(node).replace('stage_figures/','capacity_figures/')
exec(src,globals())
def mean_sd(v,fmt='.4f'):return f'{np.mean(v):{fmt}} ± {np.std(v,ddof=1):{fmt}}'
def quality(m,split='dev',mode='calibrated'):return [r for r in s['rows'] if r['method']==m and r['split']==split and r['mode']==mode]
def cap(m,split,key):return [r['splits'][split][key] for r in rows if r['method']==m]

d=Document('CAPACITY_GEOMETRY_REVIEW_20261030_zh')
d.title('QuantumGate：状态自由度增加后，功能保真仍是关键')
d.p('阶段审校报告｜2026-10-30。覆盖新增线路校准、冻结专家稀疏重放与量子几何核验。后续功能度量适配已按独立方案启动，其结果另行审校。')
d.h('主要结论：交错编码解除一部分表达约束，功能训练目标值得直接检验')
d.p('本轮找到了一条可解释的线路改进：把第二次数据编码放在 CZ 之后，能让原先重合的状态变化方向分开。新增线路在三个种子上都更好地拟合原 gate 的分数，接回同一组冻结专家后，模型输出偏移也小于旧角编码。量子几何因此已经给出了一个可验证的结构设计依据。')
d.p('当前最清楚的实际差距在专家组合的功能保真。校准后的 MLP 与经典幅度 gate 都能较好地保持原模型输出，交错 CZ 的偏移仍明显更大。下一轮把训练目标从“复现分数”移到“保持专家组合功能”，直接比较权重、专家输出和模型 Fisher 三种几何。')
d.figure('fidelity.png','同一内部 dev 上的三层证据：分数拟合、离散集合、最终输出。点为三个种子；右图使用预先规定的校准权重。',64)
d.p('分数与集合指标给出了不同判断。交错 CZ 的分数拟合在三个种子上均改善；相对旧角编码的 top-4 集合一致率在 seed43 下降。预先写下的“每个种子两项均改善”标准未全部满足，本轮保留全部对照，并转入独立的功能目标检验。')
d.table(['内部 dev／均值','旧角编码','交错 CZ','MLP','经典幅度'],[['相对原模型 KL',*[f'{np.mean([r["quality"]["mean_kl"] for r in quality(m)]):.4f}' for m in ['angle_functional','reencode_cz','mlp','affine_born']]],['主要解释','分数拟合受限','改善且仍有差距','功能保真较好','功能保真较好']])
d.page();d.h('几何解释：第二次编码是否提供新的有效方向')
d.p('考虑四量子比特实态线路，先分别施加 RY(a)，经过固定 CZ 链，再施加 RY(b)。两组角度都由输入特征产生。把 CZ 去掉时，同一比特上的两次旋转合并成 RY(a+b)；八个名义角度只对应四个独立方向。交错 CZ 改变了这种重合关系。')
d.math(r'|\psi(a,b)\rangle=R_y(b)\,CZ_{\rm chain}\,R_y(a)|0\rangle,\qquad F_{\rm product}=\begin{pmatrix}I&I\\I&I\end{pmatrix}',r'|psi(a,b) angle.r = R_(y) (b) "CZ"_"chain" R_(y) (a) |0 angle.r, quad F_"product" = mat(I,I; I,I)')
d.p('对乘积对照，(δa,−δa) 是 QFI 的零方向。对交错 CZ，解析式和独立自动微分一致，一般非退化角度处可出现八个独立方向。在第一组角度为 π/2 的工作点，八维角坐标 QFI 为单位矩阵；这为避免局部冗余提供了明确的初始化依据。')
d.table(['设计问题','本轮答案','与训练结果的联系'],[['增加名义参数是否增加方向','无 CZ 时两次 RY 合并','乘积对照的拟合接近旧乘积模型'],['CZ 放在哪里有意义','放在两次数据编码之间','新增可达方向伴随更好的分数与功能拟合'],['什么量决定专家组合风险','专家输出及其到模型分布的映射','下一轮直接训练这一映射的局部代价']])
d.p('这里的 QFI 描述量子态对角度变化的敏感性。完整、无概率地板的实态计算基测量在一般点保留该局部信息；实际路由还包含概率地板、输入网络和 top-4 选择，它们各自改变后续几何。训练中的参数方向由输入网络雅可比继续拉回，专家功能风险则由专家输出和语言模型头决定。')
d.p('经典幅度对照使用同一低秩输入特征生成十六个实幅度，再平方归一化。它表现接近 MLP，说明本轮观察到的主要瓶颈落在具体角编码流形和训练方式。线路的额外方向与实际训练收益有对应关系，量子和经典结构的工程价值还需在相同功能目标下继续比较。')
d.page();d.h('稀疏重放揭示：选哪些专家和怎样混合，必须分别核验')
d.p('对相同 top-4 集合，只改变混合权重的校准幂，就显著降低了 MLP 与经典幅度 gate 的输出偏移。校准后的结果对应“分数学习器加固定幂”的部署形式；原生概率的结果完整保留在补充材料。由此可见，集合一致率偏低时仍可能保住大部分功能，而权重形状改变也能影响模型输出。')
d.figure('calibration_time.png','左：原生与校准概率产生不同的专家混合权重；右：共享 TITAN V 上的串行 router+top-4 计时。',63)
d.p('当前 Python 状态向量实现的执行开销明显高于 MLP 和经典幅度对照。该实现含大量小算子调用，后续工程评估应进一步区分调度与计算成本，同时保留等价计算形式的对照。当前计时覆盖 router 与选择步骤；整模型与独占 GPU 的成本需要单独测量。')
d.p('输入舍入造成的集合变化数量较少，各模型的有害切换呈混合结果。部分拟合较弱的模型也有较低切换率，解释时需要同时看原模型保真、专家利用和事件后果。后续以同等功能质量下的误差尾部为比较对象更有意义。')
d.h('核验中发现并修正了一个选择器协议错误')
d.p('原始 gate 在 softmax 前按 logits 选择专家，第一次重放却按 softmax 后概率选择。一个临界样本中的不同 logits 被 FP32 舍入为相同概率，换掉了边界专家。修正后的独立重放完整复现原选择顺序，六个拆分的原始 logits 与缓存逐位相同。这次失败及定位记录保留，用于约束后续稳定性测量的算子顺序。')
d.page();d.h('已启动的下一轮：让功能几何进入训练目标')
d.p('下一轮把同一组冻结专家当作算子基底，并用原模型的稀疏权重作为功能参照。每种架构从本轮已核验 checkpoint 出发，三个损失使用相同初始化、相同采样流和相同训练预算。内部 dev 的真实模型输出 KL 统一用于 checkpoint 选择，包含零步起点。')
d.table(['损失','让 gate 学习的内容','需要回答的问题'],[['权重欧氏误差','复现原混合系数','权重目标能否弥补分数目标的不足'],['条件专家输出距离','复现当前输入上的专家组合输出','专家之间的功能替代性是否有帮助'],['模型输出 Fisher','降低会被模型头放大的组合误差','功能方向的重要性加权是否进一步有效']])
d.p('Fisher 使用原模型自己的输出分布生成无标签局部代价。几何矩阵在训练期预计算，推理仍只执行 gate、固定幂和 top-4 混合。实际 KL 会另外检验有限幅度替换的效果，区分局部训练近似与完整输出变化。')
d.p('这一阶段固定专家来识别 gate 的学习目标效应。之后若出现跨种子一致、与经典对照相比有实际价值的结果，再进入 gate 与专家联合训练，检验选择与反向优化的共同演化。当前 MLP 来源的教师和已有 checkpoint 起点也使结论具有明确条件：它测量的是对该专家系统的适配能力。')
d.table(['状态','交付内容'],[['本阶段已完成','9 项新增校准、21 模型重放、78 质量记录、156 舍入记录、78 计时记录及独立审计'],['运行中','36 项功能度量适配；四种架构、三种损失、三个种子'],['后续审校','适配的配对结果、checkpoint 核验、功能保真与切换风险；联合训练尚待该阶段判断']])
d.p('数据使用 train-only 内部分割，validation 仅作报告且已在此前研究中观察；test 保留。这里的 CE/PPL 来自最终专家模块的固定抽样位置，指标对应所记录的 BPE 协议。逐种子区间、实际参数、原始数组与执行范围见配套补充材料。')
d.link('本阶段数值补充（本地）',(HERE/'CAPACITY_GEOMETRY_SUPPLEMENT_20261030_zh.pdf').as_posix())
d.save()

a=Document('CAPACITY_GEOMETRY_SUPPLEMENT_20261030_zh');a.title('QuantumGate 线路校准与重放：补充数据')
a.p('对应 CAPACITY_GEOMETRY_REVIEW_20261030_zh。所有表均保留三个种子；区间为按窗口配对的 2,000 次重采样。该探索性比较使用此前观察过的 validation，模型选择只使用训练集内部 dev。')
a.h('A｜结构、参数与训练选择')
a.table(['方法','参数','种子','最佳步','dev R2','validation R2'],[[labels.get(r['method'],r['method']),r['parameters'],r['seed'],r['selected_step'],f'{r["splits"]["dev"]["r2_vs_constant"]:.4f}',f'{r["splits"]["validation"]["r2_vs_constant"]:.4f}'] for m in methods+['product_functional'] for r in rows if r['method']==m])
a.p('全部模型以相同教师、同一 train-only 样本、4,000 步和相同低秩输入编码初始化训练。参数数目含一个训练用缩放标量。参数接近时，各模型的可达概率分布仍有不同约束。旧 MLP／角编码／乘积模型作为冻结历史对照，新模型没有重新训练这些对照。')
a.page();a.h('B｜集合保真与校准系数')
a.table(['方法','seed','dev top-4','validation top-4','校准幂'],[[labels.get(r['method'],r['method']),r['seed'],f'{r["splits"]["dev"]["top4_set_agreement"]:.2%}',f'{r["splits"]["validation"]["top4_set_agreement"]:.2%}',f'{next(q["power"] for q in quality(r["method"]) if q["seed"]==r["seed"]):.4f}'] for m in methods+['product_functional'] for r in rows if r['method']==m])
a.p('校准幂为已训练标量的指数乘以训练目标尺度，完全由已有训练结果确定。native 为幂 1，calibrated 使用该固定幂。两种模式保持相同 native top-4 索引，改变选中权重的相对大小。此表的 top-4 为分数教师口径；重放表使用实际原始 gate 的 logits 选择口径，临界 tie 案例单独审计。')
a.h('完整配对表')
a.p('capacity_comparison_v1/paired.csv 含分数 R2 与集合一致率的 48 条配对结果；sparse_replay_v2/paired.csv 含 CE、KL、切换率、有害切换率及校准效应的 432 条配对结果。全部数字不经胜者筛选。')
for split in ['dev','validation']:
    a.page();a.h('C｜'+split+'：校准权重下的模型效果')
    qq=[r for r in s['rows'] if r['split']==split and (r['mode']=='calibrated' or r['method']=='original')]
    a.table(['方法','seed','CE','PPL','ΔCE','KL'],[[labels.get(r['method'],r['method']),r['seed'],f'{r["quality"]["sampled_ce"]:.4f}',f'{r["quality"]["sampled_ppl"]:.3f}',f'{r["quality"]["mean_delta_ce"]:+.5f}',f'{r["quality"]["mean_kl"]:.5f}'] for r in qq])
    a.p('每个 seed／split 包含 1,024 个长度 128 的窗口，每窗取位置 31、63、95、127。ΔCE 相对同一原始模型；KL 方向为原模型到候选模型。冻结最终专家模块的其余前向保持一致，整模型小样本钩子重放用于验证缓存替代公式。')
a.page();a.h('D｜关键配对差：交错 CZ 对不同对照')
pairs=read(BASE/'reviews/sparse_replay_v2/SUMMARY.json')['paired']
a.table(['对照／dev','seed','ΔKL','95% 窗口区间'],[[labels[r['reference']],r['seed'],f'{r["difference"]:+.5f}',f'[{r["ci95_low"]:+.5f}, {r["ci95_high"]:+.5f}]'] for r in pairs if r['split']=='dev' and r['mode']=='calibrated' and r['method']=='reencode_cz' and r['metric']=='delta_kl'])
a.p('正值表示交错 CZ 相对该对照有更大的模型偏移。其对旧角编码和配对乘积模型的改善在三个种子上方向一致；与 MLP、经典幅度的差距也在三个种子上方向一致。窗口区间描述给定 checkpoint 的采样不确定性，种子间变动另由 mean_sd.csv 汇总。')
a.h('原生与校准的总体差别')
a.table(['方法','native dev KL','calibrated dev KL'],[[labels[m],mean_sd([r['quality']['mean_kl'] for r in quality(m,mode='native')]),mean_sd([r['quality']['mean_kl'] for r in quality(m)])] for m in methods])
a.page();a.h('E｜输入 BF16 舍入与专家利用')
qr=[r for r in s['rows'] if r['split']=='dev' and (r['mode']=='calibrated' or r['method']=='original')]
a.table(['方法','seed','切换数','有害数','未用专家','集合数'],[[labels.get(r['method'],r['method']),r['seed'],r['roundtrip']['bf16_input_roundtrip']['flip_count'],r['roundtrip']['bf16_input_roundtrip']['harmful_flip_gt_01'],r['routing']['unused_experts'],r['routing']['unique_top4_sets']] for r in qr])
a.p('有害数按“集合发生变化且该次扰动使目标负对数概率增加超过 0.1 nat”计。每格 4,096 个位置。只对 gate 输入进行 BF16 往返舍入，专家输出保持冻结，后续计算仍为 FP32。完整 FP16、validation、均值与尾部记录见 roundtrip.csv；这些是有限事件的描述性测量。')
a.page();a.h('F｜执行开销与可核验范围')
a.table(['方法','1 行 median ms','4096 行 median ms','训练秒／run'],[[labels[m],mean_sd([t['median_ms'] for t in s['timings'] if t['method']==m and t['mode']=='native' and t['rows']==1],'.3f'),mean_sd([t['median_ms'] for t in s['timings'] if t['method']==m and t['mode']=='native' and t['rows']==4096],'.3f'),mean_sd([r['seconds_shared_gpu'] for r in rows if r['method']==m],'.1f')] for m in methods])
a.p('计时为串行 router+top-4，20 次预热、50 次 CUDA event 重复，表中为三个种子的中位时延均值±SD。TITAN V 与其他桌面／计算程序共享，gpu_before/after 保留现场。训练计时包含相应脚本中的训练与评估段，不能视为单算子理论复杂度。原始样本、P95、吞吐、增量分配显存在 summary.json／timing.csv。')
a.h('审计清单')
a.table(['项','核验结果'],[['新增训练／CPU 重放','9／18，全部通过'],['冻结模型重放','21 个，全部通过'],['质量／舍入／计时','78／156／78 条'],['原始数组哈希','234 个全部一致'],['整模型钩子对缓存公式','最大 logit 绝对误差 '+f'{audit["full_model_shortcut_max_abs"]:.2e}'],['缺失／失败（修正后正式组）','均为空；第一次失败独立保留'],['数学核验','256 个 QFI／CFI 点，解析式误差低于 9×10⁻¹⁶；Fisher Gram 与 Hessian 单元检验通过']])
a.p('原始结果位于 D:/files/MOTE/outputs/operator_geometry_round3：capacity/481aeb0bc8dd；replay_v2/ff4f146e8d29；reviews/sparse_replay_v2；reviews/capacity_comparison_v1；theory/interleaved_qfi_v1；diagnostics/selector_v1；PROTOCOL_REVISION.json。源码与配置随正式组保存快照。')
a.p('独立后续方案为 FUNCTIONAL_FOLLOWUP_PLAN.json，运行组 functional/50b8c7704256。当前报告截止于容量与重放核验完成，功能度量适配仍在进行。阶段整体状态和顺延预算见 operator_geometry_round2/SESSION_STATE.md 与 EXTENSION_STATE.json。')
a.save();print('CAPACITY_REPORT_WRITTEN',flush=True)
