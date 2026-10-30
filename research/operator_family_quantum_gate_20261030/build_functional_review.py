"""Conclusions-first functional study; numerical detail in a separate supplement."""
from pathlib import Path
import json,ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];BASE=REPO/'outputs/operator_geometry_round3'
FIG=HERE/'functional_figures';FIG.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
f=read(BASE/'reviews/functional_v1/SUMMARY.json');g=read(BASE/'reviews/functional_diagonal_v1/SUMMARY.json')
assert f['audit']['passed'] and g['audit']['passed']
t=read(BASE/'timing/functional_v1/raw_timing.json');assert read(BASE/'timing/functional_v1/AUDIT.json')['passed']
mechanism=read(BASE/'reviews/functional_mechanism_v1/SUMMARY.json');tangent=read(BASE/'theory/sparse_tangent_v1/SUMMARY.json')
rows=f['quality']+g['quality'];raw=[]
for stage in ['functional','functional_diagonal']:
    group=Path(read(BASE/f'{stage}/STATUS.json')['group']);raw+=read(group/'raw_summary.json')
old=read(BASE/'replay_v2/ff4f146e8d29/summary.json')['rows']
methods=['mlp','affine_born','reencode_cz','reencode_product'];losses=['coefficient','output','fisher','output_diag','fisher_diag']
labels={'mlp':'MLP','affine_born':'Affine Born','reencode_cz':'CZ','reencode_product':'Product'}
short={'coefficient':'C','output':'O','fisher':'F','output_diag':'O-diag','fisher_diag':'F-diag'}
colors=['#315e86','#3d806e','#b3654d','#8773a8']
def select(m,l,split='dev'):return sorted([r for r in rows if r['method']==m and r['objective']==l and r['split']==split],key=lambda r:r['seed'])
def avg(m,l,k='kl',split='dev'):return np.mean([r[k] for r in select(m,l,split)])
def ms(vals,fmt='.5f'):return f'{np.mean(vals):{fmt}} ± {np.std(vals,ddof=1):{fmt}}'
def meanrow(m,l,k,split='dev',fmt='.5f'):return ms([r[k] for r in select(m,l,split)],fmt)
def pairs(m,obj,ref,metric='delta_kl',split='dev'):
    return [r for r in g['paired'] if r['method']==m and r['objective']==obj and r['reference_objective']==ref and r['metric']==metric and r['split']==split]

plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,2,figsize=(10.2,3.2),layout='constrained')
for j,m in enumerate(methods):
    before=[r['quality']['mean_kl'] for r in old if r['method']==m and r['mode']=='calibrated' and r['split']=='dev']
    for i,(loss,values) in enumerate([('Initial',before)]+[(short[l],[r['kl'] for r in select(m,l)]) for l in losses[:3]]):
        x=j+(i-1.5)*.17;axes[0].scatter([x]*3,values,s=15,c=colors[j],marker=['x','o','s','^'][i],alpha=.85)
        axes[0].plot([x-.055,x+.055],[np.mean(values)]*2,c=colors[j],lw=2)
    axes[1].plot(range(3),[avg(m,l) for l in losses[:3]],color=colors[j],marker='o',label=labels[m])
axes[0].set_yscale('log');axes[0].set_xticks(range(4),[labels[m] for m in methods]);axes[0].set_ylabel('Exact model-output KL');axes[0].set_title('Initial / C / O / F (left to right)')
axes[1].set_yscale('log');axes[1].set_xticks(range(3),['Coefficient','Output','Fisher']);axes[1].set_ylabel('Mean exact KL');axes[1].legend(frameon=False)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.savefig(FIG/'functional_kl.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10.2,3.0),layout='constrained')
for i,(base,ax) in enumerate(zip(['output','fisher'],axes)):
    for j,m in enumerate(methods):
        ps=pairs(m,base,base+'_diag');values=np.array([r['difference'] for r in ps]);lo=np.array([r['ci95_low'] for r in ps]);hi=np.array([r['ci95_high'] for r in ps]);x=j+np.array([-.13,0,.13])
        ax.errorbar(x,values,yerr=np.stack([values-lo,hi-values]),fmt='o',ms=3,c=colors[j],capsize=2)
    ax.axhline(0,c='#55616b',lw=.8);ax.set_xticks(range(4),[labels[m] for m in methods]);ax.set_title(base.capitalize()+': full minus diagonal');ax.set_ylabel('Paired KL difference');ax.grid(axis='y',alpha=.2)
fig.savefig(FIG/'cross_terms.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9.8,3.0),layout='constrained')
for j,m in enumerate(methods):
    v=[r['median_ms'] for r in t if r['method']==m and r['rows']==4096]
    axes[0].bar(j,np.mean(v),color=colors[j],alpha=.75);axes[0].scatter([j]*len(v),v,s=8,c='#1d3244',alpha=.5)
axes[0].set_xticks(range(4),[labels[m] for m in methods]);axes[0].set_ylabel('Router + top-4 ms / 4096 rows');axes[0].grid(axis='y',alpha=.2)
for j,m in enumerate(['reencode_cz','reencode_product']):
    for i,obj in enumerate(['initial','coefficient','output','fisher']):
        z=[r for r in tangent if r['method']==m and r['objective']==obj and r['split']=='dev' and r['floor']==0]
        axes[1].bar(j+(i-1.5)*.18,np.sum([r['weight_rank2_count'] for r in z])/96,width=.16,color=['#315e86','#ba744a','#3d806e','#8773a8'][i],label=['Initial','C','O','F'][i] if j==0 else None)
axes[1].set_xticks([0,1],['CZ','Product']);axes[1].set_ylim(0,1);axes[1].set_ylabel('Rank-2 fraction (unfloored, fixed top-4)');axes[1].legend(frameon=False,ncol=2,fontsize=8)
fig.savefig(FIG/'cost_and_tangent.png');plt.close(fig)

tree=ast.parse((HERE/'build_stage_review.py').read_text(encoding='utf8'));node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Document')
exec(ast.unparse(node).replace('stage_figures/','functional_figures/'),globals())

d=Document('FUNCTIONAL_GEOMETRY_REVIEW_20261030_zh');d.title('QuantumGate：从分数拟合转向功能与任务')
d.p('阶段审校报告｜2026-10-30。功能适配、对角控制、checkpoint 重放与串行计时已完成。正文聚焦结论与设计含义，逐 seed 数据、区间和审计清单见配套补充材料。')
d.h('主要结论：先改训练对象，再判断结构的适用范围')
d.p('对实际参与计算的稀疏混合权重追加适配后，量子与经典 gate 对既有模型功能的保留均明显改善。三种目标共同追加相同训练预算，使误差集中到真正送入专家混合的权重上；本轮的主要对照是这些目标之间的差异，适配前 checkpoint 用于记录变化幅度。')
d.p('交错 CZ 门从专家输出距离中获得收益：它对原模型输出分布的偏离在三个种子上均低于权重距离目标，其中两个种子的配对区间明确向下。对角控制取得了几乎相同的效果，说明当前收益主要来自各专家的功能尺度。模型 Fisher 目标没有稳定超越输出距离；MLP 与经典幅度门采用权重距离时已经较好地保留功能。')
d.figure('functional_kl.png','同一内部 dev 上的真实输出 KL。左图每组依次为适配前、权重距离 C、专家输出距离 O、模型 Fisher 距离 F；右图展示三种目标的均值。每种子分别保留，纵轴采用对数尺度。',68)
d.p('这轮评价围绕一个已经训练好的经典 teacher 展开。它回答的是结构替换时如何保留现有功能。另一条独立评价路线允许 QuantumGate 与专家共同形成不同分工，直接以任务成绩、计算效率和稳定性评判；后续架构取舍需要同时纳入这条路线。')
d.page();d.h('两种 Fisher 分别刻画模型风险与状态方向')
d.p('专家输出决定“换一种混合方式会把隐藏表示推向哪里”，模型输出的 Fisher 决定“这些方向会使预测分布变化多少”。前者给出专家空间中的真实平方距离，后者对它再施加模型头的敏感度。两种几何都可以训练经典 gate 和 QuantumGate。')
d.math(r'd=w-w_0,\quad G_O=VV^\top/D,\quad G_F=\tfrac12 J^\top[\operatorname{diag}(p)-pp^\top]J,\quad L=d^\top Gd',r'd=w-w_0, quad G_O=V V^top / D, quad G_F=1/2 J^top ("diag"(p)-p p^top) J, quad L=d^top G d')
d.p('这里 V 的各行为中心化专家输出；J 将混合系数变化映射到最终 logits，包含 LayerNorm 与词表投影。输出目标等于真实混合输出误差；Fisher 目标是原模型分布附近 KL 的二阶近似。训练只使用训练集缓存的几何与原始权重，所有目标都用内部 dev 的真实 KL 选择 checkpoint。')
d.p('量子 QFI 处理另一个层次：状态对编码角度的可区分性。交错编码使完整态的有效角度方向由乘积对照的四个增加到八个。稀疏读出随后只保留四个、总和为一的混合权重，因此固定选择集合内最多有三个独立权重方向。状态表达、稀疏读出与模型功能需要分别检查。')
d.p('对任意正概率分布，把各分量平方根作为实振幅，其 QFI 就等于该分布的经典 Fisher。Softmax 与 Born 在同一个概率分布上的这一度量一致；具体线路约束了可达分布族及其输入映射，这才是结构比较的对象。')
d.h('一个更直接的结构线索：top-4 支持集的几何')
d.p('对无概率地板的乘积态，选中权重的对数比由四个独立比特的 log-odds 决定。若选中的四个基态形成平行四边形，局部权重只能沿两个方向变化；一般四点单纯形允许三个方向。乘积门样本中出现了大量秩二集合，CZ 样本均为秩三。微小概率地板可把乘积门的退化方向抬升成很弱的方向；这个细节与 Jacobian 谱一起保留。')
d.figure('cost_and_tangent.png','左：全部适配 checkpoint 的串行 router+top-4 计时。右：固定 top-4 下无地板权重 Jacobian 的秩二比例。右图讨论角度到权重的局部映射；跨集合选择边界需另外分析。',62)
d.page();d.h('当前 CZ 收益主要由专家自身尺度解释')
d.p('对角控制保留每个专家的功能尺度，并删除专家对之间的交叉项。它与完整几何使用相同起点、样本、训练步数、校准幂和训练集 trace 归一化。这个比较把“不同专家应该受到不同惩罚”与“专家之间的相容关系有价值”拆开。中心化专家坐标在所有控制中固定。')
d.figure('cross_terms.png','完整几何减去对应对角几何的真实 KL 配对差；每个点为一个种子，误差条为按窗口重采样的 95% 区间。负值表示完整几何保留原模型更好。',67)
d.p('CZ 的完整输出几何与对角几何几乎重合，Fisher 对角控制的 KL 点估计在三个种子上都更低。MLP 与经典幅度门的完整交叉项也未改善内部 dev KL。乘积门的完整输出几何在三个种子上均有小幅点估计改善，表明交叉项的作用会随可达混合族而变化。现阶段优先采用功能尺度更有依据。')
d.p('误差分解还显示，较小的权重误差与较低的功能敏感度可以通过不同途径降低模型偏移。跨种子平均后，乘积门在输出目标下允许略大的系数误差，同时将误差移向输出影响较小的方向。当前中心化坐标下，整体交叉项平均表现为误差放大；这一分解支持进一步检查误差方向与专家尺度的匹配。')
d.h('当前预算中的实际代价')
d.p('功能几何只进入训练目标，推理仍沿各自原有 router 与 top-4 路径执行。量子态模拟路径的串行耗时高于 MLP 与经典幅度门；本轮记录了适配后全部 checkpoint 的实际延迟。训练阶段另有专家输出和 Fisher Gram 的预计算成本，部署与训练的预算分开列示。')
d.p('输入舍入下的稳定性比较呈混合结果。有害切换事件较少，目标和种子之间的变化方向不一致；完整计数、误差尾部、负载和未用专家均放入补充材料，后续共同适配保留这些评价项。')
d.page();d.h('下一步：以任务驱动的共同分工检验 QuantumGate')
d.p('后续最有价值的实验是允许 gate 与专家共同适配，再观察它们形成的分工及任务成绩。经典 teacher 模仿能力作为机制对照保留，任务损失将成为主评价。这样可以识别两类成功：较好地复现既有计算，以及通过不同路由组织形成同样好或更好的计算。')
d.table(['评价路径','优化与固定项','主要判据'],[['既有功能保留（本轮）','专家与模型固定，调整 gate','真实 KL、CE、选择边界与开销'],['共同适配（下一阶段）','同一共享初始化，gate 与专家共同更新','等 token CE/PPL、等时间成绩、稳定性'],['分工机制分析','保留全部种子与预定控制','功能相似性、实际替换代价、任务条件分工']])
d.p('建议下一阶段预先固定 MLP、经典幅度、CZ 与乘积四组，使用相同专家初始化和采样流。先做小规模共同适配可行性检查，再以训练集内部 dev 决定是否扩大。将语义任务条件、专家激活与实际功能替换证据结合，检验任务专属算子的解释。')
d.p('扩大前还应加入最接近的已有量子路由移植对照，并沿用前次新颖性核查中对公开实现与论文读出形式的区分。功能关系驱动的候选结构需要与同规模随机关系、匹配经典读出接受相同监督，形成清楚的增量比较。')
d.h('已有研究与本轮工作的落点')
d.p('MoE 层输出蒸馏已有直接先例；B-Distill 还联合使用 router 分布匹配、熵正则与专家探索。重复数据编码的表达性也已有系统研究。本轮值得继续追踪的问题是：怎样依据实际专家功能和稀疏边界，选择或约束量子线路的可达方向，并在同等预算的经典对照中检验它。')
d.link('MoE 层输出蒸馏：原始论文','https://www.microsoft.com/en-us/research/uploads/prod/2022/05/MainzSpeech_Interspeech2022_KD_MoE_Network.pdf')
d.link('B-Distill：router 分布与专家探索','https://ojs.aaai.org/index.php/AAAI/article/download/39543/43504')
d.link('量子重复编码的表达与输入尺度研究','https://arxiv.org/html/2610.09947v1')
d.h('交付范围与下一次审校')
d.p('本阶段完成了功能目标适配、交叉项控制、数学核验和推理路径计时，保留了前轮阴性结果及选择器诊断。当前测量覆盖冻结模型最后一个专家模块，在预定窗口位置重放实际前向；自然标签只用于结果报告。validation 已被观察，test 仍保留。端到端共同适配与新任务上的独立验证列为下一阶段工作。')
d.link('逐 seed 结果、协议与核验补充',(HERE/'FUNCTIONAL_GEOMETRY_SUPPLEMENT_20261030_zh.pdf').as_posix());d.save()

a=Document('FUNCTIONAL_GEOMETRY_SUPPLEMENT_20261030_zh');a.title('功能几何实验：数据与审计补充')
a.p('C：混合权重平方距离；O：专家输出距离；F：模型 Fisher 二次代价；O-diag、F-diag：对应中心化 Gram 的对角控制。均值 ± SD 使用三个训练种子；配对区间按同一 checkpoint 的 1,024 个窗口做 2,000 次重采样。')
a.h('协议与选择规则')
a.p('四个 gate、三个种子、五种目标，共 60 项适配。每项从各自已训练的分数拟合 checkpoint 开始；同一 gate/seed 的五个目标共享全部初始参数、样本序列与固定校准幂。每项 2,000 步、batch256，学习率 0.0003；所有目标按真实内部 dev KL 从 step0 与每200步 checkpoint 中选择。')
a.p('训练缓存来自训练集前75%，内部 dev 来自后25%的独立窗口；验证集只在 checkpoint 选定后报告。训练几何、目标及 checkpoint 选择使用原模型权重和输出分布。自然标签仅用于结果报告。观察过的 validation 单列，测量范围为最终专家模块替换后的预定采样位置。')
a.table(['gate','实际参数','更新参数','执行路径'],[[labels[m],next(r['parameters_total'] for r in raw if r['method']==m),next(r['parameters_updated'] for r in raw if r['method']==m),'经典' if m in methods[:2] else '实态模拟'] for m in methods])
a.p('冻结的 log_scale 仍计入总参数，适配时不更新。专家和主干参数均固定；同一 seed 共享同一模型 checkpoint。功能 Gram 每样本16×16，输出与 Fisher 各一份；推理不读取 Gram。')
a.h('数学恒等式与计量对象')
a.math(r'\|\sum_e d_e f_e\|^2=\sum_e d_e^2\|v_e\|^2+\sum_{e\ne j}d_e d_j\langle v_e,v_j\rangle,\quad v_e=f_e-\bar f',r'norm(sum_e d_e f_e)^2 = sum_e d_e^2 norm(v_e)^2 + sum_(e != j) d_e d_j angle.l v_e,v_j angle.r, quad v_e=f_e-overline(f)')
a.p('系数和为零时，完整输出误差对专家共同平移不变。对角能量与交叉项的拆分依赖坐标原点，本轮统一使用中心化专家。Fisher Gram 与独立 JVP、精确 Hessian 的局部关系已核验。QFI 描述状态角度；稀疏权重 Jacobian 描述固定集合内的可达权重方向。')
a.p('当前目标只对被选中的权重求通常梯度；共享网络参数更新可以改变后续的选择边界，top-k 集合本身未引入替代梯度。初始 checkpoint 与适配后结果的差异同时包含追加训练与目标改变；本轮直接比较的因子为共同追加预算下的 C/O/F/对角目标。')
for split in ['dev','validation']:
    for mslice in [methods[:2],methods[2:]]:
        a.page();a.h(split+'：'+ ' / '.join(labels[m] for m in mslice))
        a.table(['gate/目标','seed','最佳步','CE','PPL','ΔCE','KL'],[[labels[m]+'/'+short[l],r['seed'],r['selected_step'],f'{r["sampled_ce"]:.4f}',f'{r["sampled_ppl"]:.3f}',f'{r["delta_ce"]:+.5f}',f'{r["kl"]:.5f}'] for m in mslice for l in losses for r in select(m,l,split)])
        a.p('ΔCE 相对各自原始模型。KL 方向为原模型到适配模型。每 seed/split 为4,096个位置，同一窗口内的位置共同重采样。所有候选完整列示。')
a.page();a.h('跨种子汇总：validation 均值 ± SD')
a.table(['gate/目标','CE','PPL','KL'],[[labels[m]+'/'+short[l],meanrow(m,l,'sampled_ce','validation','.5f'),meanrow(m,l,'sampled_ppl','validation','.3f'),meanrow(m,l,'kl','validation','.5f')] for m in methods for l in losses])
a.p('SD 使用三个 seed 的样本标准差（ddof=1），与给定 checkpoint 内按窗口重采样的区间分开。PPL 为每 seed 的采样 CE 指数，表中取其均值与SD；BPE词表与采样协议固定。')
a.page();a.h('交叉项的配对结果：完整目标减去对角目标')
a.table(['gate/目标','seed','dev ΔKL','dev 95%区间','val ΔKL'],[[labels[m]+'/'+short[l],r['seed'],f'{r["difference"]:+.5f}',f'[{r["ci95_low"]:+.5f}, {r["ci95_high"]:+.5f}]',f'{next(x["difference"] for x in pairs(m,l,l+"_diag",split="validation") if x["seed"]==r["seed"]):+.5f}'] for m in methods for l in ['output','fisher'] for r in pairs(m,l,l+'_diag')])
a.p('配对 CE、完整 validation 区间及对 coefficient 的比较保存在 reviews/functional_diagonal_v1/paired.csv；原始三目标的900项配对统计保存在 reviews/functional_v1/paired.csv。区间反映给定 checkpoint 的窗口采样变动；训练种子差异单独保留。')
a.page();a.h('数值扰动、利用率与误差机制')
a.table(['gate/目标','dev BF16切换','有害切换','未用专家','完整集合数'],[[labels[m]+'/'+short[l],sum(r['bf16_flips'] for r in select(m,l)),sum(r['bf16_harmful'] for r in select(m,l)),meanrow(m,l,'unused_experts',fmt='.1f'),meanrow(m,l,'unique_top4_sets',fmt='.1f')] for m in methods for l in losses])
a.p('切换与有害事件为三个种子共12,288个位置的计数。有害定义为集合发生变化，且该次输入舍入令目标负对数概率增加超过0.1nat。这里仅将 gate 输入做FP16/BF16舍入往返，计算核与专家保持FP32；全部FP16记录和误差尾部见原始result.json。')
a.h('机制数据')
a.p('reviews/functional_mechanism_v1 含96组逐位置误差分解：系数距离、原始权重遗漏质量、输出误差、中心化能量及交叉项。theory/sparse_tangent_v1 含24个checkpoint的1,536个位置、3,072个有/无地板Jacobian与128个独立数学检查。前者为结果描述，后者为固定支持集的局部结构。')
a.page();a.h('执行预算与串行延迟')
a.table(['gate/目标','1行ms','256行ms','4096行ms','适配秒/run'],[[labels[m]+'/'+short[l],*[ms([r['median_ms'] for r in t if r['method']==m and r['objective']==l and r['rows']==n],'.3f') for n in [1,256,4096]],ms([r['seconds_including_checks'] for r in raw if r['method']==m and r['objective']==l],'.1f')] for m in methods for l in losses])
a.p('计时逐checkpoint串行执行，20次预热、50次CUDA event重复；以上对三个种子的median取均值与SD。执行时间包括router、稀疏选择与固定校准幂。原始50次、P95、吞吐、增量显存、GPU现场在 timing/functional_v1。TITAN V与其他进程共享，测量范围为该运行环境的router路径。')
a.p('适配秒数包括训练与该脚本内的检查，不包括共享的几何预计算。完整36项与对角24项的起止状态、预计算文件和日志均保留。对角控制复用已经验证的训练Gram，完整列出原始来源及哈希。')
a.page();a.h('完成核验与可追溯路径')
a.table(['工作','核验'],[['完整功能目标','36训练、72质量、144扰动、900配对，独立CPU审计通过'],['对角控制','24训练、48质量、96扰动、48CPU checkpoint重放'],['适配后计时','60checkpoint、180组、每组50次重复'],['误差分解','96记录，Gram恒等式与平移不变性通过'],['稀疏切空间','24checkpoint、3,072Jacobian；独立微分与有限差分通过'],['缺失/失败（本轮正式组）','空；旧选择器失败v1、强正则pilot与阴性研究完整保留']])
a.p('主要结果目录：D:/files/MOTE/outputs/operator_geometry_round3。functional/50b8c7704256；functional_diagonal/260b32fdf440；reviews/functional_v1；reviews/functional_diagonal_v1；reviews/functional_mechanism_v1；theory/sparse_tangent_v1；timing/functional_v1。每组保存配置、冻结源码哈希、checkpoint、采样数组、训练曲线与原始NPZ。')
a.p('先前阶段报告：STAGE_REVIEW_20261030_zh 与 CAPACITY_GEOMETRY_REVIEW_20261030_zh，分别对应条件代价/几何机制与容量/稀疏重放阶段。当前报告新增功能适配与交叉项控制。阶段用时按用户授权排除网络中断；不重复旧训练填充预算。')
a.h('文献与后续评价')
a.link('Balanced Knowledge Distillation for Large Language Models with Mix-of-Experts','https://ojs.aaai.org/index.php/AAAI/article/download/39543/43504')
a.link('Knowledge Distillation for Mixture of Experts Models in Speech Recognition','https://www.microsoft.com/en-us/research/uploads/prod/2022/05/MainzSpeech_Interspeech2022_KD_MoE_Network.pdf')
a.link('On the Expressive Power and Capacity of Quantum Data Reuploaders','https://arxiv.org/html/2610.09947v1')
a.p('后续共同适配实验应独立预注册任务成绩与资源判据，允许不同于teacher的专家分工。当前已观察validation用于探索性汇报，未来扩大时需预留新的评估切分或任务。')
a.link('先前量子路由新颖性与最近对照核查',(HERE/'NOVELTY_REVIEW_zh.pdf').as_posix())
a.save();print('FUNCTIONAL_REPORT_WRITTEN',flush=True)
