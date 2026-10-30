"""Generate conclusion-first stage report, numerical supplement and figures."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
AUDIT=REPO/'outputs/operator_geometry_round2/reviews/stage_20261030_v1'
S=json.loads((AUDIT/'SUMMARY.json').read_text(encoding='utf8'))
FIG=HERE/'stage_figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
COLORS=['#345b86','#267f73','#c3863c']
methods=['mlp','angle_functional','product_functional'];names=['MLP','Four-angle Born','Product Born']
fig,axes=plt.subplots(1,2,figsize=(9.2,3.0),layout='constrained')
for ax,key,title in zip(axes,['r2','top4_agreement'],['Teacher-score fit','Exact top-4 set agreement']):
    vals=[[r[key] for r in S['positive'] if r['method']==m and r['split']=='validation'] for m in methods]
    ax.bar(range(3),[np.mean(v) for v in vals],color=COLORS,alpha=.8,width=.6)
    for j,v in enumerate(vals):ax.scatter(j+np.array([-.12,0,.12]),v,color='#172b40',s=23,zorder=3)
    ax.set_xticks(range(3),names);ax.set_ylim(0,1.06);ax.set_title(title);ax.set_ylabel('R²' if key=='r2' else 'Fraction');ax.grid(axis='y',alpha=.2)
fig.savefig(FIG/'positive.png');plt.close(fig)

keys=['fisher_rho','raw_local_full_distance','raw_local_energy_only','raw_global_full_distance','raw_global_energy_only','predicted_euclidean','unanchored_ray','anchored_ray_x4']
labels=['Head Fisher','Local distance','Local energy','Global distance','Global energy','Input sketch','State ray','Anchored ray']
rows=[r for r in S['risk'] if r['split']=='dev'];fig,ax=plt.subplots(figsize=(9.2,3.1),layout='constrained')
values=[[r[k] for r in rows] for k in keys]
ax.bar(range(8),[np.mean(v) for v in values],color=['#345b86']+['#267f73']*4+['#c3863c']*3,alpha=.8,width=.6)
for j,v in enumerate(values):ax.scatter(j+np.array([-.12,0,.12]),v,color='#172b40',s=22,zorder=3)
ax.set_xticks(range(8),labels,rotation=20,ha='right');ax.set_ylabel('Conditional Spearman ρ with exact KL');ax.set_ylim(-.05,1.05);ax.grid(axis='y',alpha=.2)
fig.savefig(FIG/'risk.png');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9.2,3.0),layout='constrained')
for seed,c in zip([17,29,43],COLORS):
    r=[r for r in S['geometry'] if r['seed']==seed and r['split']=='dev'];r.sort(key=lambda x:x['rank'])
    axes[0].plot([x['rank'] for x in r],[x['sketch_r2'] for x in r],'-o',color=c,label=f'Seed {seed}')
    axes[1].plot([x['rank'] for x in r],[x['predicted_euclidean'] for x in r],'-o',color=c)
    axes[1].axhline(r[0]['train_global_euclidean'],color=c,linestyle='--',alpha=.7)
axes[0].set_ylabel('Output-sketch R²');axes[1].set_ylabel('Conditional ρ with full output distance')
for ax in axes:ax.set_xlabel('Linear rank');ax.set_xticks([8,16,32]);ax.set_ylim(0,1);ax.grid(alpha=.2)
axes[0].legend(frameon=False);axes[1].set_title('Solid: predicted; dashed: train-global')
fig.savefig(FIG/'geometry.png');plt.close(fig)

class Document:
    def __init__(self,name):self.name=name;self.blocks=[]
    def title(self,s):self.blocks.append(('title',s))
    def h(self,s):self.blocks.append(('heading',s))
    def p(self,s):self.blocks.append(('p',s))
    def page(self):self.blocks.append(('page',None))
    def table(self,headers,rows):self.blocks.append(('table',(headers,rows)))
    def figure(self,name,caption,height=58):self.blocks.append(('figure',(name,caption,height)))
    def math(self,md,typ):self.blocks.append(('math',(md,typ)))
    def link(self,label,url):self.blocks.append(('link',(label,url)))
    def save(self):
        q=lambda x:json.dumps(str(x),ensure_ascii=False)
        md=[];typ=['#set page(paper: "a4", margin: (x: 22mm, y: 19mm), numbering: "1")',
            '#set text(font: ("Times New Roman", "KaiTi"), size: 10.5pt, lang: "zh")',
            '#set par(justify: true, leading: 0.65em)', '#set heading(numbering: none)',
            '#show heading.where(level: 1): set text(size: 19pt, fill: rgb("17365d"))',
            '#show heading.where(level: 2): set text(size: 13pt, fill: rgb("17365d"))',
            '#show math.equation: set text(font: "New Computer Modern Math", size: 10.5pt)']
        for kind,val in self.blocks:
            if kind=='page':typ.append('#pagebreak()')
            elif kind in ['title','heading']:
                level=1 if kind=='title' else 2;md.append('#'*level+' '+val);typ.append(f'#heading(level: {level})[#text({q(val)})]')
            elif kind=='p':md.append(val);typ.append('#text('+q(val)+')')
            elif kind=='math':md.append('$$\n'+val[0]+'\n$$');typ.append('$ '+val[1]+' $')
            elif kind=='link':md.append(f'[{val[0]}]({val[1]})');typ.append(f'#link({q(val[1])})[#text({q(val[0])})]')
            elif kind=='figure':
                name,caption,height=val;md.append(f'![{caption}]({(FIG/name).as_posix()})\n\n{caption}')
                typ.append(f'#figure(image("stage_figures/{name}", width: 100%, height: {height}mm, fit: "contain"), caption: [#text({q(caption)})])')
            elif kind=='table':
                head,rows=val;md.append('\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]))
                cells=[f'[#strong[#text({q(x)})]]' for x in head]+[f'[#text({q(x)})]' for row in rows for x in row]
                typ.append('#block(width: 100%)[#set text(size: 9pt)\n#table(columns: '+str(len(head))+', inset: 5pt, stroke: 0.35pt + rgb("aebdcb"), '+',\n'.join(cells)+')]')
        (HERE/(self.name+'.md')).write_text('\n\n'.join(md)+'\n',encoding='utf8')
        (HERE/(self.name+'.typ')).write_text('\n\n'.join(typ)+'\n',encoding='utf8')

d=Document('STAGE_REVIEW_20261030_zh')
d.title('MoTE 功能几何与 QuantumGate：阶段审校报告')
d.p('2026-10-30｜阶段范围：条件效用学习、功能风险与离散选择机制｜本报告冻结第二轮结果；顺延后的第三轮另行报告。')
d.h('主要结论：先补足量子 gate 的表达能力，再优化功能相关的选择边界')
d.p('本阶段把下一步研究收敛到一个可检验的问题：量子线路怎样在接近经典 gate 的参数预算下，表达足够丰富的专家选择，并使容易发生的选择变化对应较低的功能代价。当前四角编码在确定性路由教师上的拟合存在明显瓶颈。优先补足这一瓶颈，再研究功能加权边界，比直接扩大现有静态排列方案更有依据。')
d.table(['判断','关键证据','研究含义'],[
    ['有符号替换效用尚难泛化','全部非线性候选的内部开发集 R² 为负；强线性对照接近零','保留阴性结果；下一轮改用确定性目标分离表征问题'],
    ['当前角编码拟合受限','MLP 教师分数拟合 R²≈0.983，四角 Born≈0.370','优先检验交错数据编码与匹配经典对照'],
    ['功能代价与选择边界均可测量','逐输入距离有额外专家对信号；一阶边界解释绝大多数舍入切换','为后续边界优化提供监督与机制检验']])
d.h('最重要的正面发现')
d.p('专家之间确实存在超出单个专家输出能量的局部功能差异。用完整专家距离对有限替换的输出 KL 排序，三个种子都比只用能量更好。这种增量在逐输入条件下稳定出现，支持继续研究条件功能结构。其收益规模较小，下一步需要确认这些信息能以低成本从 gate 的可见输入中提取。')
d.h('最重要的阴性发现')
d.p('静态专家对距离的增量很弱，条件输出草图也落后于训练集全局距离。直接把专家输出归一化为量子射线会丢失有用的符号和尺度信息。现阶段的结果支持重设编码与读出，原有静态量子排列继续保留为未通过扩展标准的方案。')
d.page();d.h('一、效用学习失败的原因需要分层判断')
d.p('用推理时可见的输入预测“换一个专家能否降低当前标签损失”，本轮各类小模型都没有获得稳定的开发集增益。保守阈值让多数模型保留原始路由；零切换来自未采取干预。少数期望值适配器在每个种子的验证样本上仅触发一次替换，配对区间均包含零。这些事件适合做个案诊断，统计证据不足以支持部署选择。')
d.p('监督目标的一阶近似并非主要失败来源：在固定被移除专家的候选中，它与精确替换损失的排序高度一致。更值得追查的是标签相关收益能否由输入预测，以及当前模型的表示与优化能力。为分开这两个因素，我们增加了原 gate 分数蒸馏：教师完全由同一输入确定，训练流程、采样和预算保持一致。')
d.figure('positive.png','确定性教师校准。柱为三个种子的均值，点为单种子；验证集用于报告，checkpoint 由训练集内的开发集选定。',66)
d.p('MLP 能较准确地拟合教师分数，证明这条数据与训练流程能学习确定性的输入信号。四角 Born 和乘积态版本仍有明显误差，说明当前量子结构的表达能力或可优化性值得优先修正。教师来自 MLP，因此这项校准主要衡量对既有路由功能的兼容性。')
d.p('高分数拟合也没有自动转化为完全一致的 top-4：MLP 的集合一致率约为 45%。这使“预测分数—离散边界—专家组合”的区别成为后续设计中的核心要求。分数误差、集合变化、最终 CE/KL 应分别报告。')
d.page();d.h('二、功能几何的有效部分：局部专家对信息与输出方向')
d.p('完整专家距离对替换风险有解释力，其中相当一部分来自单专家输出能量。拆开能量与交叉项后，逐输入的专家对信息仍给出稳定的小幅增量；训练集全局专家对结构的增量则较弱。这提示“可划分任务算子族”目前更适合作为条件性假设，下一步应检验输入相关的专家相互作用。')
d.figure('risk.png','对精确输出 KL 的排序能力。固定 token、被移除专家及其质量，在十二个替代专家内计算相关后平均；点为三个种子，数据为内部开发集。',67)
d.p('Head Fisher 对风险的排序最强，因为它同时考虑专家输出方向和后续语言模型映射。有限替换下的绝对 KL 近似误差有明显尾部，因此它适合作为候选风险监督；最终评价仍使用精确输出分布。该教师需要观察专家和后续网络，推理 gate 的关键问题是如何在训练中吸收这些信息。')
d.p('学习到的低维输出草图可预测部分专家变化，增大秩能改善草图拟合；在本协议中，预测出的距离排序仍落后于训练集全局矩阵。单纯增加条件化表示没有带来更好的风险排序。应先提升对关键输出方向的表达，再决定是否值得付出额外推理开销。')
d.p('无锚点的量子射线把相反向量视为同一个态方向，同时消除范数；这与专家混合中的加性贡献存在冲突。加上固定正锚点能恢复这些信息，所得排序接近同信息的欧氏对照。这为量子编码提出明确要求：保留任务需要的方向与尺度，并与对应的经典核逐项比较。')
d.page();d.h('三、Fisher 的作用与量子线路的设计空间')
d.p('Fisher 用来描述概率变化的局部几何。相同类别概率对应相同的统计 Fisher；softmax 与 Born 的区别主要体现在参数化、可达概率集合以及从输入到概率的映射。当前实数 RY/CZ 编码，在完整、无下限截断、非零振幅的条件下，测量 Fisher 等于量子 Fisher。固定的编码后幺正变换保持编码角坐标下的量子 Fisher，同时可以改变读出概率和选择边界。')
d.math(r'G_{\mathrm{func}}(x)=V^{\mathsf T}J^{\mathsf T}[\operatorname{diag}(p_{\mathrm{LM}})-p_{\mathrm{LM}}p_{\mathrm{LM}}^{\mathsf T}]JV.',r'G_("func")(x) = V^T J^T [op("diag")(p_("LM")) - p_("LM") p_("LM")^T] J V.')
d.p('上式把专家输出矩阵 V、后续网络 Jacobian J 和语言模型分布联系起来。它描述权重小变化的功能代价；top-k 跨界产生有限专家替换，应配合精确 KL 检验。它也给出了训练与推理顺序的设计方式：训练时使用昂贵的功能教师，推理前向用低成本 gate 近似其关键结构。')
d.h('真实边界比抽象基态邻接更直接')
d.p('在原始 FP32 运算上对输入做 BF16 往返舍入，固定第四／第五名的一阶预测解释了 295 次集合变化中的 293 次；另两次由其他近邻名次交换造成。应把所有临近的已选／未选专家对纳入目标。乘积态的 top-4 第四／第五边界还有结构限制：一般情形下其 Hamming 距离为二或三，单比特邻接没有覆盖这个边界。')
d.p('一个 FP16 临界案例对算术细节敏感：改变 CPU batch 形状改变了一阶预测的符号，FP64 重放改变了实际最终间隔的符号。后续数值评估将保留模型、batch 形状和精度协议的配对记录，并单独开展原生低精度和跨引擎测试。')
d.h('下一阶段：先校准结构，再检验功能收益')
d.p('已经部署独立的交错编码校准：先做四个 RY 编码，经过 CZ 链，再加入另四个输入角度；与移除 CZ 的版本和经典幅度 gate 对比。前置核验中，乘积版本严格合并为四个角度，交错版本在预设测试点具有八个独立概率方向。正式训练将判断这些方向是否提高分数拟合和 top-4 保真。')
d.p('若内部开发集显示稳定改进，下一步对冻结 checkpoint 做真实稀疏重加权，检查 CE、KL、专家利用及舍入扰动，再测推理开销。功能加权边界训练的进入条件是这些结果与预算同时可接受。')
d.page();d.h('四、阶段交付、证据范围与审校重点')
d.p('第二轮已完成效用学习、确定性教师校准、强线性控制、功能几何、精确风险计算及边界重放。模型权重、原始数据、日志和失败方案均保留；保存的 checkpoint 已按各自协议复核，补充材料给出样本、参数、配对区间和审计路径。新增数学／流程测试十三项通过。')
d.table(['已完成的证据','数量／范围'],[['效用预测与确定性教师校准','27＋9 个训练，逐个 checkpoint 重放'],['线性效用与几何控制','9＋9 个模型，36 个拆分重放'],['精确输出风险','3 个冻结 checkpoint，约 118 万次有限替换'],['离散边界机制','24 组重放，固定四位置采样'],['独立汇总核验','源文件与 checkpoint 哈希、有限值、原始数组、配对统计']])
d.p('本阶段 CE/PPL 来自抽样位置的冻结模型替换评估；完整验证／测试集语言模型质量、原生低精度、跨推理引擎和新路由的独立开销测试尚待后续阶段。现有训练耗时是在共享 GPU 环境下记录的过程数据。验证集曾在此前研究中被观察，第三轮属于有记录的适应性探索；测试集继续保留。')
d.p('后台训练和自动核验在凌晨完成，网络中断期间没有完成交互式整理。用户已确认中断原因并授权顺延，本报告作为阶段验收提交；第三轮在独立目录运行，其结果在下一份报告中审校。')
d.h('与已有研究的关系')
d.p('功能敏感的 MoE 稳定性已有直接先例；量子距离编码中保留范数也有成熟思路。本项目应把贡献落到具体结构与可重复的取舍：哪些输入／读出自由度有用，哪些功能教师可低成本学习，以及有限切换的真实质量和时间成本。')
d.link('Puigcerver et al.：On the Adversarial Robustness of Mixture of Experts（NeurIPS 2022）','https://papers.nips.cc/paper_files/paper/2022/file/3effb91593c4fb42b1da1528328eff49-Paper-Conference.pdf')
d.link('Lloyd, Mohseni, Rebentrost：Quantum algorithms for supervised and unsupervised machine learning（2013）','https://arxiv.org/abs/1307.0411')
d.link('Imai, Yang, Pezzè：Semi-classical geometric tensor in multiparameter quantum information','https://arxiv.org/html/2504.06812v3')
d.p('建议审校重点：当前证据是否支持把“路由表征能力”列为第一优先级；逐输入专家对信息的增量是否值得进一步学习；下一阶段的同监督经典对照与进入条件是否足够严格。')
d.save()

a=Document('STAGE_SUPPLEMENT_20261030_zh');a.title('补充材料：协议、逐种子数据与核验')
a.p('对应《MoTE 功能几何与 QuantumGate：阶段审校报告》。主报告负责结论与机制，本补充负责可复核的数值。完整原始结果以本地 CSV／JSON 和 checkpoint 为准。')
a.h('A．协议与样本')
a.p('三个原始 half-MLP 语言模型 checkpoint 对应种子 17／29／43。冻结最后一个 MoTE down 模块及其上游模型，观察 16 个专家、top-4 选择。每个上下文长度 128，只评估位置 31、63、95、127。训练 token 数组前 75% 与后 25% 分别用于拟合和内部开发，窗口和目标不跨分割边界。每种子拟合 32,768 个位置，内部开发与 validation 各 4,096 个位置。')
a.p('效用教师为 realized-label CE 对隐藏状态梯度与专家输出的内积，经专家维度中心化。有限干预从原始已选四专家中移除一个，把其固定质量交给十二个未选专家之一，共 48 个候选。阈值在内部开发集选择，包括保留原始路由；精确重加权的其他协议另行评价。')
a.p('非线性模型固定 4,000 步、batch 256、AdamW 学习率 0.001、weight decay 0.0001、梯度裁剪 1。每 200 步在内部开发集选择最低 MSE checkpoint。每个种子所有模型共用采样顺序。确定性正向控制继承相同初始化和采样。区间为每种子按窗口重采样，保留同窗口位置与候选之间的配对；未做多重比较修正。')
a.table(['seed','dev CE','dev BPE PPL','validation CE','validation BPE PPL'],[[seed,*[f'{next(r for r in S["sampled_baselines"] if r["seed"]==seed and r["split"]==split)[key]:.4f}' for split in ['dev','validation'] for key in ['sampled_ce','sampled_bpe_ppl']]] for seed in [17,29,43]])
a.p('以上是抽样位置的基线均值。PPL=exp(平均 CE)，词表为训练集内 BPE；该表的度量单位与完整 word-level perplexity 不同。多数校准策略保留原始路由，因此抽样 CE/PPL 与基线相同。')
a.page();a.h('B．效用学习与实际干预')
um=[r['method'] for r in S['utility'] if r['study']=='utility' and r['seed']==17 and r['split']=='dev']
a.table(['方法','参数','dev R²：17','29','43'],[[m,next(r['parameters'] for r in S['utility'] if r['method']==m),*[f'{next(r["r2"] for r in S["utility"] if r["method"]==m and r["seed"]==seed and r["split"]=="dev"):.6f}' for seed in [17,29,43]]] for m in um])
a.p('R² 以训练集专家均值作为常数对照。全部非线性候选在内部开发集与 validation 上均为负。除 exp_adapter 的 seed17 在第 400 步选中，其余均在第 200 步选中；训练更久没有改善内部开发 MSE。选中 checkpoint 的训练集 R² 多为很小的正值。')
a.h('期望值适配器的 validation 干预：每种子仅一次')
er=[r for r in S['utility'] if r['method']=='exp_adapter' and r['split']=='validation']
a.table(['seed','ΔCE','95% 配对区间','有害次数 ΔCE>0.1'],[[r['seed'],f'{r["delta_ce"]:+.7f}',f'[{r["ci95_low"]:+.7f}, {r["ci95_high"]:+.7f}]',r['harm_gt_01']] for r in er])
a.p('其他八类非线性模型全部保留原始路由，ΔCE 与配对区间为零。相对于允许不替换的标签 oracle，三个种子的 validation 选择遗憾约为 0.285／0.293／0.273 nat。标签 oracle 使用推理时不可见的目标标签，作为反事实参照。')
a.h('强线性控制')
a.table(['秩','dev R²：17','29','43'],[[rank,*[f'{next(r["r2"] for r in S["utility"] if r["method"]==f"ridge_rank{rank}" and r["seed"]==seed and r["split"]=="dev"):.6f}' for seed in [17,29,43]]] for rank in [4,8,16]])
a.p('秩与正则候选均预先固定，正则只用内部开发集选择。个别校准阈值在 validation 产生少量有害替换；逐次行动和区间保存在 utility_and_ridge.csv。')
a.page();a.h('C．确定性原 gate 教师：逐种子结果')
a.table(['方法','seed','dev R²','validation R²','validation top-4'],[[m,r['seed'],f'{next(x["r2"] for x in S["positive"] if x["method"]==m and x["seed"]==r["seed"] and x["split"]=="dev"):.4f}',f'{r["r2"]:.4f}',f'{r["top4_agreement"]:.2%}'] for m in methods for r in S['positive'] if r['method']==m and r['split']=='validation'])
a.p('validation 三种子的 R² 均值±样本 SD：MLP 0.9828±0.0060，四角 Born 0.3700±0.0605，乘积 Born 0.2422±0.0520。集合一致率均值±SD 为 44.93%±4.42%、3.00%±3.80%、0.57%±0.86%。三种结构的训练参数分别为 10,385／10,285／10,285。')
with (AUDIT/'positive_paired.csv').open(encoding='utf-8-sig') as f:pp=list(csv.DictReader(f))
a.h('相对 MLP 的 validation 配对 R² 差')
a.table(['方法','seed','差值','窗口 95% 区间'],[[r['method'],r['seed'],f'{float(r["paired_difference"]):+.4f}',f'[{float(r["ci95_low"]):+.4f}, {float(r["ci95_high"]):+.4f}]'] for r in pp if r['split']=='validation' and r['metric']=='r2_difference'])
a.p('区间按窗口重采样误差差值，并用固定训练均值基线的 MSE 分母归一化。top-4 的逐种子配对差及区间在 positive_paired.csv。此教师校准的输出为中心化负对数概率；全局可学习缩放参与拟合。')
a.page();a.h('D．功能风险与专家对增量')
a.table(['seed／split','平均 KL','Fisher ρ','Fisher RMSE','绝对误差 P99'],[[f'{r["seed"]}/{r["split"]}',f'{r["mean_kl"]:.4f}',f'{r["fisher_rho"]:.4f}',f'{r["fisher_rmse"]:.4f}',f'{r["fisher_abs_error_p99"]:.4f}'] for r in S['risk']])
a.p('每行 4,096 个位置×48 个候选。平均 KL 和误差使用全部候选；ρ 先在固定源专家的十二个目标内计算再平均。相关度测量排序，误差测量有限步近似偏差；两者需要一起解释。')
a.h('内部开发集：完整距离减能量对照的配对 ρ 差')
a.table(['尺度／gauge','seed','差值','95% 区间'],[[r['comparison'].replace('_distance_minus_energy',''),r['seed'],f'{r["paired_rho_difference"]:+.5f}',f'[{r["ci95_low"]:+.5f}, {r["ci95_high"]:+.5f}]'] for r in S['risk_paired'] if r['split']=='dev' and 'distance_minus_energy' in r['comparison']])
a.p('raw 为原始专家输出，centered 为减去同输入所有专家均值后的输出。全距离对公共输出平移不变；能量与交叉项的分解依赖这个 gauge。两个 gauge 都显示局部增量；静态全局增量更小，部分区间触零或为负。')
a.page();a.h('E．输入条件化几何与数值边界')
a.figure('geometry.png','提高线性秩改善输出草图拟合；预测的专家距离排序仍低于训练集全局距离。内部开发集，三个种子。',65)
a.table(['风险代理：dev ρ','seed17','seed29','seed43'],[[name,*[f'{next(r[key] for r in S["risk"] if r["seed"]==seed and r["split"]=="dev"):.4f}' for seed in [17,29,43]]] for name,key in [('输入草图欧氏距离','predicted_euclidean'),('无锚射线','unanchored_ray'),('锚点×4射线','anchored_ray_x4'),('训练全局距离','train_global_euclidean')]])
a.p('固定锚点由训练集范数分布的第 95 百分位确定，锚点倍数预先指定；对应同信息经典核直接计算。原始八维随机草图是压缩参照，可能含估计噪声，不能当作任意预测器的上界。')
a.table(['输入往返条件','集合变化','固定4/5跨界','一阶 TP／FP／FN'],[[cond,sum(r['set_flips'] for r in S['boundary'] if r['condition']==cond),sum(r['pair_crossings'] for r in S['boundary'] if r['condition']==cond),'/'.join(str(sum(r[k] for r in S['boundary'] if r['condition']==cond)) for k in ['prediction_true_positive','prediction_false_positive','prediction_false_negative'])] for cond in ['bf16_input_roundtrip','fp16_input_roundtrip']])
a.p('计数汇合了 24 个相关模型／拆分重放；每组 4,096 位置，不能视为同等数量的独立试验。BF16 两个漏出固定边界的案例分别为原排名 3→5 与 4→6。FP16 临界案例的 batch／FP64 复核保留在 boundary_exceptions_v1。')
a.page();a.h('F．审计、预算与可复核路径')
a.p('独立阶段审计通过：27 个效用模型、9 个教师校准模型、18 个线性控制、6 组风险数组和 24 组边界结果齐全。风险数组的源／目标／窗口索引与原始有限干预一一对应，精确 NLL 重放一致；当前源文件、快照和冻结 checkpoint 哈希一致。缺失清单与失败清单为空。开发阶段的 smoke 和旧失败 pilot 独立保留，不计入这些正式数量。')
a.table(['正向控制','参数','共享 GPU 秒／run（均值±SD）'],[[names[i],next(r['parameters'] for r in S['positive'] if r['method']==m),f'{S["positive_validation"][m]["shared_gpu_seconds"]["mean"]:.2f} ± {S["positive_validation"][m]["shared_gpu_seconds"]["sd"]:.2f}'] for i,m in enumerate(methods)])
a.p('此耗时包含训练与开发集 checkpoint 选择，记录于共享 TITAN V 环境；独立推理延迟、吞吐和显存对照将按串行协议另测。新 gate 的参数预算应计入它是否附加在原 gate 之后；效用替换策略若部署，需要原 gate 加预测器，而教师蒸馏替换 gate 的预算不同。')
a.p('以下路径均相对于 D:/files/MOTE。完整数字保存在 outputs/operator_geometry_round2/reviews/stage_20261030_v1：utility_and_ridge.csv、positive_control.csv、positive_paired.csv、risk_metrics.csv、risk_paired.csv、geometry.csv、boundary.csv、sampled_ce_ppl.csv、utility_paired_vs_mlp.csv、SUMMARY.json 和 AUDIT.json。')
a.table(['原始证据组','路径（outputs/operator_geometry_round2/ 下）'],[['缓存与有限替换','data/86a2ba9f5aa2'],['效用／教师模型','utility/3a0386d9a187；positive/d4ee19e23787'],['强线性／几何','ridge/01c1341f7586；geometry/input_geometry_v1'],['能量／风险','energy/decomposition_v1；risk/3a17e2fefbb6'],['边界／例外','boundary_gradient/input_sensitivity_v1；reviews/boundary_exceptions_v1'],['数学检验','theory/real_state_top4_v1；theory/metric_bridge_v1']])
a.p('数学推导另见 GEOMETRY_METRIC_NOTE_zh.md／.typ／.pdf。该笔记保留了较早的进度截面；本阶段完成状态以本报告和 STAGE_COMPLETION_AUDIT.json 为准。后续独立结构校准位于 outputs/operator_geometry_round3，方案在 CAPACITY_PLAN.json，预算顺延记录在 round2/EXTENSION_STATE.json。')
a.save()
print('REPORTS_WRITTEN',HERE)
