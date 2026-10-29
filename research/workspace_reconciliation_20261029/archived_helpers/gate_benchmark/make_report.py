"""Generate the factual Chinese report only after all audits and timings pass."""
import datetime
import json
from pathlib import Path
import statistics as st
from fitmotn.experiments.gate_benchmark.config import load_config, digest

root = Path('outputs/gate_benchmark')
c = load_config('experiments/gate_benchmark/configs/titan_v.json')
group = root / digest(c)[:12]
audit = json.loads((group / 'audit.json').read_text())
assert audit['passed'] and audit['completed_audited'] == 54
analysis = json.loads((root / 'formal_analysis.json').read_text())
assert analysis['completed'] == 54
eff = json.loads((root / 'serial_efficiency_summary.json').read_text())
assert len(eff['runs']) == 54
assert all(r['final_reevaluation_matches'] and r['routing_reevaluation_matches'] for r in eff['runs'])
pilot = json.loads((root / '7d5c8d3cbd27/audit.json').read_text())
assert pilot['passed']
equiv = json.loads(Path('outputs/gate_parallel_pilot/8463c90d231e/serial_equivalence.json').read_text())
assert equiv['passed']
summary = json.loads((root / 'summary.json').read_text())
assert not summary['missing_runs'] and not summary['incomplete_runs']
aggregates = {(x['task'], x['gate']): x for x in analysis['aggregates']}
effgroups = {(x['task'], x['gate']): x for x in eff['groups']}
learning = json.loads((root / 'learning_diagnostics.json').read_text())
assert len(learning['runs']) == 54
learning_groups = {(x['task'], x['gate']): x for x in learning['aggregates']}
runs = [json.loads(p.read_text()) for p in sorted(group.glob('*/result.json'))]
manifests = [json.loads(p.read_text()) for p in sorted(group.glob('*/manifest.json'))]
tasks = c['mlp']['tasks'] + c['bert']['tasks']
gates = c['gates']
labels = {'mlp': 'MLP', 'compact_born': '联合 Born', 'mps': '经典 MPS'}
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
session_hours = (datetime.datetime.fromisoformat(now) - datetime.datetime.fromisoformat('2026-10-27T01:45:00+08:00')).total_seconds()/3600
events = [json.loads(x) for x in (group / 'execution.jsonl').read_text().splitlines()]
starts = [datetime.datetime.fromisoformat(x['time']) for x in events if x['event'] == 'start']
ends = [datetime.datetime.fromisoformat(x['time']) for x in events if x['event'] == 'end']
elapsed = (max(ends) - min(starts)).total_seconds()/3600


def ms(a, metric, scale=1, digits=2):
    m, s = a[metric+'_mean']*scale, a[metric+'_std']
    return f'{m:.{digits}f} ± {s*scale:.{digits}f}' if s is not None else f'{m:.{digits}f}（n=1）'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


lines = [
    '# TITAN V 上 MoTE 三种 gate 对比实验', '', f'报告生成：{now}（Asia/Shanghai）', '',
    '**完成状态：原始 pilot 6/6、并行等价性复核 6/6、默认正式矩阵 54/54、单训练进程效率诊断 54/54；正式失败、缺失及未完成 run 均为 0。**', '',
    '## 1. 执行、版本与验收', '',
    '- 仓库 `https://github.com/intproduct/MOTE.git`，分支 `mote-quantum`，Git commit `807366cb8afc58b1fc37086f998ae791e8bb16a6`。未修改仓库训练代码；所有执行与分析辅助脚本位于忽略的 outputs 目录。',
    '- NVIDIA TITAN V 12 GB / Windows 11 WDDM / 驱动 582.28；Xeon E5-2680 v4（14 核/28 线程）、32 GB RAM；Python 3.11.18、PyTorch 2.6.0+cu118、CUDA runtime 11.8、Transformers 4.57.6、Datasets 3.6.0、NumPy 2.2.6。完整依赖见 `logs/pip_freeze.txt`。nvidia-smi 显示的 CUDA 13.0 是驱动支持上限。',
    '- 独立环境 `.venv-titan`；FP32、严格确定性、TF32 关闭，不使用 BF16、torch.compile 或随机模型替代预训练模型。保留已有桌面 GPU 进程。',
    '- doctor 验证 64 专家三种 router 与 MoTE CUDA backward；相关回归 128 passed；6 组 CUDA 合成 smoke 通过。首次 pytest 有 10 个系统临时目录权限错误，保留日志并换项目内临时目录后完整重跑通过。',
    '- pilot 独立审计最初误用 `drop_rate == 0`，BERT FP32 归约给出约 1.43e-8。直接检查每组 77,700 个层内有效 token、310,800 次分派，实际违反 Top-k=4 的 token 为 0，重评估指标完全复现；修正浮点审计容差后通过。原始错误报告全部保留。',
    '- 两个局部计算优化候选虽通过逐位等价验证但无稳定明显加速，均未采用。串行 pilot 外推约 10 小时，因此用独立进程并行执行默认矩阵。原始串行与三进程 pilot 的所有记录点、最终指标和 checkpoint 均逐位一致。',
    f'- 正式分组 `{group.name}`，最多 6 个并行 GPU 进程，先完成 MLP 27 runs 再进入 BERT 27 runs。主矩阵实际执行窗口 {min(starts).isoformat(timespec="seconds")} 至 {max(ends).isoformat(timespec="seconds")}，约 {elapsed:.2f} 小时（不含前期安装/pilot及后续审计）。',
    '- 并行训练的单 run 秒数与吞吐包含资源竞争，不能当作单进程 GPU 的算法效率。另对全部 54 个最佳 checkpoint 的副本逐个测量统一训练步吞吐；正式模型和分数不被改写。', '',
    '## 2. 协议与数据隔离', '',
    '每个图像 run 为宽度 1024、4 个残差块、16 专家/Top-k=4、batch=128、1000 步；每个文本 run 使用 `google/bert_uncased_L-4_H-256_A-4`，截取前三层，hidden=256、batch=16、max_length=128、600 步。三个种子为 17/29/43。router 预算采用 matched，整数宽度取整带来少量差异，实际参数量见下表。', '',
    'AdamW 学习率分别为图像 1e-3、文本 1e-4，weight_decay=0.01，梯度范数裁剪阈值 1.0；每 100 步在内部 dev 评估并选模，每 25 步记录训练诊断；step 0 的初始 dev 仅作诊断，不参与选模。预训练 commit 为 `387825ce42dbb39b87911cdf8e383ee3b25184f8`，记录于 `../gate_cache/81d9d986c6a6333d/bert.json`；数据文件 hash 位于各 run 的 manifest。', '',
    '三种分数生成器分别为：MLP（两层线性层与 SiLU）；compact_born（4 比特、depth=2 的 RY/RZ 与双向相邻 CRY，读取完整 16 维联合 Born 概率并取平滑对数）；经典 MPS（16 sites、bond_dim=8 的归一化残差矩阵乘积特征）。本实验未使用旧的一专家一个 Z 期望值的 quantum gate。', '',
    '共同外壳是输入 LayerNorm、可学习正 gain、temperature=1 和原 Top-k 路由；gate dropout/jitter=0（BERT 主干及分类头保留预训练配置的 dropout=0.1），balance 系数 0.01，z-loss=0，drop_tokens=false。matched 只近似匹配可训练 router 参数，不表示 FLOPs 或运行时间相同。', '',
    '三种 gate 共享每任务/种子的 backbone、专家、分类头初态与数据 batch 顺序；BERT 公共专家 warm initialization 为 40 步，与标签无关。共享 state hash、数据文件 hash、专家位置和参数预算均已独立审计。', '',
    '以下“最终”指标是固定最大训练步数内，按内部 dev 最佳 score 选出的 checkpoint 在最终 split 上的指标，并非最后一步权重的指标。MRPC score=(accuracy+F1)/2，其余 score=accuracy。所有均值与标准差均跨三个种子计算，标准差为样本标准差（ddof=1）；百分点差值为同种子配对差值。', '',
    table(['任务', '训练', '内部 dev', '最终评估', '最终来源'], [[t, audit['data'][t]['sizes']['train'], audit['data'][t]['sizes']['dev'], audit['data'][t]['sizes']['test'], audit['data'][t]['source_split']] for t in tasks]), '',
    '内部 dev 从官方 train 按标签分层划分，data_seed=20261027；train/dev 索引无交集。图像最终 split 为官方 test 的分层子集；GLUE 最终 split 是官方 validation，不参与 checkpoint 选择。未使用无标签的 GLUE 隐藏 test 计算或声称榜单成绩。', '',
    '## 3. 最终任务成绩', '',
    '单位为百分数，均值 ± 样本标准差。MRPC 在此表显示综合 score，accuracy 和 F1 单列于下一表。', '',
    table(['任务/指标', 'MLP', '联合 Born', '经典 MPS'], [[t+(' / score' if t == 'mrpc' else ' / accuracy')] + [ms(aggregates[t,g], 'final_score', 100) for g in gates] for t in tasks]), '',
    table(['MRPC gate', 'Accuracy (%)', 'F1 (%)', 'Score (%)'], [[labels[g]] + [ms(aggregates['mrpc',g], k, 100) for k in ('final_accuracy','final_f1','final_score')] for g in gates]), '',
    '### 同种子相对 MLP 的配对差值', '',
    '单位：百分点；MRPC 为综合 score 差值，其他任务为 accuracy 差值。accuracy、F1 和 score 的完整配对记录见 `formal_paired_deltas.csv`。', '',
]
paired_rows = []
for t in tasks:
    for g in ('compact_born', 'mps'):
        dd = sorted([x for x in analysis['paired_deltas'] if x['task']==t and x['gate']==g], key=lambda x:x['seed'])
        a = aggregates[t,g]
        paired_rows.append([t, labels[g], *[f"{x['final_score_delta_pp']:+.2f}" for x in dd],
                            f"{a['paired_score_delta_pp_mean']:+.2f} ± {a['paired_score_delta_pp_std']:.2f}"])
lines += [table(['任务','gate','seed 17','seed 29','seed 43','均值 ± SD'], paired_rows), '',
          '## 4. 参数、时间、吞吐与显存', '',
          '此表的训练秒数/吞吐为主矩阵实测，包含最多六进程的 GPU 资源竞争。训练秒数包括取 batch、搬运、前后向与 optimizer；不含 dev 评估。wall_seconds 额外包含训练期间 dev、checkpoint 和最终评估，但不含模型加载、公共初始化和 initial dev。原始逐 run wall_seconds 见扩展 CSV。峰值 MiB 是 PyTorch 分配量，覆盖训练与评估，不等于整卡占用；整卡遥测见 logs/gpu_telemetry.csv。', '',
          table(['任务','gate','router 参数','总参数','训练秒（并行）','样本/秒（并行）','峰值 MiB','best step（17/29/43）'],
                [[t,labels[g],int(aggregates[t,g]['router_parameters_mean']),int(aggregates[t,g]['total_parameters_mean']),
                  ms(aggregates[t,g],'training_seconds',digits=1),ms(aggregates[t,g],'examples_per_second',digits=1),
                  ms(aggregates[t,g],'peak_cuda_mib',digits=1), '/'.join(str(x['best_step']) for x in sorted([r for r in runs if r['task']==t and r['gate']==g],key=lambda x:x['seed']))]
                 for t in tasks for g in gates]), '',
          '### 单训练进程的统一效率诊断', '',
          '此诊断中本实验同时只有一个训练进程；已有桌面 GPU 进程仍保留，并非完全专用设备测量。逐个加载最佳 checkpoint 的副本，使用相同任务/种子的训练 batch 顺序与原 batch size；新建 AdamW，3 步预热后测量 10 个完整优化步。包含取数、搬运、前向、反向、梯度裁剪与 optimizer。短程延续后的副本丢弃，不用于任何 accuracy 结论。此测量描述选中 checkpoint 附近的运行效率，不能替代从头训练全程的单进程计时。', '',
          table(['任务','gate','单进程样本/秒','单进程秒/步','单进程峰值 MiB'], [[t,labels[g],ms(effgroups[t,g],'examples_per_second',digits=1),ms(effgroups[t,g],'seconds_per_step',digits=3),ms(effgroups[t,g],'peak_cuda_mib',digits=1)] for t in tasks for g in gates]), '',
          '## 5. 路由与训练稳定性', '',
          '每个 result 的路由统计先在完整最终 split 上对有效 token 加权，然后本表对模型各 MoTE 层取平均、再跨种子统计。H_tok 是最终 routing softmax 专家概率的单 token Shannon 熵，H_dataset 是数据集平均专家概率分布熵，均为自然对数单位；16 专家最大熵 ln(16)≈2.773。负载 CV 基于实际 Top-k 分派。unused 为每层在最终评估 split 上未获任何分派的专家个数，层间平均后可为小数。', '',
          table(['任务','gate','H_tok','H_dataset','负载 CV','unused/层'], [[t,labels[g],*[ms(aggregates[t,g],'route_mean_'+k,digits=3) for k in ('H_tok','H_dataset','load_cv','unused_experts')]] for t in tasks for g in gates]), '',
          '所有正式 run 完成了计划步数，训练程序逐步检查 finite loss，FP32 梯度裁剪使用 error_if_nonfinite；独立审计复查所有日志、最佳权重有限值、参数更新、dev 选模和最终文件。AMP skipped steps 全为 0（本实验为 FP32）。逐层 importance/load 向量与统计保留在 formal_routing_per_layer.csv 和 result.json。', '',
          '训练结束后另行逐个加载全部 54 个最佳 checkpoint，串行重新评估最终 split：accuracy/F1/score 完全复现，CE 与逐层路由统计在 1e-6 容差内复现，token 数及未使用专家数完全一致。每组的真实/预测类别数量、混淆矩阵和逐样本预测保存在 serial_efficiency/<run>/final_predictions.json，便于检查多数类预测等退化。', '',
          '## 6. 学习曲线与效率图', '',
          '相同步数下的比较使用内部 dev 曲线；时间轴曲线是主矩阵实际并行运行时间，包含资源竞争，不据此独立宣称计算效率优势。单进程效率应结合上一节的统一诊断。浅线代表单个种子，粗线为三种子均值。MLP 另附省略 step 0 的放大曲线，以便辨认训练后的小幅差异；完整初始点保留在全程图与 CSV 中。', '']
for filename in ['mlp_dev_vs_step','mlp_dev_vs_step_trained','bert_dev_vs_step','mlp_dev_vs_training_seconds','bert_dev_vs_training_seconds', 'mlp_train_ce','bert_train_ce','outcome_efficiency_memory','paired_score_deltas','routing_diagnostics','isolated_efficiency']:
    lines += [f'![{filename}](figures/{filename}.png)', '']
lines += ['## 7. 结论边界与后续验证', '',
          '以下观察均限于这组匹配预算、固定初始化、固定短程步数和相同评估协议。三种 gate 在同种子下的增减见配对表；三个种子不足以支持宽泛的显著性或普适性宣称。任何更高均值都不能单独证明量子优势或纠缠贡献；compact_born 在此是 GPU 上的精确小状态向量模拟，没有真实量子硬件运行。', '']
for t in tasks:
    parts=[]
    for g in ('compact_born','mps'):
        a=aggregates[t,g]
        dd=[x['final_score_delta_pp'] for x in analysis['paired_deltas'] if x['task']==t and x['gate']==g]
        parts.append(f"{labels[g]}相对 MLP 为 {a['paired_score_delta_pp_mean']:+.2f} ± {a['paired_score_delta_pp_std']:.2f} 个百分点，三个种子中 {sum(v>0 for v in dd)} 个提高、{sum(v<0 for v in dd)} 个降低、{sum(v==0 for v in dd)} 个持平")
    lines.append(f'- {t}：'+'；'.join(parts)+'。')
lines += ['', '### 成绩与计算成本的联合观察', '']
for gate in ('compact_born', 'mps'):
    ratios = [effgroups[t,gate]['seconds_per_step_mean']/effgroups[t,'mlp']['seconds_per_step_mean'] for t in tasks]
    higher = [t for t in tasks if aggregates[t,gate]['paired_score_delta_pp_mean'] > 0]
    consistent = [t for t in tasks if all(x['final_score_delta_pp'] > 0 for x in analysis['paired_deltas'] if x['task']==t and x['gate']==gate)]
    lines.append(f"- {labels[gate]}：最终 score 均值高于 MLP 的任务为 {('、'.join(higher)) or '无'}；三个配对种子都提高的任务为 {('、'.join(consistent)) or '无'}。单训练进程诊断的平均秒/步相对 MLP 为 {min(ratios):.2f}–{max(ratios):.2f} 倍（按任务分别求均值之比）。此处只描述当前六任务和短程预算，分数差值与速度比都应结合各自标准差读取。")
endpoint_counts = {stage: sum(r['stage']==stage and r['best_step']==c[stage]['steps'] for r in runs) for stage in ('mlp','bert')}
lines += ['', f"最佳 checkpoint 落在最后计划评估点的 run：图像 {endpoint_counts['mlp']}/27、BERT {endpoint_counts['bert']}/27。固定预算及离散 dev 选模本身不足以确认充分收敛；这部分结果应结合学习曲线，并在后续独立分组中用延长训练验证。"]
mnist_mlp, mnist_mps = aggregates['mnist','mlp'], aggregates['mnist','mps']
max_unused_mps = max(v['unused_experts'] for r in runs if r['task']=='mnist' and r['gate']=='mps' for v in r['final_routing'].values())
lines += ['',
    f"MNIST 上，经典 MPS 的平均 H_tok={mnist_mps['route_mean_H_tok_mean']:.3f}，接近 ln(16)，但负载 CV={mnist_mps['route_mean_load_cv_mean']:.3f}，MLP 对应为 {mnist_mlp['route_mean_load_cv_mean']:.3f}；MPS 的单层未使用专家数最大为 {max_unused_mps}。这说明本组实验中，接近均匀的概率分布仍可伴随不均衡的 Top-k 分派。离散 Top-k 对相对排序敏感，微小概率差异也可能改变专家入选频率。此观察并未单独确定精度差异的成因。", '',
    '不同任务的提升或退化可能涉及优化、路由分配和对数据的拟合方式；当前矩阵没有分别控制这些因素，不能把相关现象直接解释为某种 gate 结构的因果优势。后续机制结论需要同预算的专门对照。']
bert_born = [aggregates[t,'compact_born'] for t in c['bert']['tasks']]
bert_mlp = [aggregates[t,'mlp'] for t in c['bert']['tasks']]
lines += ['',
    f"BERT 的三个任务也有类似但更明显的现象：联合 Born 的平均 H_tok 在 {min(x['route_mean_H_tok_mean'] for x in bert_born):.3f}–{max(x['route_mean_H_tok_mean'] for x in bert_born):.3f}，但实际负载 CV 为 {min(x['route_mean_load_cv_mean'] for x in bert_born):.3f}–{max(x['route_mean_load_cv_mean'] for x in bert_born):.3f}，每层平均有 {min(x['route_mean_unused_experts_mean'] for x in bert_born):.2f}–{max(x['route_mean_unused_experts_mean'] for x in bert_born):.2f} 个专家在最终评估中未被分派。MLP 的 CV 为 {min(x['route_mean_load_cv_mean'] for x in bert_mlp):.3f}–{max(x['route_mean_load_cv_mean'] for x in bert_mlp):.3f}，未使用专家数均为 0。数值稳定、参数确实更新，并不等于所有专家都被充分使用；目前还不能把最终分数差异归因于负载不均衡。", '',
    '### 学习曲线中的具体观察', '',
    'CIFAR-10 最后五个已记录训练 batch 的 CE，先在每个种子内平均，再跨种子统计：' + '；'.join(f"{labels[g]} {ms(learning_groups['cifar10',g], 'last5_logged_batch_ce_mean', digits=3)}" for g in gates) + '。该统计不是完整训练集评估；它表明本轮固定步数下 Born 的后期 batch 拟合程度较低，而最终 accuracy 的差异只有零点几个百分点，尚不足以单独判定过拟合、优化不足或结构限制。', '',
    'SST-2 的内部 dev 最佳 accuracy 分别为 ' + '、'.join(f"{labels[g]} {ms(learning_groups['sst2',g], 'best_dev_score', 100)}%" for g in gates) + '，最终官方 validation 为 ' + '、'.join(f"{labels[g]} {ms(aggregates['sst2',g], 'final_accuracy', 100)}%" for g in gates) + '。这是不同划分上的测量，学习曲线中的约 84% 不能替代最终表中的约 78%；本轮没有用官方 validation 回选 checkpoint，也未据此重新调参。差距的来源尚未通过独立实验分解。', '',
    'RTE 的内部 dev 曲线在后期没有稳定持续改善，最佳 checkpoint 常早于最后一步；因此最终结果采用内部 dev 选模，而不是把最后一步分数当作最优表现。完整初始点、逐次 dev 评估与实际训练时间均已保留。']
warm=[]
for m in manifests:
    if m['task']=='sst2' and m['gate']=='mlp':
        warm.extend(m['warm_initialization'].values())
majority_rows = []
collapsed = []
for task in c['bert']['tasks']:
    counts = audit['data'][task]['label_counts']
    majority = max(range(len(counts['train'])), key=lambda k: counts['train'][k])
    acc = counts['test'][majority]/sum(counts['test'])
    f1 = 2*counts['test'][1]/(sum(counts['test'])+counts['test'][1]) if majority == 1 else 0.
    score = (acc+f1)/2 if task == 'mrpc' else acc
    majority_rows.append([task,majority,f'{100*acc:.2f}',f'{100*f1:.2f}' if task=='mrpc' else '—',f'{100*score:.2f}'])
    for gate in gates:
        for seed in c['seeds']:
            name=f'bert_{task}_{gate}_seed{seed}'
            pred=json.loads((root/'serial_efficiency'/name/'final_predictions.json').read_text())
            if sum(v>0 for v in pred['prediction_counts']) == 1:
                collapsed.append(name)
lines += ['', '### 文本任务的常数预测参照', '',
          '只依据训练集标签数量选择多数类，然后在最终 split 上恒定预测该类；这是诊断参照，不是额外训练 run，不用最终标签选择多数类。MRPC 的不平衡标签会使常数预测也获得较高 F1，应结合 accuracy 与混淆矩阵判断。', '',
          table(['任务','训练多数类','accuracy (%)','MRPC F1 (%)','score (%)'],majority_rows), '',
          ('最终 split 只预测一个类别的正式 run：'+ '、'.join(collapsed)+'。') if collapsed else '全部 27 个文本正式 run 在最终 split 上均至少预测两个类别；逐样本预测和各类别召回情况可从保存的混淆矩阵复核。']
lines += ['', f"BERT 公共专家初始化的 held-out Gaussian probe 相对 MSE：拟合前均值 {st.mean(v['relative_mse_before'] for v in warm):.4f}，40 步后均值 {st.mean(v['relative_mse_after'] for v in warm):.4f}，拟合后范围 {min(v['relative_mse_after'] for v in warm):.4f}–{max(v['relative_mse_after'] for v in warm):.4f}（SST-2 三个共享种子、每个模型六个 MoTE 投影；同种子三种 gate 共用）。该误差衡量 Gaussian 探针上的线性算子近似，显示公共拟合仍不充分；其在任务分布上的影响需要额外测量。因此，当前矩阵不能把整体 BERT 表现的限制全部归因于 gate，还需要保留 dense FFN 的参考对照。", '',
          '后续优先考虑：相同大小的三种 gate 学习率搜索与更多种子；增加公共 FFN 初始化预算或加入保留 dense FFN 的参考；随后独立分组进行 2 层 BERT、64 专家/Top-k=8、同 rank 对照和 entangle=false 消融。本轮没有把这些不同设置混入默认矩阵。', '',
          '## 8. 文件索引与未解决事项', '',
          '- `45c21049b433/`：默认正式矩阵、config.json、各 run 的 manifest/initial_dev/curves/best.pt/result、audit.json 和 execution.jsonl。',
          '- `7d5c8d3cbd27/`：原始串行 pilot、独立审计和整数 mask 复核。`../gate_parallel_pilot/8463c90d231e/`：并行 pilot 与逐位等价性证据。',
          '- `runs.csv`、`summary.json`：仓库 summarize 原始汇总，按配置分组；正式缺失和未完成列表均为空。',
          '- `formal_runs_extended.csv`、`formal_aggregate.csv`、`formal_paired_deltas.csv`、`formal_analysis.json`：正式完整指标、均值/SD及配对差值。',
          '- `formal_dev_curves.csv`、`formal_routing_per_layer.csv`：学习曲线与逐层路由。',
          '- `learning_diagnostics.csv/json`：后期已记录训练 batch CE、内部 dev 与最终评估的独立汇总。',
          '- `serial_efficiency_runs.csv`、`serial_efficiency_summary.csv/json`、`serial_efficiency/`：单进程效率测量及逐步时间。',
          '- `figures/`、`logs/`、`RUN_LOG_zh.md`：图表、版本/驱动/安装/测试/失败与训练日志、执行决策记录。',
          '- 未解决的训练失败或缺失：无。研究限制：三个种子、短程预算、BERT FFN 近似误差、无 dense BERT 参考、无纠缠消融；并行主训练计时有竞争，单进程效率诊断是短程测量而非全程单进程重跑。',
          '- 数据、预训练模型、checkpoint 均留在本机忽略目录，未提交到 Git。', '']
report = '\n'.join(lines)
report = report.replace('原始串行与三进程 pilot 的所有记录点、最终指标和 checkpoint 均逐位一致。',
                        '原始串行与三进程 pilot 的全部学习曲线数值（不含计时）、最终指标和 checkpoint 均逐位一致。')
report = report.replace('完整依赖见 `logs/pip_freeze.txt`。',
                        '完整依赖见 `logs/pip_freeze_final.txt` 与 `logs/dependency_versions_final.json`。')
report = report.replace('平均 H_tok 在 2.770–2.770', '平均 H_tok 约为 2.770')
report = report.replace('## 2. 协议与数据隔离',
                        f'从 01:45 开始环境检查到本报告生成约 {session_hours:.2f} 小时，超过原先 5–6 小时规划；为保全默认矩阵与复核，没有缩减训练预算。10 秒间隔整卡遥测的采样峰值为 8,046 MiB、60°C，包含桌面及本实验所有进程；无 OOM。\n\n## 2. 协议与数据隔离')
(root / 'REPORT_zh.md').write_text(report, encoding='utf-8')
print(root / 'REPORT_zh.md')
