"""Tables, parameter/accuracy frontiers and a Chinese report for round two."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fitmotn.experiments.gate_benchmark.config import load_config, digest

ROOT=Path(__file__).resolve().parent
C=load_config('experiments/gate_benchmark/configs/pareto_formal.json')
GROUP=Path(C['output_dir'])/digest(C)[:12]
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':175})


def family(gate):
    if gate.startswith('mpo_'):return 'MPO'
    if gate.startswith('amp_'):return '幅度 Born'
    if gate=='compact_born':return '原 Born'
    return 'MLP'


def label(gate):
    if gate=='compact_born':return '原 compact_born'
    text=family(gate)+(' 半预算' if gate.endswith('half') else ' 全预算')
    if gate.startswith('mpo_'):text += ' χ='+str(C['variants'][gate]['bond_dim'])
    if gate.startswith('amp_'):text += ' L='+str(C['variants'][gate]['depth'])
    return text


COLORS={'MLP':'#286FA7','MPO':'#278064','幅度 Born':'#A04498','原 Born':'#C57520'}


def csv_write(path,rows):
    if not rows:return
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def stats(values):
    return st.mean(values), st.stdev(values) if len(values)>1 else None


def main():
    audit=json.loads((GROUP/'audit.json').read_text())
    assert audit['passed']
    subprocess.run([sys.executable,str(ROOT/'recovery_audit.py'),'--require-complete'],check=True)
    incidents=[json.loads(p.read_text(encoding='utf-8-sig'))
               for p in sorted((ROOT/'interruptions').glob('*/incident.json'))]
    interrupted_attempts=[attempt for incident in incidents for attempt in incident['archived_incomplete']]
    for attempt in interrupted_attempts:
        assert (GROUP/attempt['run']/'result.json').exists(),f"Interrupted run not recovered: {attempt['run']}"
    executions=[json.loads(line) for line in (GROUP/'execution.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    latest_starts={event['run']:event for event in executions if event['event']=='start'}
    expected=sum(len(C[s]['tasks']) for s in ['mlp','bert'])*len(C['gates'])*len(C['seeds'])
    records,curves,routes=[],[],[]
    lookup={}
    for p in sorted(GROUP.glob('*/result.json')):
        r=json.loads(p.read_text())
        m=json.loads((p.parent/'manifest.json').read_text())
        initial=json.loads((p.parent/'initial_dev.json').read_text())
        row={k:r[k] for k in ['stage','task','gate','seed','best_step','router_parameters','total_parameters',
                              'training_seconds','wall_seconds','examples_per_second']}
        row.update({f'final_{k}':v for k,v in r['final'].items()})
        row.update(run_dir=str(p.parent),peak_cuda_mib=r['peak_cuda_bytes']/2**20,
                   shared_state_sha256=r['shared_state_sha256'],data_sha256=r['data_sha256'],
                   source_split=m['data']['final_source_split'],best_dev_score=r['best_dev']['score'])
        reference_path=p.parent/'reference.json'
        row['training_origin']='historical_reused' if reference_path.exists() else 'new_training'
        if reference_path.exists():row['historical_source_run']=json.loads(reference_path.read_text())['source_run']
        else:
            row['training_max_concurrent_workers']=latest_starts[p.parent.name]['max_concurrent_workers']
            row['prior_interrupted_attempts']=sum(a['run']==p.parent.name for a in interrupted_attempts)
        for key in ['H_tok','H_dataset','load_cv','unused_experts']:
            row['route_'+key]=st.mean(v[key] for v in r['final_routing'].values())
            row['initial_route_'+key]=st.mean(v[key] for v in initial['routing'].values())
        measure_path=ROOT/'serial_efficiency'/p.parent.name/'measurement.json'
        assert measure_path.exists(),str(measure_path)
        measurement=json.loads(measure_path.read_text())
        assert measurement['final_reevaluation_matches'] and measurement['routing_reevaluation_matches']
        for key in ['seconds_per_step','examples_per_second','peak_cuda_mib']:
            row['serial_'+key]=measurement[key]
        records.append(row)
        lookup[r['task'],r['gate'],r['seed']]=row
        log=[json.loads(line) for line in (p.parent/'curves.jsonl').read_text().splitlines()]
        trains={v['step']:v for v in log if 'train_ce' in v}
        for v in log:
            if 'dev' in v:
                curves.append(dict(stage=r['stage'],task=r['task'],gate=r['gate'],seed=r['seed'],
                    step=v['step'],dev_score=v['dev']['score'],training_seconds=trains[v['step']]['training_seconds']))
        for v in log:
            if 'train_ce' in v:
                curves.append(dict(stage=r['stage'],task=r['task'],gate=r['gate'],seed=r['seed'],
                    step=v['step'],train_ce=v['train_ce'],training_seconds=v['training_seconds']))
        for layer,route in r['final_routing'].items():
            routes.append(dict(task=r['task'],gate=r['gate'],seed=r['seed'],layer=layer,
                **{k:v for k,v in route.items() if not isinstance(v,list)},
                load=json.dumps(route['load']),importance=json.dumps(route['importance'])))
    assert len(records)==expected
    grouped=defaultdict(list)
    for row in records:grouped[row['stage'],row['task'],row['gate']].append(row)
    aggregates,paired=[],[]
    metrics=['final_accuracy','final_f1','final_score','final_ce','best_step','router_parameters',
             'total_parameters','training_seconds','wall_seconds','examples_per_second','peak_cuda_mib',
             'serial_seconds_per_step','serial_examples_per_second','serial_peak_cuda_mib',
             'route_H_tok','route_H_dataset','route_load_cv','route_unused_experts',
             'initial_route_H_tok','initial_route_load_cv','initial_route_unused_experts']
    for (stage,task,gate),rows in grouped.items():
        a=dict(stage=stage,task=task,gate=gate,n=len(rows),label=label(gate))
        for metric in metrics:
            vals=[r[metric] for r in rows if metric in r]
            if vals:a[metric+'_mean'],a[metric+'_std']=stats(vals)
        for base in ['mlp','mlp_half','compact_born']:
            ds=[]
            for row in rows:
                baseline=lookup[task,base,row['seed']]
                assert (row['shared_state_sha256'],row['data_sha256'])==(baseline['shared_state_sha256'],baseline['data_sha256'])
                d=dict(stage=stage,task=task,gate=gate,baseline=base,seed=row['seed'])
                for metric in ['accuracy','score','f1']:
                    if 'final_'+metric in row:
                        d[metric+'_delta_pp']=100*(row['final_'+metric]-baseline['final_'+metric])
                d['router_parameter_ratio']=row['router_parameters']/baseline['router_parameters']
                d['total_parameter_ratio']=row['total_parameters']/baseline['total_parameters']
                d['serial_time_ratio']=row['serial_seconds_per_step']/baseline['serial_seconds_per_step']
                paired.append(d);ds.append(d['score_delta_pp'])
            mean,sd=stats(ds)
            a['delta_vs_'+base+'_pp_mean']=mean;a['delta_vs_'+base+'_pp_std']=sd
            # Descriptive Student interval for three paired seeds, not adjusted for multiple comparisons.
            if len(ds)==3:
                radius=4.3026527299*sd/math.sqrt(3)
                a['delta_vs_'+base+'_95ci_low']=mean-radius
                a['delta_vs_'+base+'_95ci_high']=mean+radius
        baseline=lookup[task,'mlp',rows[0]['seed']]
        a['router_fraction_vs_mlp']=a['router_parameters_mean']/baseline['router_parameters']
        a['total_fraction_vs_mlp']=a['total_parameters_mean']/baseline['total_parameters']
        aggregates.append(a)
    csv_write(ROOT/'formal_runs.csv',records)
    csv_write(ROOT/'formal_aggregate.csv',aggregates)
    csv_write(ROOT/'formal_paired_deltas.csv',paired)
    csv_write(ROOT/'formal_curves.csv',curves)
    csv_write(ROOT/'formal_routing.csv',routes)
    summary=dict(expected=expected,completed=len(records),new_training=sum(r['training_origin']=='new_training' for r in records),
                  reused_historical=sum(r['training_origin']=='historical_reused' for r in records),config=str(GROUP/'config.json'),
                  aggregates=aggregates,paired_deltas=paired,interruption_incidents=incidents,
                  recovered_interrupted_attempts=len(interrupted_attempts))
    (ROOT/'formal_analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    tasks=C['mlp']['tasks']+C['bert']['tasks']
    amap={(a['task'],a['gate']):a for a in aggregates}
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for ax,task in zip(axs.flat,tasks):
        for gate in C['gates']:
            a=amap[task,gate]
            ax.errorbar(a['router_parameters_mean']/1000,100*a['final_score_mean'],
                        yerr=100*a['final_score_std'],fmt='s' if gate.endswith('half') else 'o',
                        color=COLORS[family(gate)],capsize=3,label=label(gate))
        for fam in ['MLP','MPO','幅度 Born']:
            aa=sorted([a for a in aggregates if a['task']==task and family(a['gate'])==fam],key=lambda a:a['router_parameters_mean'])
            ax.plot([a['router_parameters_mean']/1000 for a in aa],[100*a['final_score_mean'] for a in aa],
                    color=COLORS[fam],alpha=.5,lw=1)
        ax.set(title=task.upper(),xlabel='全部路由参数 / 千',ylabel='最终 score / %')
        ax.grid(alpha=.18)
    handles,labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9)
    fig.suptitle('参数—效果：3 个种子的均值 ± 样本标准差；MRPC 使用综合 score')
    fig.tight_layout(rect=[0,.09,1,.96]);fig.savefig(FIG/'parameter_accuracy.png');plt.close(fig)
    half=[g for g in C['gates'] if g.endswith('half')]
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for ax,task in zip(axs.flat,tasks):
        ax.axvline(0,color='#777777',ls='--',lw=1)
        for j,gate in enumerate(half):
            values=[r['score_delta_pp'] for r in paired if r['task']==task and r['gate']==gate and r['baseline']=='mlp']
            ax.scatter(values,j+np.linspace(-.09,.09,len(values)),color=COLORS[family(gate)],alpha=.55)
            ax.errorbar(st.mean(values),j,xerr=st.stdev(values),fmt='D',color=COLORS[family(gate)],capsize=4)
        ax.set(yticks=list(range(len(half))),yticklabels=[label(g) for g in half],title=task.upper(),
               xlabel='相对全预算 MLP 的同 seed 差值 / 百分点')
        ax.grid(axis='x',alpha=.18)
    fig.suptitle('半预算 gate：点为各 seed，菱形为配对均值，横线为样本标准差')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(FIG/'half_budget_paired.png');plt.close(fig)
    for stage in ['mlp','bert']:
        for axis in ['step','training_seconds']:
            fig,axs=plt.subplots(1,3,figsize=(16,4.8))
            for ax,task in zip(axs,C[stage]['tasks']):
                for gate in C['gates']:
                    cc=[r for r in curves if r['task']==task and r['gate']==gate and 'dev_score' in r]
                    steps=sorted(set(r['step'] for r in cc))
                    xx=[st.mean(r[axis] for r in cc if r['step']==s) for s in steps]
                    yy=[100*st.mean(r['dev_score'] for r in cc if r['step']==s) for s in steps]
                    ax.plot(xx,yy,color=COLORS[family(gate)],ls='--' if gate.endswith('half') else '-',label=label(gate),lw=1.7)
                ax.set(title=task.upper(),xlabel='训练步数' if axis=='step' else '并发运行的累计训练秒数',ylabel='内部 dev score / %')
                ax.grid(alpha=.18)
            handles,labels=axs[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
            fig.suptitle(stage.upper()+('：相同步数比较' if axis=='step' else '：实际并发时间曲线，受 GPU 资源竞争影响'))
            fig.tight_layout(rect=[0,.13,1,.94]);fig.savefig(FIG/f'{stage}_dev_{axis}.png');plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(16,8))
    for ax,task in zip(axs.flat,tasks):
        base=amap[task,'mlp']['serial_seconds_per_step_mean']
        vals=[amap[task,g]['serial_seconds_per_step_mean']/base for g in C['gates']]
        ax.barh(range(len(vals)),vals,color=[COLORS[family(g)] for g in C['gates']],alpha=.8)
        ax.axvline(1,color='#666666',ls='--',lw=1)
        ax.set(yticks=range(len(vals)),yticklabels=[label(g) for g in C['gates']],title=task.upper(),xlabel='串行单步耗时 / 全预算 MLP')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
    fig.suptitle('独立串行效率诊断：每个最佳 checkpoint，3 步预热 + 10 步测量；续训权重丢弃')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(FIG/'serial_efficiency.png');plt.close(fig)

    lines=['# 第二轮 gate 参数效率实验报告','',f'正式比较矩阵 **{len(records)}/{expected}** 完成：90 个本轮新训练，36 个同协议历史对照。全部通过独立审计与最终 checkpoint 重评估。',
           '', '本轮比较两个新家族：经典非线性残差 MPO 和幅度编码＋重上传 Born；MLP full/half 与原 compact_born 为对照。',
           '', '## 方法与实验边界','',
           '- residual_mpo：D→r→tanh→E 联合坐标，两层 MPO 残差激活，E→E 读出。它是经典 TN，不能称为物理量子门。',
           '- amplitude_born：非线性编码生成正实初始幅度，归一化后逐层 RY 数据重上传与双向 CRY，联合 Born 读出。它使用实数幺正演化，不含原 compact_born 的 RZ 相位。',
           '- 调整 rank 配平实际全部 router 参数；MPO 的 χ 和 Born 的 depth 由 pilot 内部 dev 选择。半预算组约为全 MLP 的 50%，整数取整以实测参数为准。',
           '- 共享 backbone、专家初态、样本顺序、优化器、步数与评估协议。沿用原 hard+soft STE；因此结果适用于这一既有训练规则。',
           '- 历史对照的四个 pilot 已在新代码中逐项精确复现；原始结果和 checkpoint 带 SHA256 与来源记录，不把复用的训练冒充本轮新训练。全部效率计时在本轮重新串行测量。',
           '- 图像使用官方 test，GLUE 使用官方 validation 作最终评估；内部训练集划出的 dev 单独选 checkpoint。没有使用 GLUE 隐藏 test。',
           '- 新家族有有限结构搜索，MLP 使用固定原架构与学习率，未做各家族等规模的完整超参搜索。',
           '- TITAN V 12GB，PyTorch 2.6.0+cu118，FP32。图像模型宽 1024、4 个残差块，1000 步、batch 128、lr 1e-3；BERT 截取 google/bert_uncased_L-4_H-256_A-4 前 3 层，600 步、batch 16、lr 1e-4。均为 E=16、TopK=4，seed=17/29/43。',
           '- BERT 在替换 FFN 后先做 40 步共同专家拟合，再复制共享初态；第一轮记录的算子近似误差较大，该协议不代表完整保留原预训练 FFN。实验是短程分类基准，不是大型 LLM 生成质量验证。',
           '- 均值与样本标准差使用 3 个种子（ddof=1）。误差条不是等价性证明。CSV 同时给出配对差值；未调整多重比较的 t 区间仅作描述。','',
           '## 最终任务成绩','', '| 任务 | Gate | accuracy % | MRPC F1 % | score % | 对 full MLP 差值 pp |',
           '|---|---|---:|---:|---:|---:|']
    def fmt(a,key,scale=1):
        if key+'_mean' not in a:return '—'
        return f"{scale*a[key+'_mean']:.2f} ± {scale*a[key+'_std']:.2f}"
    for task in tasks:
        for gate in C['gates']:
            a=amap[task,gate]
            lines.append(f"| {task} | {label(gate)} | {fmt(a,'final_accuracy',100)} | {fmt(a,'final_f1',100)} | {fmt(a,'final_score',100)} | {a['delta_vs_mlp_pp_mean']:+.2f} ± {a['delta_vs_mlp_pp_std']:.2f} |")
    lines+=['','![参数效果](figures/parameter_accuracy.png)','', '## 半预算是否保持能力','',
            '以下只描述本次观测，不以“差异不显著”宣称等价。相近的任务仍需更多种子或更长训练验证。','',
            '| 候选 | 平均 score 不低于 full MLP 的任务数 | 平均差值在 ±0.5 pp 内的任务数 | 相对 half MLP 的六任务平均差值 pp |',
            '|---|---:|---:|---:|']
    for gate in half:
        aa=[amap[task,gate] for task in tasks]
        lines.append(f"| {label(gate)} | {sum(a['delta_vs_mlp_pp_mean']>=0 for a in aa)}/6 | {sum(abs(a['delta_vs_mlp_pp_mean'])<=.5 for a in aa)}/6 | {st.mean(a['delta_vs_mlp_half_pp_mean'] for a in aa):+.2f} |")
    lines+=['','这里的跨任务平均只是六项任务的等权描述，不是统一测试集的 accuracy。',
            '', '![半预算配对](figures/half_budget_paired.png)','', '## 参数与效率','',
            '| 阶段/任务 | Gate | router 参数 | router/full MLP | 总参数 | 总参数/full MLP | 串行单步 s | 串行吞吐 samples/s | 峰值 MiB | 最佳步数 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in tasks:
        for gate in C['gates']:
            a=amap[task,gate]
            lines.append(f"| {task} | {label(gate)} | {a['router_parameters_mean']:.0f} | {a['router_fraction_vs_mlp']:.3f} | {a['total_parameters_mean']:.0f} | {a['total_fraction_vs_mlp']:.5f} | {fmt(a,'serial_seconds_per_step')} | {fmt(a,'serial_examples_per_second')} | {fmt(a,'serial_peak_cuda_mib')} | {fmt(a,'best_step')} |")
    lines+=['', '路由参数节省不能等同于整个模型参数节省。完整模型含大量共享骨干与专家参数，整体比例见上表。',
            '', '训练使用多进程共享 GPU。正式日志的训练时间/吞吐包含资源竞争，不用于固有效率排名。上表串行诊断在正式训练结束后进行，仍保留桌面 GPU 进程；是短程续训测量，不是重新完整训练。',
            '', '![串行效率](figures/serial_efficiency.png)','', '## 学习曲线','']
    for stage in ['mlp','bert']:
        for axis in ['step','training_seconds']:
            lines += [f'![{stage} {axis}](figures/{stage}_dev_{axis}.png)','']
    lines+=['## 路由诊断','', '| 任务 | Gate | token 路由熵 | 数据集路由熵 | 负载 CV | 未用专家数 |',
            '|---|---|---:|---:|---:|---:|']
    for task in tasks:
        for gate in C['gates']:
            a=amap[task,gate]
            lines.append(f"| {task} | {label(gate)} | {fmt(a,'route_H_tok')} | {fmt(a,'route_H_dataset')} | {fmt(a,'route_load_cv')} | {fmt(a,'route_unused_experts')} |")
    lines+=['','每层指标先按有效 token 聚合，再在路由层间取平均；表中未用专家数是层与种子的平均值，允许非整数。']
    if incidents:
        lines+=['','## 中断与恢复记录','',
                f'共保留 {len(incidents)} 次系统中断事件、{len(interrupted_attempts)} 个中断训练现场；这些尝试不作为完成结果，也未从记录中删除。全部对应 run 已从相同种子和冻结配置重新完整训练。',
                '', '2026-10-27 19:33 左右 Windows 在 bugcheck 0x00000133 后重启，具体驱动原因尚未确定。重启前 4 个新增完成结果及 36 个历史对照逐文件 SHA256 核验未变，6 个中断 run 的日志与 checkpoint 已移至 interruptions/20261027_193345_system_reboot/ 并校验归档 hash。',
                '', '恢复后将调度并发上限从 6 降到 3，训练模型、配置、批次、优化步数和 RNG 协议不变。并发上限记于 formal_runs.csv；不同资源竞争下的训练秒数不作直接效率排名。所有最终 checkpoint 另用同一串行协议重测。',
                '', '中断及重跑消耗的时间属于实验执行开销，未计入成功 run 的完整训练耗时；原始尝试与 Windows 事件见 interruptions/，执行事件见正式组 execution.jsonl。']
    lines+=[
            '', '## 实测解释与推荐','',
            '此处需结合配对差值、效率、路由与各 seed 明细填写人工审阅结论；不由自动脚本按最高均值宣布胜者。',
            '', '## 可复现产物','',
            f'- 冻结配置：`{GROUP / "config.json"}`；完整性审计：`{GROUP / "audit.json"}`。',
            '- `formal_runs.csv`、`formal_aggregate.csv`、`formal_paired_deltas.csv`：单 run、均值/SD、对三个基线的配对差值。',
            '- `formal_curves.csv`、`formal_routing.csv`、`formal_analysis.json`：学习曲线、逐层路由与原始汇总。',
            '- `serial_efficiency/`：每个 checkpoint 的最终预测、路由重评估与独立串行计时。',
            '- `source_snapshot_v1/`、`RUN_LOG_zh.md`、`logs/`：代码、修改来源与日志；模型/数据不提交 Git。','']
    (ROOT/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'completed':len(records),'figures':len(list(FIG.glob('*.png'))),'report':str(ROOT/'REPORT_zh.md')}))


if __name__=='__main__':main()
