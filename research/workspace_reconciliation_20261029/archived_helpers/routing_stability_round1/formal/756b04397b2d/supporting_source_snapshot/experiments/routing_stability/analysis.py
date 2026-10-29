from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st
from .runner import read_json, canonical_hash, audit, write_json, timestamp

LABELS={'mlp_half':'MLP half','mlp_margin':'MLP + margin',
        'amp_half':'Born original','amp_margin':'Born original + margin',
        'born_lite':'Born shared','born_lite_margin':'Born shared + margin'}
COLORS=['#2563eb','#60a5fa','#b45309','#f59e0b','#15803d','#4ade80']


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def aggregate(rows,keys):
    out={}
    for key in keys:
        values=[r[key] for r in rows if r[key] is not None]
        out[key+'_mean']=st.mean(values) if values else None
        out[key+'_std']=st.stdev(values) if len(values)>1 else 0.
        if len(values)!=len(rows):out[key+'_defined_n']=len(values)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--root',default='outputs/routing_stability_round1');args=parser.parse_args()
    root=Path(args.root).resolve();cfg=read_json(args.config)
    group=root/cfg['phase']/canonical_hash(cfg)[:12]
    checked=audit(group,cfg);assert checked['passed'] and cfg['phase']=='formal'
    rows=[];curves=[];stability=[];tails=[];routing=[];generations=[];missing=[]
    for seed in cfg['seeds']:
        for variant in cfg['gates']:
            path=group/'runs'/f'{variant}_s{seed}';r=read_json(path/'result.json')
            absent=[f for f in ['evaluation.json','stability.json'] if not (path/f).exists()]
            if absent:
                missing.append({'run':path.name,'missing':absent});continue
            e=read_json(path/'evaluation.json');p=read_json(path/'stability.json')
            assert e['checkpoint_sha256']==p['checkpoint_sha256']==r['checkpoint_sha256']
            row={'variant':variant,'seed':seed,'test_ce':e['test']['ce'],'test_ppl':e['test']['ppl'],
                 'dev_ce':r['best_validation']['ce'],'dev_ppl':r['best_validation']['ppl'],
                 'router_parameters':r['parameters']['router_total'],'total_parameters':r['parameters']['total'],
                 'best_step':r['best_step'],'steps':r['steps'],'tokens_seen':r['tokens_seen'],
                 'training_seconds':r['training_seconds'],'wall_seconds':r['wall_seconds'],
                 'training_tokens_per_second':r['training_tokens_per_second'],
                 'serial_step_seconds':e['serial_benchmark']['mean_step_seconds'],
                 'serial_tokens_per_second':e['serial_benchmark']['tokens_per_second'],
                 'peak_allocated_mb':e['serial_benchmark']['peak_allocated_mb'],
                 'checkpoint_sha256':r['checkpoint_sha256'],'shared_init_hash':r['shared_init_hash']}
            for b in e['inference_benchmark']:
                row[f'inference_b{b["batch"]}_l{b["length"]}_ms']=1000*b['median_seconds']
            for key in ['entropy','load_cv','unused_experts']:
                row[key]=st.mean(v[key] for v in e['test']['routing'].values())
            for module,values in e['test']['routing'].items():
                routing.append({'variant':variant,'seed':seed,'module':module,**values})
            for condition in sorted({v['condition'] for v in p['rows']}):
                sub=[v for v in p['rows'] if v['condition']==condition]
                n=sum(v['n'] for v in sub);changed=sum(v['changed'] for v in sub)
                summary={'variant':variant,'seed':seed,'condition':condition,'decisions':n,'changed':changed,
                         'flip_percent':100*changed/n,'replaced_percent':100*sum(v['replaced'] for v in sub)/(n*cfg['router']['topk']),
                         'measured_bound_safe_percent':100*sum(v['measured_bound_safe'] for v in sub)/n,
                         'max_score_difference':max(v['max_score_difference'] for v in sub)}
                stability.append(summary);row['flip_'+condition+'_percent']=summary['flip_percent']
                if condition=='repeat':assert changed==0
            dist=p['gap_distributions']
            row['gap_mean']=st.mean(v['gap_mean'] for v in dist)
            row['gap_p05_batch_mean']=st.mean(v['gap_p05'] for v in dist)
            row['gap_below_threshold_percent']=100*sum(v['below_threshold'] for v in dist)/sum(v['n'] for v in dist)
            row['boundary_ties']=sum(v['ties'] for v in dist)
            for condition in sorted({v['condition'] for v in p['end_to_end']}):
                sub=[v for v in p['end_to_end'] if v['condition']==condition];n=sum(v['n'] for v in sub)
                tail={'variant':variant,'seed':seed,'condition':condition,'predictions':n,
                      'target_logprob_mae':sum(v['target_logprob_abs_sum'] for v in sub)/n,
                      'target_logprob_max':max(v['target_logprob_abs_max'] for v in sub),
                      'target_logprob_p99_batch_mean':st.mean(v['target_logprob_abs_p99'] for v in sub),
                      'target_logprob_gt01_percent':100*sum(v['target_logprob_abs_gt_01'] for v in sub)/n,
                      'argmax_changed_percent':100*sum(v['next_token_changed'] for v in sub)/n,
                      'max_vocab_logit_difference':max(v['max_vocab_logit_difference'] for v in sub)}
                tails.append(tail)
                for key in ['target_logprob_mae','target_logprob_max','target_logprob_gt01_percent','argmax_changed_percent']:
                    row[condition+'_'+key]=tail[key]
            rows.append(row)
            curves += [{'variant':variant,'seed':seed,**json.loads(line)} for line in (path/'curves.jsonl').read_text().splitlines()]
            generations += [{'variant':variant,'seed':seed,**g} for g in e['generation']]
    write_json(group/'missing_failed_unfinished.json',{'training':checked['problems'],'postprocessing':missing,'time':timestamp()})
    assert not missing
    metrics=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k!='seed']
    agg=[{'variant':v,'n':len(cfg['seeds']),**aggregate([r for r in rows if r['variant']==v],metrics)} for v in cfg['gates']]
    by={(r['variant'],r['seed']):r for r in rows};pairs=[];pair_summary=[]
    comparisons=[(v,'mlp_half') for v in cfg['gates'] if v!='mlp_half']+[('amp_margin','amp_half'),('born_lite_margin','born_lite'),('born_lite','amp_half')]
    for candidate,baseline in comparisons:
        subset=[]
        for seed in cfg['seeds']:
            a,b=by[candidate,seed],by[baseline,seed]
            z={'variant':candidate,'baseline':baseline,'seed':seed,'delta_ce':a['test_ce']-b['test_ce'],
               'delta_ppl':a['test_ppl']-b['test_ppl'],'ppl_change_percent':100*(a['test_ppl']/b['test_ppl']-1),
               'router_fraction':a['router_parameters']/b['router_parameters'],
               'total_parameter_fraction':a['total_parameters']/b['total_parameters'],
               'train_time_ratio':a['serial_step_seconds']/b['serial_step_seconds'],
               'inference_time_ratio':a['inference_b16_l128_ms']/b['inference_b16_l128_ms'],
               'bf16_flip_delta_pp':a['flip_input_bf16_roundtrip_percent']-b['flip_input_bf16_roundtrip_percent'],
               'bf16_flip_ratio':a['flip_input_bf16_roundtrip_percent']/b['flip_input_bf16_roundtrip_percent'] if b['flip_input_bf16_roundtrip_percent'] else None}
            subset.append(z);pairs.append(z)
        summary={'variant':candidate,'baseline':baseline,**aggregate(subset,[k for k in subset[0] if k not in ['variant','baseline','seed']])}
        half=4.3026527299*summary['delta_ce_std']/math.sqrt(3)
        summary.update(delta_ce_ci95_low=summary['delta_ce_mean']-half,delta_ce_ci95_high=summary['delta_ce_mean']+half)
        pair_summary.append(summary)
    for name,values in [('runs',rows),('aggregate',agg),('paired_deltas',pairs),('paired_summary',pair_summary),
                        ('curves',curves),('routing',routing),('stability',stability),('error_tails',tails)]:csv_write(root/f'{name}.csv',values)
    write_json(root/'raw_summary.json',{'config':cfg,'runs':rows,'aggregate':agg,'paired':pairs,'paired_summary':pair_summary,'stability':stability,'error_tails':tails,'audit':checked})
    write_json(root/'generation_samples.json',generations)
    figures(root,cfg,agg,curves)
    report(root,group,cfg,agg,pair_summary)
    write_json(root/'DELIVERY_CHECK.json',{'expected':len(cfg['gates'])*len(cfg['seeds']),'training':len(rows),'reevaluations':len(rows),
        'serial_benchmarks':len(rows),'stability_probes':len(rows),'figures':[f.name for f in (root/'figures').glob('*.png')],
        'review_pending':True,'time':timestamp()})


def figures(root,cfg,agg,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'savefig.dpi':160})
    for axis,file in [('step','learning_steps'),('training_seconds','learning_time')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for i,v in enumerate(cfg['gates']):
            rec=[r for r in curves if r['variant']==v and r['kind']=='validation']
            steps=sorted({r['step'] for r in rec})
            x=[st.mean(r[axis] for r in rec if r['step']==s) for s in steps]
            y=[st.mean(r['ce'] for r in rec if r['step']==s) for s in steps]
            sd=[st.stdev(r['ce'] for r in rec if r['step']==s) for s in steps]
            for ax,start in zip(axes,[0,len(steps)//2]):
                ax.plot(x[start:],y[start:],color=COLORS[i],label=LABELS[v],marker='o',markersize=3)
                ax.fill_between(x[start:],np.array(y[start:])-sd[start:],np.array(y[start:])+sd[start:],color=COLORS[i],alpha=.12)
        for ax in axes:ax.set(xlabel=axis,ylabel='Validation CE (nats / BPE token)');ax.grid(alpha=.2)
        axes[0].set_title('Full trajectory');axes[1].set_title('Late training; mean +/- seed SD')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3)
        fig.tight_layout(rect=(0,.14,1,1));fig.savefig(out/f'{file}.png');plt.close(fig)
    names=[LABELS[r['variant']] for r in agg]
    for specs,file,title in [
        ([('test_ppl','Test BPE PPL'),('flip_input_bf16_roundtrip_percent','Router input BF16 roundtrip: flip %')],'quality_stability','Quality and numerical sensitivity; lower is better'),
        ([('serial_tokens_per_second','Training tokens / second'),('inference_b16_l128_ms','Batch16 x 128 forward (ms)')],'efficiency','Serial timing; training includes margin, inference removes it'),
        ([('gap_p05_batch_mean','Standardized gap: mean of batch p05'),('gap_below_threshold_percent','Decisions below threshold (%)')],'boundary_margin','Boundary gap measured on fixed validation windows'),
        ([('load_cv','Hard expert load CV'),('unused_experts','Unused experts (module mean)')],'expert_load','Test routing; context windows overlap')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,(key,label) in zip(axes,specs):
            ax.bar(range(len(agg)),[r[key+'_mean'] for r in agg],yerr=[r[key+'_std'] for r in agg],color=COLORS,capsize=3)
            ax.set_xticks(range(len(agg)),names,rotation=30,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        fig.suptitle(title);fig.tight_layout();fig.savefig(out/f'{file}.png');plt.close(fig)


def report(root,group,cfg,agg,pairs):
    lines=['# 路由稳定性：边界训练与共享 Born 实验','',
           '**自动阶段完成；实测解释和图表复核待完成。**','',f'正式组 `{group.relative_to(root)}`；18个新增训练，六组×三种子。',
           'WikiText-2 raw，训练集BPE4096，2层hidden256、FFN1024、context128、batch32、4000步。TITAN V FP32；每run16,384,000训练token。所有任务/非router初始化/采样顺序相同，有无margin配对的完整初始权重hash相同。',
           'validation每500步选最低CE；全部训练后重载best检查完整validation与test。test此前旧实验已观察过，本轮不再据其调参，仍需要新语料确认。',
           f'边界训练参数：tau={cfg["margin"]["threshold"]}，lambda最大值={cfg["margin"]["weight"]}；第{int(cfg["margin"]["start_fraction"]*cfg["train"]["steps"])}步后线性启用，第{int(cfg["margin"]["full_fraction"]*cfg["train"]["steps"])}步达到最大值。全部方法使用同一系数和启用规则。',
           'PPL是BPE token计量，不能与word-level PPL混用。均值±样本标准差n=3，配对区间未校正多重比较。','',
           '## 效果与实际参数','', '| Gate | Test CE | Test PPL | Router参数 | 总参数 | 最佳步数 |','|---|---:|---:|---:|---:|---:|']
    for r in agg:
        lines.append(f'| {LABELS[r["variant"]]} | {r["test_ce_mean"]:.5f} ± {r["test_ce_std"]:.5f} | {r["test_ppl_mean"]:.2f} ± {r["test_ppl_std"]:.2f} | {r["router_parameters_mean"]:.0f} | {r["total_parameters_mean"]:.0f} | {r["best_step_mean"]:.0f} |')
    lines+=['','## 配对变化','','| 方法 | 对照 | ΔCE ± SD | PPL变化% | BF16切换率比 | 推理延迟比 |','|---|---|---:|---:|---:|---:|']
    for r in pairs:
        ratio='未定义（对照零事件）' if r['bf16_flip_ratio_mean'] is None else f'{r["bf16_flip_ratio_mean"]:.3f}'
        lines.append(f'| {LABELS[r["variant"]]} | {LABELS[r["baseline"]]} | {r["delta_ce_mean"]:+.5f} ± {r["delta_ce_std"]:.5f} | {r["ppl_change_percent_mean"]:+.3f} | {ratio} | {r["inference_time_ratio_mean"]:.3f} |')
    lines+=['','## 稳定性与效率','','| Gate | BF16输入切换% | FP16输入切换% | batch16→1切换% | 训练token/s | 推理ms | 显存MiB | 负载CV |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        lines.append(f'| {LABELS[r["variant"]]} | {r["flip_input_bf16_roundtrip_percent_mean"]:.4f} | {r["flip_input_fp16_roundtrip_percent_mean"]:.4f} | {r["flip_batch16_vs1_percent_mean"]:.5f} | {r["serial_tokens_per_second_mean"]:.0f} | {r["inference_b16_l128_ms_mean"]:.3f} | {r["peak_allocated_mb_mean"]:.1f} | {r["load_cv_mean"]:.3f} |')
    lines+=['','稳定性取128个固定validation窗口，独立router输入舍入和全模型传播分别保留。BF16/FP16为输入roundtrip后FP32计算，不是TITAN V原生低精度、跨引擎或RL对齐。分位数汇总为各batch分位数均值；不能当作全体token的精确分位数。',
            '推理为GPU驻留输入batch16×128，10次预热+25次测量取中位数，包含整模型前向。共享U在预热缓存；另有batch1×128与batch1×1计时，后者没有KV cache。训练计时含实际margin分支；三种子是独立训练重复，token相关性不能用于扩大显著性。',
            '','## 实测解释','', '<!-- REVIEW_REQUIRED -->','',
            '须检查工程目标：相对half MLP PPL劣化≤0.5%、完整前向延迟增加≤5%、输入BF16舍入集合切换率至少减半；同时检查相对各自baseline、负载退化、错误尾部及各seed。目标不等于承诺，未显著变差不是非劣性证明。',
            '','## 方法限制','',
            '边界损失训练时启用，推理hook移除；标准化分母可导。共享Born保留有界联合幅度编码，删除输入相关角度，改为输入无关RY/CRY链式正交U，并取消原可学习gain、直接概率读出。这不是原Born的无损替换。所有hard top-k都有边界，不保证任意扰动零切换。',
            '本轮没有无U、无CRY、经典正交平方读出的正式消融。即使性能更好，也不能归因于量子或纠缠机制；应称结构化路由的工程折中。单语料、小模型、3seed和本机FP32不能外推大型分布式MoE。',
            '强正则的第一组pilot虽梯度有限，却造成固定少数专家的退化，已拒绝并保留在pilot/5b7d1b6a5189。后续候选由训练侧梯度及pilot validation选择，不混合不同配置结果。稳定性收益必须伴随质量与专家利用率检查。',
            '','## 图表','']
    for f in sorted((root/'figures').glob('*.png')):lines += [f'![{f.stem}](figures/{f.name})','']
    lines+=['## 原始结果','',
            '`runs.csv/aggregate.csv/paired_deltas.csv/paired_summary.csv/curves.csv/routing.csv/stability.csv/error_tails.csv/raw_summary.json`；全部固定提示生成保留于`generation_samples.json`，不筛选流畅样例。',
            f'逐run checkpoint、日志、评估、稳定性原始记录在`{group.relative_to(root)}/runs/`；完整性与缺失清单在正式组`audit.json/missing_failed_unfinished.json`。',
            '`CALIBRATION.json`仅训练侧校准，`experiments/routing_stability/README.md`为预先冻结的方法说明。','']
    path=root/'REPORT_zh.md'
    if path.exists() and '<!-- REVIEW_REQUIRED -->' not in path.read_text(encoding='utf-8'):
        raise RuntimeError('refusing to overwrite reviewed report')
    path.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
