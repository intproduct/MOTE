from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from .runner import read_json, canonical_hash, audit, timestamp
from .data import write_json


LABELS = {'mlp': 'MLP full', 'mlp_half': 'MLP half', 'mpo_c2_full': 'MPO full (chi=2)',
          'mpo_c1_half': 'MPO half (chi=1)', 'amp_d1_full': 'Born full', 'amp_d1_half': 'Born half'}
COLORS = {'mlp':'#2563eb','mlp_half':'#60a5fa','mpo_c2_full':'#15803d',
          'mpo_c1_half':'#4ade80','amp_d1_full':'#b45309','amp_d1_half':'#fbbf24'}


def csv_write(path, rows):
    if not rows:
        raise ValueError(f'no rows for {path}')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys):
    result = {}
    for key in keys:
        values = [r[key] for r in rows]
        result[key+'_mean'] = statistics.mean(values)
        result[key+'_std'] = statistics.stdev(values) if len(values) > 1 else 0.
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--root', default='outputs/autoregressive_gate')
    args = p.parse_args()
    root, cfg = Path(args.root).resolve(), read_json(args.config)
    group = root/cfg['phase']/canonical_hash(cfg)[:12]
    checked = audit(group, cfg)
    assert checked['passed'] and cfg['phase'] == 'formal'
    rows, curves, routing, generations = [], [], [], []
    missing_evaluation = []
    for seed in cfg['seeds']:
        for variant in cfg['gates']:
            path = group/'runs'/f'{variant}_s{seed}'
            result = read_json(path/'result.json')
            if not (path/'evaluation.json').exists():
                missing_evaluation.append(path.name)
                continue
            evaluated = read_json(path/'evaluation.json')
            assert evaluated['checkpoint_sha256'] == result['checkpoint_sha256']
            row = {'variant': variant, 'seed': seed,
                   'test_ce': evaluated['test']['ce'], 'test_ppl': evaluated['test']['ppl'],
                   'test_accuracy': evaluated['test']['next_token_accuracy'],
                   'dev_ce': result['best_validation']['ce'], 'dev_ppl': result['best_validation']['ppl'],
                   'initial_dev_ce': result['initial_validation']['ce'],
                   'last_dev_ce': result['last_validation']['ce'],
                   'router_parameters': result['parameters']['router_total'],
                   'total_parameters': result['parameters']['total'],
                   'best_step': result['best_step'], 'steps': result['steps'],
                   'tokens_seen': result['tokens_seen'], 'training_seconds': result['training_seconds'],
                   'wall_seconds': result['wall_seconds'],
                   'training_tokens_per_second': result['training_tokens_per_second'],
                   'peak_allocated_mb': result['peak_allocated_mb'],
                   'serial_step_seconds': evaluated['serial_benchmark']['mean_step_seconds'],
                   'serial_tokens_per_second': evaluated['serial_benchmark']['tokens_per_second'],
                   'serial_peak_allocated_mb': evaluated['serial_benchmark']['peak_allocated_mb'],
                   'checkpoint_sha256': result['checkpoint_sha256'],
                   'shared_init_hash': result['shared_init_hash']}
            route = evaluated['test']['routing']
            for key in ['entropy','load_cv','unused_experts']:
                row[key] = statistics.mean(v[key] for v in route.values())
            rows.append(row)
            for name, values in route.items():
                routing.append({'variant':variant,'seed':seed,'module':name,
                                **{k:v for k,v in values.items() if k != 'load'},
                                **{f'expert_{i}_load':v for i,v in enumerate(values['load'])}})
            for line in (path/'curves.jsonl').read_text(encoding='utf-8').splitlines():
                curves.append({'variant':variant,'seed':seed,**json.loads(line)})
            for generation in evaluated['generation']:
                generations.append({'variant':variant,'seed':seed,**generation})
    write_json(group/'missing_failed_unfinished.json', {'training': checked['problems'],
               'missing_evaluation': missing_evaluation, 'time': timestamp()})
    assert not missing_evaluation
    metrics = [key for key, value in rows[0].items() if isinstance(value, (int,float)) and key != 'seed']
    aggregated = [{'variant':variant,'n':len(cfg['seeds']),
                   **aggregate([r for r in rows if r['variant']==variant], metrics)} for variant in cfg['gates']]
    indexed = {(r['variant'], r['seed']):r for r in rows}
    pairs, pair_summary = [], []
    for baseline in ['mlp','mlp_half']:
        for variant in cfg['gates']:
            if variant == baseline:
                continue
            here = []
            for seed in cfg['seeds']:
                a, b = indexed[variant,seed], indexed[baseline,seed]
                delta = {'variant':variant,'baseline':baseline,'seed':seed,
                         'delta_ce':a['test_ce']-b['test_ce'],
                         'delta_ppl':a['test_ppl']-b['test_ppl'],
                         'relative_ppl_percent':100*(a['test_ppl']/b['test_ppl']-1),
                         'router_fraction':a['router_parameters']/b['router_parameters'],
                         'total_parameter_fraction':a['total_parameters']/b['total_parameters'],
                         'serial_time_ratio':a['serial_step_seconds']/b['serial_step_seconds']}
                pairs.append(delta)
                here.append(delta)
            summary = {'variant':variant,'baseline':baseline,
                       **aggregate(here, ['delta_ce','delta_ppl','relative_ppl_percent','router_fraction',
                                          'total_parameter_fraction','serial_time_ratio'])}
            if len(here) == 3:
                halfwidth = 4.3026527299*summary['delta_ce_std']/math.sqrt(3)
                summary['delta_ce_95ci_low'] = summary['delta_ce_mean']-halfwidth
                summary['delta_ce_95ci_high'] = summary['delta_ce_mean']+halfwidth
            pair_summary.append(summary)
    for name, values in [('runs',rows),('aggregate',aggregated),('paired_deltas',pairs),
                         ('paired_summary',pair_summary),('curves',curves),('routing',routing)]:
        csv_write(root/f'{name}.csv', values)
    write_json(root/'raw_summary.json', {'config':cfg,'runs':rows,'aggregate':aggregated,
                                        'paired':pairs,'paired_summary':pair_summary,'audit':checked})
    write_json(root/'generation_samples.json', generations)
    figures(root, cfg, aggregated, curves, pair_summary)
    write_report(root, group, cfg, aggregated, pair_summary)
    write_json(root/'DELIVERY_CHECK.json', {'training_complete':len(rows), 'checkpoint_reevaluated':len(rows),
               'serial_benchmarks':len(rows), 'expected':len(cfg['gates'])*len(cfg['seeds']),
               'figures':[p.name for p in (root/'figures').glob('*.png')],
               'human_review_pending':True, 'created':timestamp()})


def figures(root, cfg, agg, curves, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out = root/'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':120,'savefig.dpi':160,'font.size':10})
    for axis, filename, xlabel in [('step','learning_steps','Optimizer steps (same token budget)'),
                                  ('training_seconds','learning_time','Training compute seconds (excludes evaluation)')]:
        fig, ax = plt.subplots(figsize=(10,6))
        for gate in cfg['gates']:
            records = [r for r in curves if r['variant']==gate and r['kind']=='validation']
            steps = sorted(set(r['step'] for r in records))
            x = [statistics.mean(r[axis] for r in records if r['step']==step) for step in steps]
            y = [statistics.mean(r['ce'] for r in records if r['step']==step) for step in steps]
            sd = [statistics.stdev(r['ce'] for r in records if r['step']==step) for step in steps]
            ax.plot(x,y,label=LABELS[gate],color=COLORS[gate],linestyle='--' if 'half' in gate else '-')
            ax.fill_between(x,np.array(y)-sd,np.array(y)+sd,color=COLORS[gate],alpha=.12)
        ax.set(xlabel=xlabel,ylabel='Full validation cross-entropy (nats/token)',title='WikiText-2 causal LM: mean +/- seed SD')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9,ncol=2)
        fig.tight_layout()
        fig.savefig(out/f'{filename}.png')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,6))
    for r in agg:
        gate = r['variant']
        x,y=r['router_parameters_mean']/1000,r['test_ppl_mean']
        ax.errorbar(x,y,yerr=r['test_ppl_std'],fmt='o',color=COLORS[gate],capsize=4,label=LABELS[gate])
        ax.annotate(LABELS[gate],(x,y),xytext=(6,8 if 'full' in gate or gate=='mlp' else -14),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Router parameters (thousands; excludes shared backbone)',ylabel='Test BPE perplexity (lower is better)',title='Parameter efficiency: mean +/- seed SD')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'parameter_efficiency.png')
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    names=[LABELS[r['variant']] for r in agg]
    colors=[COLORS[r['variant']] for r in agg]
    for ax,key,label in [(axes[0],'serial_tokens_per_second','Training tokens / second'),
                         (axes[1],'serial_peak_allocated_mb','Peak allocated GPU memory (MiB)')]:
        ax.bar(range(len(agg)),[r[key+'_mean'] for r in agg],yerr=[r[key+'_std'] for r in agg],color=colors,capsize=3)
        ax.set_xticks(range(len(agg)),names,rotation=32,ha='right')
        ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Serial end-to-end training microbenchmark: disposable checkpoint copies')
    fig.tight_layout()
    fig.savefig(out/'serial_efficiency.png')
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax,baseline in zip(axes,['mlp','mlp_half']):
        subset=[r for r in pairs if r['baseline']==baseline]
        means=np.array([r['delta_ce_mean'] for r in subset])
        low=np.array([r['delta_ce_95ci_low'] for r in subset])
        high=np.array([r['delta_ce_95ci_high'] for r in subset])
        ax.errorbar(means,range(len(subset)),xerr=[means-low,high-means],fmt='o',capsize=4)
        ax.set_yticks(range(len(subset)),[LABELS[r['variant']] for r in subset])
        ax.axvline(0,color='black',linewidth=1)
        ax.set(xlabel='Paired test CE difference (negative is better)',title=f'Against {LABELS[baseline]}: n=3, unadjusted 95% t CI')
        ax.grid(axis='x',alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'paired_differences.png')
    plt.close(fig)


def write_report(root, group, cfg, agg, pairs):
    data = read_json(root/'data/manifest.json')
    lines = ['# MoTE 自回归语言建模实验', '',
             '**状态：训练与自动汇总完成，实测解释及图表人工复核待完成。**', '',
             f'正式组：`{group.relative_to(root)}`。六种路由器 × 三种子，共 {len(cfg["gates"])*len(cfg["seeds"])} 个真实 GPU run。',
             '全部从零训练，TITAN V，FP32；不是 BERT 分类，也不是预训练大模型指令微调。', '',
             '## 1. 协议和数据', '',
             f'- WikiText-2 raw 固定 revision `{data["revision"]}`，官方 train/validation/test。',
             f'- BPE 词表 {data["vocab_size"]}，仅在 train 拟合。train/validation/test token 数：'+
             '/'.join(str(data['splits'][s]['tokens']) for s in ['train','validation','test'])+'。',
             '- 跨 split 完全相同文章数均为 0；这不等于排除全部近重复文本。',
             '- 模型：2 层因果 Transformer，hidden 256，4 heads，FFN 1024，上下文 128，输入输出词嵌入共享。每层两个 MoTE 投影，16 专家、top-4。',
             f'- 每个 run {cfg["train"]["steps"]} 步，batch {cfg["train"]["batch_size"]}，共 {cfg["train"]["steps"]*cfg["train"]["batch_size"]*128:,} 训练 token；随机窗口有放回采样。',
             '- 三种子 17/29/43；同种子共享权重哈希、专家位置、tokenizer、采样窗口顺序一致。路由预算按实际参数量（含归一化/投影/读出）计算。',
             '- validation 最低 token CE 选模；矩阵训练完毕后统一重载 checkpoint 评估 test。测试集不选模。',
             '- PPL=exp(总 NLL / 有效目标 token 数)。评估滑窗 context128/stride64，每个目标只计一次，padding不参与loss/路由。',
             '- 此处为 4096 词表 BPE PPL，不能直接对比公开 word-level WikiText PPL。', '',
             '## 2. 最终测试集结果', '',
             '均值 ± 样本标准差（3 seeds）；CE、PPL 越低越好。', '',
             '| Gate | Test CE | Test PPL | Router参数 | 总参数 | 最佳步数均值 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in agg:
        lines.append(f'| {LABELS[r["variant"]]} | {r["test_ce_mean"]:.5f} ± {r["test_ce_std"]:.5f} | {r["test_ppl_mean"]:.2f} ± {r["test_ppl_std"]:.2f} | {r["router_parameters_mean"]:.0f} | {r["total_parameters_mean"]:.0f} | {r["best_step_mean"]:.0f} |')
    lines += ['', '## 3. 同种子配对差值', '',
              'Δ=候选−对照；负 CE/PPL 差值更好。95% t 区间仅为三种子描述性估计，未作多重比较校正。', '',
              '| Candidate | 对照 | ΔCE均值±SD | ΔCE 95%区间 | ΔPPL均值 | 路由参数比例 | 串行耗时比例 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        lines.append(f'| {LABELS[r["variant"]]} | {LABELS[r["baseline"]]} | {r["delta_ce_mean"]:+.5f} ± {r["delta_ce_std"]:.5f} | [{r["delta_ce_95ci_low"]:+.5f}, {r["delta_ce_95ci_high"]:+.5f}] | {r["delta_ppl_mean"]:+.2f} | {100*r["router_fraction_mean"]:.2f}% | {r["serial_time_ratio_mean"]:.3f}× |')
    lines += ['', '## 4. 效率与路由', '',
              '| Gate | 训练秒数 | 串行token/s | 峰值显存MiB | soft路由熵 | hard负载CV | 未用专家数 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        lines.append(f'| {LABELS[r["variant"]]} | {r["training_seconds_mean"]:.1f} | {r["serial_tokens_per_second_mean"]:.0f} | {r["serial_peak_allocated_mb_mean"]:.1f} | {r["entropy_mean"]:.4f} | {r["load_cv_mean"]:.4f} | {r["unused_experts_mean"]:.2f} |')
    lines += ['', '训练秒数不含评估和保存；wall_seconds 另见 runs.csv。串行计时为 5 步预热+20步前后向/裁剪/AdamW，使用可丢弃副本和新优化器；不是推理速度，也不直接代表达到同一 PPL 的用时。',
              '路由指标对四个模块取平均，来自 test 重评估；滑窗上下文会重复计入路由token，目标NLL不会重复。', '',
              '## 5. 实测解释（待复核填写）', '', '<!-- REVIEW_REQUIRED -->', '',
              '需要结合相同步数曲线、时间曲线、配对差值与参数比例解释能力/效率，不能仅凭最高均分选胜者。', '',
              '## 6. 图表', '']
    for name in ['learning_steps','learning_time','parameter_efficiency','serial_efficiency','paired_differences']:
        lines += [f'![{name}](figures/{name}.png)', '']
    lines += ['## 7. 生成样例和限制', '',
              '所有 checkpoint 均用相同三个固定提示生成，贪心解码最多64个新token；完整原文与token ID见 generation_samples.json，没有按流畅度筛选。样例用于检查能否自回归运行，不作为统计排名指标。',
              'MPO 是经典非线性张量网络。幅度 Born 的实数幅度经 RY/CRY 演化后平方读出；depth1 的连接不覆盖所有 qubit 分区，本轮不能识别纠缠贡献。',
              '半预算为路由器参数减半，整个模型节省比例明显更小。相同层数/宽度/训练token并不意味着各门计算量相同。',
              '仅一个小英文语料、短上下文、小模型和三个种子，不支持大模型泛化、量子优势或推理加速结论。', '',
              '## 8. 原始交付物', '',
              '- runs.csv / aggregate.csv：逐run和均值标准差。',
              '- paired_deltas.csv / paired_summary.csv：同种子配对差值。',
              '- curves.csv / routing.csv / raw_summary.json / generation_samples.json。',
              f'- `{group.relative_to(root)}/runs/`：manifest、训练曲线、路由日志、窗口索引、best/last checkpoint、重评估/串行计时。',
              f'- `{group.relative_to(root)}/audit.json` 与 missing_failed_unfinished.json：完整性核验。',
              '- 数据来源：[Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext)；计量方法：[Hugging Face PPL](https://huggingface.co/docs/transformers/perplexity)。', '']
    path = root/'REPORT_zh.md'
    if path.exists() and '<!-- REVIEW_REQUIRED -->' not in path.read_text(encoding='utf-8'):
        raise RuntimeError('reviewed report exists; refusing to overwrite')
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
