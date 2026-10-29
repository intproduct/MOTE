"""Reproducible tables/figures for the default formal matrix only."""
import collections
import csv
import json
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fitmotn.experiments.gate_benchmark.config import load_config, digest

ROOT = Path('outputs/gate_benchmark')
CFG = load_config('experiments/gate_benchmark/configs/titan_v.json')
EXP = ROOT / digest(CFG)[:12]
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
GATES = CFG['gates']
COLORS = {'mlp': '#2878B5', 'compact_born': '#D45039', 'mps': '#24966B'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 180, 'axes.grid': True, 'grid.alpha': .18})


def stats(values):
    return {'n': len(values), 'mean': st.mean(values), 'std': st.stdev(values) if len(values) > 1 else None}


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader()
        w.writerows(rows)


def main():
    records, routes, curves_flat = [], [], []
    groups, series = collections.defaultdict(list), collections.defaultdict(list)
    for p in sorted(EXP.glob('*/result.json')):
        r = json.loads(p.read_text())
        manifest = json.loads((p.parent / 'manifest.json').read_text())
        initial = json.loads((p.parent / 'initial_dev.json').read_text())
        row = {'experiment': EXP.name, 'run_dir': str(p.parent),
               'execution_mode': 'up_to_6_concurrent_GPU_processes',
               **{k: r[k] for k in ('stage', 'task', 'gate', 'seed', 'best_step', 'training_seconds', 'wall_seconds',
                                   'examples_per_second', 'router_parameters', 'total_parameters', 'amp_skipped_steps')},
               **{f'final_{k}': v for k, v in r['final'].items()},
               'peak_cuda_mib': r['peak_cuda_bytes'] / 2**20,
               'shared_state_sha256': r['shared_state_sha256'], 'data_sha256': r['data_sha256'],
               'final_source_split': manifest['data']['final_source_split'],
               'planned_steps': CFG[r['stage']]['steps'], 'best_dev_score': r['best_dev']['score']}
        for metric in ('H_tok', 'H_dataset', 'load_cv', 'unused_experts'):
            row['route_mean_' + metric] = st.mean(v[metric] for v in r['final_routing'].values())
            row['route_initial_mean_' + metric] = st.mean(v[metric] for v in initial['routing'].values())
        for layer, route in r['final_routing'].items():
            routes.append({'task': r['task'], 'gate': r['gate'], 'seed': r['seed'], 'layer': layer,
                           **{k: v for k, v in route.items() if not isinstance(v, list)},
                           'load_json': json.dumps(route['load']), 'importance_json': json.dumps(route['importance'])})
        records.append(row)
        groups[(r['stage'], r['task'], r['gate'])].append(row)
        log = [json.loads(x) for x in (p.parent / 'curves.jsonl').read_text().splitlines()]
        train = {x['step']: x for x in log if 'train_ce' in x}
        dev = [x for x in log if 'dev' in x]
        initial_point = {'step': 0, 'score': initial['metrics']['score'], 'training_seconds': 0.0}
        sd = {'seed': r['seed'], 'train': list(train.values()), 'dev': [initial_point]}
        curves_flat.append({'stage': r['stage'], 'task': r['task'], 'gate': r['gate'], 'seed': r['seed'], **initial_point})
        for x in dev:
            point = {'step': x['step'], 'score': x['dev']['score'], 'training_seconds': train[x['step']]['training_seconds']}
            sd['dev'].append(point)
            curves_flat.append({'stage': r['stage'], 'task': r['task'], 'gate': r['gate'], 'seed': r['seed'], **point})
        series[(r['stage'], r['task'], r['gate'])].append(sd)
    save_csv(ROOT / 'formal_runs_extended.csv', records)
    save_csv(ROOT / 'formal_routing_per_layer.csv', routes)
    save_csv(ROOT / 'formal_dev_curves.csv', curves_flat)
    aggregates, deltas = [], []
    lookup = {(r['task'], r['gate'], r['seed']): r for r in records}
    measurements = ['final_accuracy', 'final_score', 'final_f1', 'final_ce', 'training_seconds', 'wall_seconds',
                    'examples_per_second', 'peak_cuda_mib', 'best_step', 'router_parameters', 'total_parameters',
                    'route_mean_H_tok', 'route_mean_H_dataset', 'route_mean_load_cv', 'route_mean_unused_experts']
    for (stage, task, gate), rows in sorted(groups.items()):
        a = {'stage': stage, 'task': task, 'gate': gate, 'n': len(rows)}
        for metric in measurements:
            vals = [r[metric] for r in rows if metric in r]
            if vals:
                ss = stats(vals)
                a[metric + '_mean'], a[metric + '_std'] = ss['mean'], ss['std']
        paired = []
        for r in sorted(rows, key=lambda x: x['seed']):
            base = lookup.get((task, 'mlp', r['seed']))
            if base is None:
                continue
            assert r['shared_state_sha256'] == base['shared_state_sha256'] and r['data_sha256'] == base['data_sha256']
            d = {'stage': stage, 'task': task, 'gate': gate, 'seed': r['seed']}
            for metric in ('final_accuracy', 'final_score', 'final_f1'):
                if metric in r:
                    d[metric + '_delta_pp'] = 100 * (r[metric] - base[metric])
            d['throughput_ratio_vs_mlp'] = r['examples_per_second'] / base['examples_per_second']
            d['training_time_ratio_vs_mlp'] = r['training_seconds'] / base['training_seconds']
            deltas.append(d)
            paired.append(d['final_score_delta_pp'])
        if paired:
            ps = stats(paired)
            a['paired_score_delta_pp_mean'], a['paired_score_delta_pp_std'] = ps['mean'], ps['std']
            a['paired_n'] = len(paired)
        aggregates.append(a)
    save_csv(ROOT / 'formal_aggregate.csv', aggregates)
    save_csv(ROOT / 'formal_paired_deltas.csv', deltas)
    summary = {'experiment': EXP.name, 'completed': len(records), 'expected': 54, 'aggregates': aggregates, 'paired_deltas': deltas}
    (ROOT / 'formal_analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    tasks = CFG['mlp']['tasks'] + CFG['bert']['tasks']
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, task in zip(axs.flat, tasks):
        ax.axvline(0, color='#777777', lw=1, ls='--')
        for j, gate in enumerate(('compact_born', 'mps')):
            values = [r['final_score_delta_pp'] for r in sorted(deltas, key=lambda r:r['seed'])
                      if r['task'] == task and r['gate'] == gate]
            if not values:
                continue
            ax.scatter(values, j+np.linspace(-.10,.10,len(values)), color=COLORS[gate], alpha=.55, s=25)
            ax.errorbar(st.mean(values), j, xerr=st.stdev(values) if len(values)>1 else 0,
                        color=COLORS[gate], marker='D', capsize=5, markersize=6, lw=2)
        ax.set_yticks([0,1], ['compact_born','mps'])
        ax.set(title=task.upper(), xlabel='Paired score difference vs MLP (pp)', ylim=(-.45,1.45))
    fig.suptitle('Same-seed differences: points = 3 seeds; diamonds = mean; bars = sample SD\nMRPC uses (accuracy + F1) / 2; all other tasks use accuracy')
    fig.tight_layout()
    fig.savefig(FIG / 'paired_score_deltas.png')
    plt.close(fig)
    for stage in ('mlp', 'bert'):
        tasks = CFG[stage]['tasks']
        for axis in ('step', 'training_seconds'):
            fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
            for ax, task in zip(axs, tasks):
                for gate in GATES:
                    ss = series.get((stage, task, gate), [])
                    if not ss:
                        continue
                    for s in ss:
                        ax.plot([p[axis] for p in s['dev']], [100*p['score'] for p in s['dev']], color=COLORS[gate], alpha=.22, lw=.9)
                    x = np.mean([[p[axis] for p in s['dev']] for s in ss], axis=0)
                    y = np.mean([[100*p['score'] for p in s['dev']] for s in ss], axis=0)
                    ax.plot(x, y, color=COLORS[gate], label=gate, lw=2, marker='o', ms=3)
                ax.set(title=task.upper(), xlabel='Optimization step' if axis == 'step' else 'Training seconds (concurrent execution)', ylabel='Internal dev score (%)')
            axs[-1].legend(fontsize=8)
            fig.suptitle(f'{stage.upper()}: dev-selected protocol; faint lines = individual seeds' + ('; timing includes contention' if axis != 'step' else ''))
            fig.tight_layout()
            fig.savefig(FIG / f'{stage}_dev_vs_{axis}.png')
            plt.close(fig)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
        for ax, task in zip(axs, tasks):
            for gate in GATES:
                ss = series.get((stage, task, gate), [])
                for s in ss:
                    ax.plot([p['step'] for p in s['train']], [p['train_ce'] for p in s['train']], color=COLORS[gate], alpha=.22, lw=.9)
                if ss:
                    x = [p['step'] for p in ss[0]['train']]
                    y = np.mean([[p['train_ce'] for p in s['train']] for s in ss], axis=0)
                    ax.plot(x, y, color=COLORS[gate], lw=2, label=gate)
            ax.set(title=task.upper(), xlabel='Optimization step', ylabel='Training CE (logged batch)')
        axs[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f'{stage}_train_ce.png')
        plt.close(fig)
    tasks = CFG['mlp']['tasks'] + CFG['bert']['tasks']
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, task in zip(axs, CFG['mlp']['tasks']):
        for gate in GATES:
            ss = series.get(('mlp', task, gate), [])
            points = [[p for p in s['dev'] if p['step'] > 0] for s in ss]
            for pp in points:
                ax.plot([p['step'] for p in pp], [100*p['score'] for p in pp], color=COLORS[gate], alpha=.22, lw=.9)
            if points:
                ax.plot([p['step'] for p in points[0]], np.mean([[100*p['score'] for p in pp] for pp in points], axis=0),
                        color=COLORS[gate], label=gate, lw=2, marker='o', ms=3)
        ax.set(title=task.upper(), xlabel='Optimization step', ylabel='Internal dev accuracy (%)')
    axs[-1].legend(fontsize=8)
    fig.suptitle('MLP: trained checkpoints only (step 0 omitted to show learning differences)\nFaint lines = individual seeds; bold lines = mean')
    fig.tight_layout()
    fig.savefig(FIG / 'mlp_dev_vs_step_trained.png')
    plt.close(fig)
    for metrics, filename in [(['final_score', 'examples_per_second', 'peak_cuda_mib'], 'outcome_efficiency_memory'),
                              (['route_mean_H_tok', 'route_mean_load_cv', 'route_mean_unused_experts'], 'routing_diagnostics')]:
        fig, axs = plt.subplots(3, 1, figsize=(11, 10))
        for ax, metric in zip(axs, metrics):
            for j, gate in enumerate(GATES):
                a = [next((v for v in aggregates if v['task'] == t and v['gate'] == gate), None) for t in tasks]
                scale = 100 if metric == 'final_score' else 1
                means = [v[metric+'_mean']*scale if v else np.nan for v in a]
                stds = [(v[metric+'_std'] or 0)*scale if v else 0 for v in a]
                ax.bar(np.arange(6)+(j-1)*.24, means, width=.23, color=COLORS[gate], label=gate, yerr=stds, capsize=3)
            ax.set_xticks(range(6), tasks)
            ax.set_ylabel(metric + (' (%)' if metric == 'final_score' else ''))
        axs[0].legend(ncol=3)
        fig.suptitle('Mean and sample SD over seeds; MRPC score = (accuracy + F1) / 2' + ('\nThroughput here includes concurrent GPU contention; see isolated efficiency figure' if filename == 'outcome_efficiency_memory' else ''))
        fig.tight_layout()
        fig.savefig(FIG / (filename + '.png'))
        plt.close(fig)
    print(json.dumps({'experiment': EXP.name, 'completed': len(records), 'figures': [p.name for p in FIG.glob('*.png')]}, indent=2))


if __name__ == '__main__':
    main()
