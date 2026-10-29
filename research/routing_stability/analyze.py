"""Summarize the independent routing probe, without touching old experiments."""
from pathlib import Path
import csv
import json
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]/'outputs/routing_stability/20261028'
raw = json.loads((ROOT/'native_topk/RESULTS.json').read_text())
variants = list(dict.fromkeys(x['variant'] for x in raw['rows']))
labels = dict(zip(variants, ['MLP full','MLP half','MPO full','MPO half','Born full','Born half']))
conditions = list(dict.fromkeys(x['condition'] for x in raw['rows']))
aggregate = []
for variant in variants:
    for condition in conditions:
        rows = [x for x in raw['rows'] if x['variant']==variant and x['condition']==condition]
        per_seed = [sum(x['changed'] for x in rows if x['seed']==s)/sum(x['n'] for x in rows if x['seed']==s) for s in [17,29,43]]
        aggregate.append({'variant':variant, 'condition':condition,
            'changed':sum(x['changed'] for x in rows), 'n':sum(x['n'] for x in rows),
            'flip_pct_mean':100*st.mean(per_seed), 'flip_pct_seed_sd':100*st.stdev(per_seed),
            'max_centered_score_difference':max(x['max_centered_score_difference'] for x in rows)})
with (ROOT/'aggregate.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
(ROOT/'aggregate.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
lookup={(x['variant'],x['condition']):x for x in aggregate}
selected=['model_batch16_vs_batch1','input_roundtrip_fp16','input_roundtrip_bf16','input_uniform_rms_0.001','input_uniform_rms_0.01']
table=['| Gate | FP32 batch16→1 切换数 | 输入FP16舍入 % | 输入BF16舍入 % | 0.1% RMS微扰 % | 1% RMS微扰 % |',
       '|---|---:|---:|---:|---:|---:|']
for v in variants:
    batch=lookup[v,selected[0]]
    table.append(f"| {labels[v]} | {batch['changed']}/{batch['n']:,} | "+' | '.join(f"{lookup[v,c]['flip_pct_mean']:.3f}" for c in selected[1:])+' |')
(ROOT/'TABLE.md').write_text('\n'.join(table),encoding='utf-8')
colors=['#1f5d9b','#4ba0d5','#bd5b13','#e59c3f','#6c3fa0','#bb6fbc']
fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
x=np.arange(len(variants)); width=.34
for shift,c,label,col in [(-width/2,'input_roundtrip_fp16','FP16 input rounding','#629aca'),(width/2,'input_roundtrip_bf16','BF16 input rounding','#6c3fa0')]:
    values=[lookup[v,c]['flip_pct_mean'] for v in variants]
    errors=[lookup[v,c]['flip_pct_seed_sd'] for v in variants]
    axes[0].bar(x+shift,values,width,color=col,label=label,yerr=errors,capsize=3)
axes[0].set_xticks(x,labels.values(),rotation=30,ha='right')
axes[0].set_ylabel('Tokens with changed top-4 expert set (%)')
axes[0].set_title('Input rounding only; router stays FP32')
axes[0].legend(frameon=False,fontsize=9)
levels=[1e-6,1e-4,1e-3,1e-2]
for v,col in zip(variants,colors):
    values=[lookup[v,f'input_uniform_rms_{a:g}']['flip_pct_mean'] for a in levels]
    axes[1].plot(levels,values,marker='o',label=labels[v],color=col)
axes[1].set_xscale('log');axes[1].set_xlabel('Uniform input perturbation bound / per-token input RMS')
axes[1].set_ylabel('Tokens with changed top-4 expert set (%)')
axes[1].set_title('Fixed random direction; diagnostic, not adversarial attack')
axes[1].legend(frameon=False,ncol=2,fontsize=9)
for ax in axes:
    ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
fig.suptitle('18 frozen checkpoints | validation only | 24,576 token-module decisions per gate\nThree-seed means; bars show seed SD. No native BF16 kernels or new training.',fontsize=11)
fig.savefig(ROOT/'routing_sensitivity.png',dpi=170)
plt.close(fig)
prelim=json.loads((ROOT/'RESULTS.json').read_text())
old={(x['variant'],x['seed'],x['module'],x['condition']):x['changed'] for x in prelim['rows']}
diff=[{k:x[k] for k in ['variant','seed','module','condition','changed']} | {'preliminary_changed':old[x['variant'],x['seed'],x['module'],x['condition']]}
      for x in raw['rows'] if x['changed']!=old[x['variant'],x['seed'],x['module'],x['condition']]]
(ROOT/'PROBE_REVIEW.json').write_text(json.dumps({'final_source':'native_topk/RESULTS.json',
    'correction':'Preserve native dtype for top-k selection, then convert scores to FP64 for error arithmetic only.',
    'preliminary_preserved':True,'changed_counts':diff,'checkpoint_count':len(raw['checkpoints']),
    'comparisons':len(raw['rows']),'source_and_checkpoint_hashes_unchanged':True},indent=2),encoding='utf-8')
print('\n'.join(table));print('Native-dtype vs preliminary count differences:',len(diff))
