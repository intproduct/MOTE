"""Plot a completed audit without rewriting training artifacts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('review',type=Path);a=p.parse_args()
    rows=list(csv.DictReader((a.review/'runs.csv').open(encoding='utf-8')))
    pairs=list(csv.DictReader((a.review/'paired.csv').open(encoding='utf-8')))
    curves=json.loads((a.review/'curves.json').read_text())
    out=a.review/'figures';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    names={'mlp':'MLP','spectral_functional':'Classical functional','spectral_random':'Classical random',
           'shared_born':'Shared Born','angle_natural':'Angle: natural','angle_random':'Angle: random',
           'angle_functional':'Angle: functional','product_functional':'Product: functional','exp_adapter':'Expectation + adapter'}
    methods=list(names);colors=plt.colormaps['tab10'](np.arange(len(methods)))
    fig,ax=plt.subplots(figsize=(9.8,5.3),layout='constrained')
    for j,m in enumerate(methods):
        r=[x for x in rows if x['method']==m];vals=[float(x['delta_ce_vs_original']) for x in r]
        ax.scatter(vals,np.full(3,j),color=colors[j],s=34,zorder=3)
        ax.plot([np.mean(vals)-np.std(vals,ddof=1),np.mean(vals)+np.std(vals,ddof=1)],[j,j],color=colors[j],lw=2)
        ax.scatter([np.mean(vals)],[j],marker='|',s=180,color='black',zorder=4)
    ax.set_yticks(range(len(methods)),[names[m] for m in methods]);ax.invert_yaxis();ax.axvline(0,color='.6',lw=1)
    ax.set_xlabel('Sampled validation CE difference from original router (nat/token)')
    ax.set_title('Fixed final step: 3 seeds; points = seeds, bars = mean +/- seed SD')
    ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'validation_ce.png',dpi=190);fig.savefig(out/'validation_ce.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True,layout='constrained')
    chosen=['mlp','spectral_functional','angle_random','angle_functional','product_functional']
    for ax,seed in zip(axes,[17,29,43]):
        base=float(next(x['baseline_ce'] for x in rows if int(x['seed'])==seed))
        for m in chosen:
            seq=next(c['values'] for c in curves if c['method']==m and c['seed']==seed)
            ax.plot([v['step'] for v in seq if v['step']>0],[v['ce']-base for v in seq if v['step']>0],
                    label=names[m],color=colors[methods.index(m)],marker='.',ms=5)
        ax.axhline(0,color='.6',lw=.8);ax.axvline(400,color='.4',ls='--',lw=1)
        ax.set_title(f'Seed {seed}');ax.set_xlabel('Training step');ax.grid(alpha=.2)
    axes[0].set_ylabel('Sampled CE difference from original router')
    axes[1].legend(loc='upper center',fontsize=9)
    fig.suptitle('Steps 1-400: imitation; steps 401-800: CE + teacher KL')
    fig.savefig(out/'curves.png',dpi=190);fig.savefig(out/'curves.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.8,5.7),layout='constrained')
    labels=[]
    for j,r in enumerate(pairs):
        c=float(r['delta_ce']);low=float(r['window_bootstrap_ci95_low']);high=float(r['window_bootstrap_ci95_high'])
        color=colors[[17,29,43].index(int(r['seed']))]
        ax.plot([low,high],[j,j],color=color,lw=1.6);ax.scatter([c],[j],color=color,s=20)
        labels.append(f"s{r['seed']}  {names[r['candidate']]} - {names[r['reference']]}")
    ax.set_yticks(range(len(labels)),labels,fontsize=9);ax.invert_yaxis();ax.axvline(0,color='.4',ls='--')
    ax.set_xlabel('Paired CE difference (nat/token); negative favors candidate')
    ax.set_title('Within-checkpoint 95% window bootstrap intervals (128 windows per seed)')
    ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'paired_ce.png',dpi=190);fig.savefig(out/'paired_ce.pdf');plt.close(fig)
    print(out)


if __name__=='__main__':main()
