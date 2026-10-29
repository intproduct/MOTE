import json
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('outputs/gate_benchmark')
s = json.loads((root / 'serial_efficiency_summary.json').read_text())
tasks = ['mnist','fashion_mnist','cifar10','sst2','mrpc','rte']
gates = ['mlp','compact_born','mps']
colors = {'mlp':'#2878B5','compact_born':'#D45039','mps':'#24966B'}
lookup = {(r['task'],r['gate'],r['seed']):r for r in s['runs']}
groups = {(r['task'],r['gate']):r for r in s['groups']}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
fig, axs = plt.subplots(3,1,figsize=(11,10))
for j,g in enumerate(gates):
    x=np.arange(6)+(j-1)*.24
    axs[0].bar(x,[groups[t,g]['examples_per_second_mean'] for t in tasks],width=.23,color=colors[g],label=g,
               yerr=[groups[t,g]['examples_per_second_std'] for t in tasks],capsize=3)
    ratios=[[lookup[t,g,seed]['examples_per_second']/lookup[t,'mlp',seed]['examples_per_second'] for seed in (17,29,43)] for t in tasks]
    axs[1].bar(x,[st.mean(r) for r in ratios],width=.23,color=colors[g],yerr=[st.stdev(r) for r in ratios],capsize=3)
    axs[2].bar(x,[groups[t,g]['peak_cuda_mib_mean'] for t in tasks],width=.23,color=colors[g],
               yerr=[groups[t,g]['peak_cuda_mib_std'] for t in tasks],capsize=3)
for ax,label in zip(axs,['Isolated examples / second','Paired throughput ratio vs MLP','Peak allocated GPU memory (MiB)']):
    ax.set_xticks(range(6),tasks)
    ax.set_ylabel(label)
axs[0].legend(ncol=3)
axs[1].axhline(1,color='#666666',lw=.8,ls='--')
fig.suptitle('One benchmark process: best-checkpoint copies, 3 warmup + 10 measured updates\nMean and sample SD across 3 seeds; desktop GPU processes remain running')
fig.tight_layout()
fig.savefig(root/'figures/isolated_efficiency.png',dpi=180)
plt.close(fig)
