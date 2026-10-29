"""Summarize observed learning curves without altering any model or result."""
import csv
import json
import statistics as st
from pathlib import Path

root = Path('outputs/gate_benchmark')
group = root / '45c21049b433'
rows = []
for path in sorted(group.glob('*/result.json')):
    result = json.loads(path.read_text())
    curve = [json.loads(line) for line in (path.parent/'curves.jsonl').read_text().splitlines()]
    train = [p for p in curve if 'train_ce' in p]
    dev = [p for p in curve if 'dev' in p]
    rows.append({**{k:result[k] for k in ('stage','task','gate','seed','best_step')},
        'last5_logged_batch_ce_mean': st.mean(p['train_ce'] for p in train[-5:]),
        'best_dev_score': result['best_dev']['score'],
        'last_dev_score': dev[-1]['dev']['score'], 'final_score': result['final']['score']})
assert len(rows) == 54
aggregates = []
for task, gate in sorted({(r['task'],r['gate']) for r in rows}):
    selected = [r for r in rows if r['task']==task and r['gate']==gate]
    assert len(selected) == 3
    row = {'task':task, 'gate':gate, 'n':3}
    for metric in ('last5_logged_batch_ce_mean','best_dev_score','last_dev_score','final_score'):
        row[metric+'_mean'] = st.mean(r[metric] for r in selected)
        row[metric+'_std'] = st.stdev(r[metric] for r in selected)
    aggregates.append(row)
out = {'runs':rows, 'aggregates':aggregates,
    'note':'CE is the average of the last five logged training batches per seed, then mean/sample SD over seeds; not a full-training-set evaluation.'}
(root/'learning_diagnostics.json').write_text(json.dumps(out,indent=2,allow_nan=False))
with (root/'learning_diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f,list(aggregates[0]))
    writer.writeheader()
    writer.writerows(aggregates)
print(json.dumps([r for r in aggregates if r['task'] in ('cifar10','sst2')],indent=2))
