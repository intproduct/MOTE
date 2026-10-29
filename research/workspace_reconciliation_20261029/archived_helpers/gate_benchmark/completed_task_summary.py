"""Summarize a task only once all nine default runs have finished."""
import json
import statistics as st
import sys
from pathlib import Path

task = sys.argv[1]
root = Path('outputs/gate_benchmark')
group = root / '45c21049b433'
c = json.loads((group / 'config.json').read_text())
stage = next(s for s in ('mlp', 'bert') if task in c[s]['tasks'])
results = {}
for gate in c['gates']:
    for seed in c['seeds']:
        results[gate, seed] = json.loads((group / f'{stage}_{task}_{gate}_seed{seed}/result.json').read_text())
for seed in c['seeds']:
    assert len({results[g, seed]['shared_state_sha256'] for g in c['gates']}) == 1
    assert len({results[g, seed]['data_sha256'] for g in c['gates']}) == 1
rows = []
for gate in c['gates']:
    values = [100*results[gate,s]['final']['score'] for s in c['seeds']]
    deltas = [100*(results[gate,s]['final']['score']-results['mlp',s]['final']['score']) for s in c['seeds']]
    rows.append({'gate': gate, 'seeds': c['seeds'], 'scores_percent': values,
        'mean_percent': st.mean(values), 'sample_sd_percent': st.stdev(values),
        'paired_deltas_pp': deltas, 'paired_delta_mean_pp': st.mean(deltas),
        'paired_delta_sample_sd_pp': st.stdev(deltas)})
out = {'task': task, 'completed': 9, 'expected': 9,
    'metric': 'mean_of_accuracy_and_f1' if task == 'mrpc' else 'accuracy',
    'shared_initialization_and_data_hashes_match': True, 'groups': rows}
(root / f'{task}_completed_summary.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out))
