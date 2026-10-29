"""Independent CPU-only delivery checks and descriptive post-training analysis."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math
import statistics as st
import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
GROUP = ROOT / 'formal/9b3af7494afc'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def finite(v):
    if isinstance(v, dict):
        return all(finite(x) for x in v.values())
    if isinstance(v, (tuple, list)):
        return all(finite(x) for x in v)
    return not isinstance(v, float) or math.isfinite(v)

def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(name, rows):
    with (ROOT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

cfg = read(GROUP/'config.json')
assert cfg == read(REPO/'experiments/autoregressive_gate/configs/titan_v.json')
# Match the frozen runner's sorted-key JSON encoding, including default spaces.
cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
frozen = read(GROUP/'frozen_sources.json')
assert all(sha(REPO/p) == h for p,h in frozen.items())
data = read(ROOT/'data/manifest.json')
assert all(sha(ROOT/'data'/p) == h for p,h in data['files'].items())
assert data['tokenizer_training_split'] == 'train'
assert all(v == 0 for v in data['exact_article_overlap'].values())
tok = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
raw = read(ROOT/'raw_summary.json')
assert raw['config'] == cfg and raw['audit']['passed']
rows = {(r['variant'],r['seed']): r for r in raw['runs']}
expected = {(v,s) for v in cfg['gates'] for s in cfg['seeds']}
assert set(rows) == expected and len(rows) == 18
assert {p.name for p in (GROUP/'runs').iterdir() if p.is_dir()} == {f'{v}_s{s}' for v,s in expected}
checks, thresholds, generations, init, order, positions = [], [], [], {}, {}, {}
torch.set_num_threads(1)
for seed in cfg['seeds']:
    for variant in cfg['gates']:
        folder = GROUP/'runs'/f'{variant}_s{seed}'
        m,r,e = [read(folder/name) for name in ['manifest.json','result.json','evaluation.json']]
        assert r['config_hash'] == m['config_hash'] == cfg_hash
        assert m['environment']['source_sha256'] == frozen
        assert r['steps'] == r['finite_steps'] == r['nonzero_router_grad_steps'] == 4000
        assert r['tokens_seen'] == 16384000 and r['shared_parameters_updated']
        assert all(v > 0 for v in r['router_parameter_max_changes'].values())
        assert finite(r) and finite(e)
        assert sha(folder/'best.pt') == r['checkpoint_sha256'] == e['checkpoint_sha256']
        assert sha(folder/'batch_starts.npy') == r['batch_order_sha256']
        starts = np.load(folder/'batch_starts.npy')
        assert starts.shape == (4000,32) and starts.min() >= 0
        assert starts.max()+128 < data['splits']['train']['tokens']
        assert m['data_manifest_sha256'] == r['data_manifest_sha256'] == sha(ROOT/'data/manifest.json')
        init.setdefault(seed,set()).add(r['shared_init_hash'])
        order.setdefault(seed,set()).add(r['batch_order_sha256'])
        positions.setdefault(seed,set()).add(json.dumps(r['parameters']['positions'],sort_keys=True))
        curves = [json.loads(line) for line in (folder/'curves.jsonl').read_text().splitlines()]
        routes = [json.loads(line) for line in (folder/'routing.jsonl').read_text().splitlines()]
        assert finite(curves) and finite(routes)
        val = [c for c in curves if c['kind']=='validation']
        assert [c['step'] for c in val] == list(range(0,4001,500))
        assert r['best_step'] == min(val,key=lambda c:c['ce'])['step'] == 4000
        assert abs(e['validation_reload']['ce']-r['best_validation']['ce']) < 1e-6
        for split,key in [('validation','validation_reload'),('test','test')]:
            metric=e[key]
            assert metric['tokens'] == data['splits'][split]['tokens']-1
            assert math.isclose(metric['ce'],metric['nll_sum']/metric['tokens'],abs_tol=1e-12)
            assert math.isclose(metric['ppl'],math.exp(metric['ce']),rel_tol=1e-12)
        assert len(e['test']['routing']) == 4
        for mod in e['test']['routing'].values():
            assert len(mod['load']) == 16 and math.isclose(sum(mod['load']),4,abs_tol=1e-6)
            assert 0 <= mod['entropy'] <= math.log(16)+1e-6 and mod['load_cv'] >= 0
            assert mod['unused_experts'] == sum(v == 0 for v in mod['load'])
        benchmark=e['serial_benchmark']
        assert benchmark['warmup_steps'] == 5 and benchmark['measured_steps'] == 20
        assert len(benchmark['step_seconds']) == 20 and min(benchmark['step_seconds']) > 0
        assert math.isclose(st.mean(benchmark['step_seconds']),benchmark['mean_step_seconds'],rel_tol=1e-12)
        assert math.isclose(4096/benchmark['mean_step_seconds'],benchmark['tokens_per_second'],rel_tol=1e-12)
        assert len(e['generation']) == 3
        for g in e['generation']:
            assert 1 <= len(g['token_ids']) <= 64
            assert tok.decode(g['token_ids']) == g['continuation']
            generations.append({'variant':variant,'seed':seed,**g})
        for ckpt_name in ['best.pt','last.pt']:
            ckpt=torch.load(folder/ckpt_name,map_location='cpu',weights_only=False)
            assert ckpt['step'] == 4000
            assert all(torch.isfinite(t).all().item() for t in ckpt['model'].values())
            assert abs(ckpt['validation']['ce']-r['best_validation']['ce']) < 1e-12
            del ckpt
        source=rows[variant,seed]
        assert source['test_ce'] == e['test']['ce'] and source['test_ppl'] == e['test']['ppl']
        assert source['router_parameters'] == r['parameters']['router_total']
        assert source['total_parameters'] == r['parameters']['total']
        reached=next(c for c in val if c['ppl'] <= 90)
        thresholds.append({'variant':variant,'seed':seed,'validation_ppl_threshold':90,
                           'first_observed_step':reached['step'],
                           'first_observed_training_seconds':reached['training_seconds'],
                           'observed_validation_ppl':reached['ppl']})
        checks.append({'run':folder.name,'passed':True,'validation_reload_ce_abs_difference':
                       abs(e['validation_reload']['ce']-r['best_validation']['ce']),
                       'best_checkpoint_sha256':r['checkpoint_sha256'],
                       'last_checkpoint_sha256':sha(folder/'last.pt'),
                       'generation_count':3,'serial_measured_steps':20})
assert all(len(x)==1 for d in [init,order,positions] for x in d.values())
assert generations == read(ROOT/'generation_samples.json')
assert len(generations) == 54
for a in raw['aggregate']:
    vals=[r for (v,s),r in rows.items() if v==a['variant']]
    for key,value in a.items():
        if key.endswith('_mean') or key.endswith('_std'):
            metric=key.rsplit('_',1)[0]
            calculated=(st.mean if key.endswith('_mean') else st.stdev)(r[metric] for r in vals)
            assert math.isclose(value,calculated,rel_tol=1e-12,abs_tol=1e-12)
for p in raw['paired']:
    a,b=rows[p['variant'],p['seed']],rows[p['baseline'],p['seed']]
    for key,metric in [('delta_ce','test_ce'),('delta_ppl','test_ppl')]:
        assert math.isclose(p[key],a[metric]-b[metric],abs_tol=1e-12)
for name,count in [('runs',18),('aggregate',6),('paired_deltas',30),('paired_summary',10),('routing',72),('curves',1620)]:
    csvrows=list(csv.DictReader((ROOT/f'{name}.csv').open(encoding='utf-8-sig')))
    assert len(csvrows)==count,(name,len(csvrows))
for p in raw['paired_summary']:
    ps=[a for a in raw['paired'] if a['variant']==p['variant'] and a['baseline']==p['baseline']]
    for metric in ['delta_ce','delta_ppl','relative_ppl_percent','router_fraction','total_parameter_fraction','serial_time_ratio']:
        assert math.isclose(p[metric+'_mean'],st.mean(a[metric] for a in ps),abs_tol=1e-12)
        assert math.isclose(p[metric+'_std'],st.stdev(a[metric] for a in ps),abs_tol=1e-12)
variances=[{'variant':a['variant'],'test_ce_sample_variance':a['test_ce_std']**2,
            'test_ppl_sample_variance':a['test_ppl_std']**2} for a in raw['aggregate']]
derived=[]
for a in raw['aggregate']:
    vals=[r for r in thresholds if r['variant']==a['variant']]
    derived.append({'variant':a['variant'],
        'router_reduction_vs_full_mlp_percent':100*(1-a['router_parameters_mean']/50956),
        'total_reduction_vs_full_mlp_percent':100*(1-a['total_parameters_mean']/1698572),
        'first_observed_ppl90_seconds_mean':st.mean(r['first_observed_training_seconds'] for r in vals),
        'first_observed_ppl90_seconds_std':st.stdev(r['first_observed_training_seconds'] for r in vals),
        'first_observed_ppl90_steps':[r['first_observed_step'] for r in vals],
        'test_ppl_by_seed':[rows[a['variant'],s]['test_ppl'] for s in cfg['seeds']]})
write_csv('sample_variances.csv',variances)
write_csv('validation_thresholds.csv',thresholds)
write('REVIEW_METRICS.json',{'variances':variances,'derived':derived,
      'threshold_note':'Post-hoc descriptive PPL90 milestone, evaluated every 500 steps; no interpolation, not exact first passage; excludes eval/checkpoint time.'})
write('REVIEW_AUDIT.json',{'checked_at':datetime.datetime.now().astimezone().isoformat(),
      'passed':True,'training_complete':18,'checkpoint_reevaluated':18,'serial_benchmarks':18,
      'generation_samples':54,'checkpoint_files_checked_on_cpu':36,'data_files_hash_checked':True,
      'frozen_sources_unchanged':True,'shared_init_positions_batch_order_consistent':True,
      'aggregate_and_pairs_recomputed':True,'missing':[],'failed':[],'unfinished':[],
      'runs':checks,'report_and_visual_review_pending':True})
print(json.dumps({'audit':'passed','derived':derived,'variances':variances},ensure_ascii=False,indent=2))
