"""Independent CPU-only delivery review; never trains or rewrites raw runs."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import math
import statistics as st
import torch
from fitmotn.experiments.routing_stability.model import CausalLM
from fitmotn.experiments.gate_benchmark.models import parameter_report

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
GROUP = ROOT / 'formal/756b04397b2d'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg = read(GROUP/'config.json')
summary = read(ROOT/'raw_summary.json')
assert cfg == summary['config'] == read(REPO/'experiments/routing_stability/configs/titan_v.json')
config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
assert config_hash == '756b04397b2d8392d676c2c07833d28a9825a510da312c39c11604243f6f7296'
for name, value in read(GROUP/'frozen_sources.json').items():
    assert sha(REPO/name) == sha(GROUP/'source_snapshot'/name) == value, name
health = read(ROOT/'LATEST_HEALTH_AUDIT.json')
assert health['complete'] == 18 and not health['problems']
assert health['config_and_frozen_sources_unchanged']
assert health['shared_and_pair_initialization_and_sampling_consistent']
assert len(health['before_margin_pair_checks']) == 9
assert all(x['common_logged_steps_before_margin']==41 and x['max_ce_difference_before_margin']==0 for x in health['before_margin_pair_checks'])
assert not (ROOT/'PIPELINE_LOCK.json').exists()
assert (ROOT/'logs/formal_pipeline.err').stat().st_size == 0
assert not read(GROUP/'missing_failed_unfinished.json')['training']
assert not read(GROUP/'missing_failed_unfinished.json')['postprocessing']

def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(a,b)

def finite(x):
    if isinstance(x,float): assert math.isfinite(x), x
    elif isinstance(x,dict):
        for v in x.values(): finite(v)
    elif isinstance(x,list):
        for v in x: finite(v)

expected={f'{v}_s{s}' for v in cfg['gates'] for s in cfg['seeds']}
assert {p.name for p in (GROUP/'runs').iterdir() if p.is_dir()} == expected
assert len(summary['runs']) == 18
lookup={(x['variant'],x['seed']):x for x in summary['runs']}
assert len(lookup)==18
data=read(REPO/cfg['data_root']/'manifest.json')
assert data['tokenizer_training_split']=='train'
records=[]; generations=[]; curve_rows=[]; route_rows=[]
conditions={'repeat','batch16_vs1','input_bf16_roundtrip','input_fp16_roundtrip',
            'input_uniform_rms_0.001','input_uniform_rms_0.01',
            'propagated_input_bf16_roundtrip','propagated_input_fp16_roundtrip'}
tail_conditions={'batch16_vs1','input_fp16_roundtrip','input_bf16_roundtrip'}
common_starts=None
all_raw_files={}
for seed in cfg['seeds']:
    for variant in cfg['gates']:
        path=GROUP/'runs'/f'{variant}_s{seed}'
        for name in ['manifest.json','result.json','evaluation.json','stability.json','best.pt','last.pt','curves.jsonl','routing.jsonl','batch_starts.npy']:
            assert (path/name).is_file() and (path/name).stat().st_size>0
            all_raw_files[str((path/name).relative_to(ROOT))]=sha(path/name)
        r,e,p,m=[read(path/name) for name in ['result.json','evaluation.json','stability.json','manifest.json']]
        for x in [r,e,p,m]:
            finite(x); assert x['variant']==variant and x['seed']==seed
        assert r['checkpoint_sha256']==e['checkpoint_sha256']==p['checkpoint_sha256']==sha(path/'best.pt')
        assert r['config_hash']==m['config_hash']==config_hash
        assert r['steps']==r['finite_steps']==r['nonzero_router_grad_steps']==4000
        assert r['tokens_seen']==16384000 and r['best_step']==4000
        assert r['shared_parameters_updated'] and all(x>0 for x in r['router_parameter_max_changes'].values())
        assert m['data_manifest_sha256']==sha(REPO/cfg['data_root']/'manifest.json')
        model=CausalLM(cfg,variant,seed).cpu()
        for name in ['best.pt','last.pt']:
            checkpoint=torch.load(path/name,map_location='cpu',weights_only=False)
            assert checkpoint['variant']==variant and checkpoint['seed']==seed and checkpoint['step']==4000
            assert checkpoint['config']==cfg
            assert all(torch.isfinite(t).all().item() for t in checkpoint['model'].values())
            model.load_state_dict(checkpoint['model'],strict=True)
        assert parameter_report(model)==r['parameters']==m['parameters']
        del model,checkpoint
        for split,value in [('validation',e['validation_reload']),('test',e['test'])]:
            assert value['tokens']==data['splits'][split]['tokens']-1
            close(value['ce'],value['nll_sum']/value['tokens'])
            close(value['ppl'],math.exp(value['ce']))
        close(e['validation_reload']['ce'],r['best_validation']['ce'])
        assert set(e['test']['routing'])==set(r['parameters']['routers'])
        for name,z in e['test']['routing'].items():
            assert z['unused_experts']==sum(t==0 for t in z['load'])
            route_rows.append({'variant':variant,'seed':seed,'module':name,**z})
        serial=e['serial_benchmark']
        assert serial['warmup_steps']==5 and serial['measured_steps']==20
        assert len(serial['step_seconds'])==20 and min(serial['step_seconds'])>0
        close(serial['mean_step_seconds'],st.mean(serial['step_seconds']))
        close(serial['tokens_per_second'],4096/serial['mean_step_seconds'])
        assert {(b['batch'],b['length']) for b in e['inference_benchmark']}=={(16,128),(1,128),(1,1)}
        for b in e['inference_benchmark']:
            assert b['warmup']==10 and b['repeats']==25 and len(b['seconds'])==25 and min(b['seconds'])>0
            close(b['median_seconds'],st.median(b['seconds']))
            close(b['tokens_per_second'],b['batch']*b['length']/b['median_seconds'])
        assert len(e['generation'])==3
        for g in e['generation']:
            assert len(g['token_ids'])==64 and all(0<=t<4096 for t in g['token_ids']) and g['continuation']
            generations.append({'variant':variant,'seed':seed,**g})
        starts=p['protocol']['window_starts']
        assert len(starts)==128 and len(set(starts))==128
        assert all(b-a>=128 for a,b in zip(starts,starts[1:]))
        assert min(starts)>=0 and max(starts)+128<data['splits']['validation']['tokens']
        if common_starts is None: common_starts=starts
        assert common_starts==starts
        assert len(p['rows'])==256 and len(p['gap_distributions'])==32 and len(p['end_to_end'])==24
        modules=set(r['parameters']['routers']); offsets=set(range(0,128,16))
        assert {(x['module'],x['batch_offset'],x['condition']) for x in p['rows']}=={(m,o,c) for m in modules for o in offsets for c in conditions}
        for x in p['rows']:
            assert x['n']==2048 and 0<=x['changed']<=x['replaced']<=4*x['changed']
            assert x['measured_bound_safe']+x['changed']<=x['n']
            if x['condition']=='repeat': assert x['changed']==0
        for condition in conditions:
            raw=[x for x in p['rows'] if x['condition']==condition]
            z=next(x for x in summary['stability'] if (x['variant'],x['seed'],x['condition'])==(variant,seed,condition))
            assert z['decisions']==sum(x['n'] for x in raw)==65536
            assert z['changed']==sum(x['changed'] for x in raw)
            close(z['flip_percent'],100*z['changed']/z['decisions'])
            close(z['replaced_percent'],100*sum(x['replaced'] for x in raw)/(65536*4))
        assert {(x['batch_offset'],x['condition']) for x in p['end_to_end']}=={(o,c) for o in offsets for c in tail_conditions}
        for condition in tail_conditions:
            raw=[x for x in p['end_to_end'] if x['condition']==condition]
            z=next(x for x in summary['error_tails'] if (x['variant'],x['seed'],x['condition'])==(variant,seed,condition))
            assert z['predictions']==sum(x['n'] for x in raw)==16384
            close(z['target_logprob_mae'],sum(x['target_logprob_abs_sum'] for x in raw)/16384)
            close(z['target_logprob_max'],max(x['target_logprob_abs_max'] for x in raw))
            close(z['target_logprob_p99_batch_mean'],st.mean(x['target_logprob_abs_p99'] for x in raw))
            close(z['target_logprob_gt01_percent'],100*sum(x['target_logprob_abs_gt_01'] for x in raw)/16384)
            close(z['argmax_changed_percent'],100*sum(x['next_token_changed'] for x in raw)/16384)
        row=lookup[variant,seed]
        for key,value in [('test_ce',e['test']['ce']),('test_ppl',e['test']['ppl']),('router_parameters',r['parameters']['router_total']),('total_parameters',r['parameters']['total']),('serial_tokens_per_second',serial['tokens_per_second'])]:close(row[key],value)
        curves=[json.loads(line) for line in (path/'curves.jsonl').read_text().splitlines()]
        assert {x['step'] for x in curves if x['kind']=='validation'}==set(range(0,4001,500))
        curve_rows.extend({'variant':variant,'seed':seed,**x} for x in curves)
        records.append({'run':path.name,'best_sha256':r['checkpoint_sha256'],'parameters':{k:r['parameters'][k] for k in ['router_total','total']},'best_step':4000,'validation_tokens':322802,'test_tokens':368043,'validation_reload_ce_difference':e['validation_reload']['ce']-r['best_validation']['ce'],'stability_decisions_per_condition':65536,'finite_checkpoint_tensors':True,'strict_cpu_load_both_checkpoints':True})
assert read(ROOT/'generation_samples.json')==generations and len(generations)==54
for agg in summary['aggregate']:
    rows=[x for x in summary['runs'] if x['variant']==agg['variant']]
    assert len(rows)==agg['n']==3
    for key,value in rows[0].items():
        if isinstance(value,(float,int)) and key!='seed':
            close(agg[key+'_mean'],st.mean(x[key] for x in rows))
            close(agg[key+'_std'],st.stdev(x[key] for x in rows))
for pair in summary['paired']:
    a=lookup[pair['variant'],pair['seed']];b=lookup[pair['baseline'],pair['seed']]
    close(pair['delta_ce'],a['test_ce']-b['test_ce'])
    close(pair['ppl_change_percent'],100*(a['test_ppl']/b['test_ppl']-1))
    close(pair['bf16_flip_ratio'],a['flip_input_bf16_roundtrip_percent']/b['flip_input_bf16_roundtrip_percent'])
    close(pair['inference_time_ratio'],a['inference_b16_l128_ms']/b['inference_b16_l128_ms'])

csv_sizes={}
for name,key in [('runs','runs'),('aggregate','aggregate'),('paired_deltas','paired'),('paired_summary','paired_summary'),('stability','stability'),('error_tails','error_tails'),('curves',None),('routing',None)]:
    expected_rows=summary[key] if key else (curve_rows if name=='curves' else route_rows)
    with (ROOT/f'{name}.csv').open(encoding='utf-8-sig',newline='') as f: actual=list(csv.DictReader(f))
    assert len(actual)==len(expected_rows),(name,len(actual),len(expected_rows))
    for a,b in zip(actual,expected_rows):
        for k,v in b.items():
            if isinstance(v,(float,int)):close(float(a[k]),v)
            elif v is None: assert a[k]==''
            else: assert a[k]==str(v),(name,k)
    csv_sizes[name]=len(actual)
finite(summary)
checks={'time':dt.datetime.now().astimezone().isoformat(),'passed':True,'task_complete':False,
        'scope':'CPU delivery data review; report signoff recorded separately',
        'config_hash':config_hash,'training':18,'best_checkpoint_reevaluations':18,
        'cpu_loaded_checkpoints':36,'serial_train_benchmarks':18,'inference_benchmark_shapes':54,
        'inference_timing_samples':18*3*25,'serial_training_timing_samples':18*20,
        'stability_records':18,'stability_raw_comparisons':18*256,'generation_samples':54,
        'finite_training_steps':72000,'nonzero_router_gradient_steps':72000,
        'config_and_frozen_sources_unchanged':True,'shared_initialization_sampling_checks':health['before_margin_pair_checks'],
        'csv_row_counts':csv_sizes,'run_checks':records,'missing':[],'failed':[],'unfinished':[],
        'pipeline_lock_absent':True,'pipeline_stderr_empty':True,'raw_artifact_sha256':all_raw_files}
(ROOT/'REVIEW_DATA_AUDIT.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in checks.items() if k not in ['run_checks','raw_artifact_sha256','shared_initialization_sampling_checks']},ensure_ascii=False,indent=2))
