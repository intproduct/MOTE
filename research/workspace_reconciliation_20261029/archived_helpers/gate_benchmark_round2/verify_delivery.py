"""Read-only final matrix, prediction, summary and report verification."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import statistics as st

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
GROUP = ROOT/'formal/20c0fefebf11'
C = json.loads((GROUP/'config.json').read_text())
errors, missing, unfinished, failed = [], [], [], []
expected = [(s,t,g,n) for s in ['mlp','bert'] for t in C[s]['tasks'] for n in C['seeds'] for g in C['gates']]

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def finite(v):
    if isinstance(v,float): return math.isfinite(v)
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,list): return all(finite(x) for x in v)
    return True

def require(condition, message):
    if not condition: errors.append(message)

def csv_read(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

audit = read(GROUP/'audit.json')
require(audit['passed'] and audit['completed_audited']==126 and audit['issues']==[], 'formal audit')
pilot = read(ROOT/'pilot/34c8d881126c/audit.json')
require(pilot['passed'] and pilot['completed_audited']==26, 'pilot audit')
recovery = read(ROOT/'recovery_audit.json')
require(recovery['require_complete'] and recovery['passed'] and recovery['all_recovered'], 'recovery audit')
require(sum(x['compared_curve_rows'] for x in recovery['checks'])==105, 'recovery row count')
require('146 passed' in (ROOT/'logs/regression_1822.log').read_text(), 'regression log')
frozen = read(ROOT/'source_snapshot_formal/files_sha256.json')
require(all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in frozen.items()), 'frozen source files')

origin, commits, labels_by_task, sources = Counter(), Counter(), {}, set()
results = {}
predictions_checked = 0
for stage, task, gate, seed in expected:
    name=f'{stage}_{task}_{gate}_seed{seed}'
    dest=GROUP/name
    for file in ['manifest.json','initial_dev.json','curves.jsonl','best.pt','result.json']:
        p=dest/file
        if not p.is_file() or not p.stat().st_size: missing.append(str(p))
    if not (dest/'result.json').exists(): unfinished.append(name); continue
    m,r=read(dest/'manifest.json'),read(dest/'result.json')
    results[task,gate,seed]=r
    origin['historical_reused' if (dest/'reference.json').exists() else 'new_training']+=1
    commits[m['environment']['git_commit']]+=1
    if not (dest/'reference.json').exists():
        sources.add(json.dumps(m['environment']['source_sha256'],sort_keys=True))
    serial=ROOT/'formal/serial_efficiency'/name
    for file in ['measurement.json','final_predictions.json','final_routing_reevaluation.json']:
        if not (serial/file).is_file(): missing.append(str(serial/file))
    if not (serial/'measurement.json').exists(): continue
    measure=read(serial/'measurement.json'); pred=read(serial/'final_predictions.json'); routes=read(serial/'final_routing_reevaluation.json')
    require(all(measure[k]==v for k,v in [('stage',stage),('task',task),('gate',gate),('seed',seed)]), name+' measurement identity')
    require(measure['final_reevaluation_matches'] and measure['routing_reevaluation_matches'] and pred['matches_recorded'] and routes['matches_recorded'], name+' reeval status')
    require(measure['measured_steps']==10 and measure['warmup_steps']==3 and len(measure['step_seconds'])==10, name+' timing protocol')
    require(finite(measure) and all(t>0 for t in measure['step_seconds']), name+' finite timing')
    require(abs(st.mean(measure['step_seconds'])-measure['seconds_per_step'])<1e-9,name+' timing mean')
    require(all(abs(pred['metrics'][k]-v)<=(1e-6 if k=='ce' else 0) for k,v in r['final'].items()), name+' final metric equality')
    require(routes['routing']==r['final_routing'],name+' routing equality')
    require(pred['source_split']==('test' if stage=='mlp' else 'validation'),name+' split')
    labels,preds=pred['labels'],pred['predictions']
    require(len(labels)==len(preds)==m['data']['sizes']['test'],name+' evaluation size')
    require(sum(pred['label_counts'])==len(labels)==sum(pred['prediction_counts']),name+' prediction counts')
    require(abs(sum(x==y for x,y in zip(labels,preds))/len(labels)-pred['metrics']['accuracy'])<1e-7,name+' accuracy from predictions')
    if task in labels_by_task: require(labels_by_task[task]==labels,name+' paired labels')
    else: labels_by_task[task]=labels
    require(sum(v>0 for v in pred['prediction_counts'])>1,name+' single-class prediction collapse')
    predictions_checked+=1
require(origin=={'new_training':90,'historical_reused':36},'training origin counts')
require(len(sources)==1,'new training source identity')
require(len(results)==126 and predictions_checked==126,'complete predictions')

executions=[json.loads(v) for v in (GROUP/'execution.jsonl').read_text(encoding='utf-8-sig').splitlines()]
failed=[e for e in executions if e['event']=='end' and e['exit_code']!=0]
require(not failed,'nonzero execution exits')
require(sum(e['event']=='interrupted' for e in executions)==6,'retained interruptions')
summary=read(ROOT/'formal_analysis.json')
require(summary['completed']==126 and summary['new_training']==90 and summary['reused_historical']==36,'analysis counts')
require(finite(summary),'analysis finite')
for a in summary['aggregates']:
    rr=[results[a['task'],a['gate'],seed] for seed in C['seeds']]
    for metric in ['accuracy','score']+(['f1'] if a['task']=='mrpc' else []):
        vals=[r['final'][metric] for r in rr]
        require(abs(st.mean(vals)-a['final_'+metric+'_mean'])<1e-12,'aggregate mean')
        require(abs(st.stdev(vals)-a['final_'+metric+'_std'])<1e-12,'aggregate sample std')
    for baseline in ['mlp','mlp_half','compact_born']:
        diffs=[100*(r['final']['score']-results[a['task'],baseline,r['seed']]['final']['score']) for r in rr]
        require(abs(st.mean(diffs)-a['delta_vs_'+baseline+'_pp_mean'])<1e-10,'paired mean')
        require(abs(st.stdev(diffs)-a['delta_vs_'+baseline+'_pp_std'])<1e-10,'paired sample std')
counts={}
for name,expected_count in [('formal_runs.csv',126),('formal_aggregate.csv',42),('formal_paired_deltas.csv',378),('formal_curves.csv',5166),('formal_routing.csv',882),('formal/runs.csv',126),('formal/serial_efficiency_runs.csv',126),('formal/serial_efficiency_summary.csv',42)]:
    counts[name]=len(csv_read(ROOT/name))
    require(counts[name]==expected_count,name+' row count')
require(len(read(ROOT/'formal/summary.json')['groups'])==42,'repository summarize groups')

figures=['parameter_accuracy.png','half_budget_paired.png','mlp_dev_step.png','mlp_dev_training_seconds.png','bert_dev_step.png','bert_dev_training_seconds.png','serial_efficiency.png']
for f in figures: require((ROOT/'figures'/f).stat().st_size>10000,f+' figure')
report=(ROOT/'REPORT_zh.md').read_text(encoding='utf-8')
require('此处需结合' not in report and '填写人工审阅结论' not in report,'report placeholder')
for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report): require((ROOT/target).is_file(),'report image link '+target)
require('c7d6690' in report and '0x00000133' in report and 'validation' in report,'report provenance')
record={'time':datetime.now().astimezone().isoformat(),'passed':not errors and not missing and not unfinished and not failed,
        'expected':126,'completed':len(results),'new_training':origin['new_training'],'reused_historical':origin['historical_reused'],
        'pilot_audited':26,'formal_checkpoint_audited':126,'final_predictions_and_routing_reevaluated':predictions_checked,'serial_measurements':126,
        'failed_runs':failed,'missing_files':missing,'unfinished_runs':unfinished,'issues':errors,
        'interrupted_attempts_retained':6,'interrupted_runs_recovered':6,'recovered_curve_rows_exact':105,
        'new_training_source_hash_sets':len(sources),'frozen_source_files_unchanged':len(frozen),'manifest_git_commit_counts':dict(commits),
        'csv_row_counts':counts,'figures_visually_reviewed':figures,'figure_review':'All seven PNGs individually viewed; axes, legends and Chinese labels readable; no clipping. Concurrent-time curves explicitly distinguished from independent serial timing.',
        'report_sha256':hashlib.sha256((ROOT/'REPORT_zh.md').read_bytes()).hexdigest(),
        'unresolved_limits':['Windows bugcheck 0x00000133 precise driver cause unknown','Three seeds, short classification runs and limited structure search; no general quantum/entanglement advantage established','Serial efficiency is short continuation with fresh AdamW, not full equal-wallclock retraining or isolated inference benchmark']}
(ROOT/'COMPLETION_AUDIT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
raise SystemExit(0 if record['passed'] else 1)
