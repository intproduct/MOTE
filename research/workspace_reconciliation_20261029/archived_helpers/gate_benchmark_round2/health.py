"""Cheap read-only checks while training; never changes frozen experiment files."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

root=Path(__file__).resolve().parent
workspace=root.parents[1]
p=root/'formal/20c0fefebf11'
frozen=json.loads((root/'source_snapshot_formal/files_sha256.json').read_text())
issues=[]
for name,expected in frozen.items():
    if hashlib.sha256((workspace/name).read_bytes()).hexdigest()!=expected:
        issues.append({'source_changed':name})
rows=[]
for f in p.glob('*/manifest.json'):
    if (f.parent/'reference.json').exists():continue
    m=json.loads(f.read_text())
    baseline=p/f"{m['stage']}_{m['task']}_mlp_seed{m['seed']}"/'manifest.json'
    b=json.loads(baseline.read_text())
    for key in ['shared_state_sha256']:
        if m[key]!=b[key]:issues.append({'run':f.parent.name,'mismatch':key})
    if m['data']['tensor_file_sha256']!=b['data']['tensor_file_sha256']:
        issues.append({'run':f.parent.name,'mismatch':'data'})
    if m['parameters']['positions']!=b['parameters']['positions']:
        issues.append({'run':f.parent.name,'mismatch':'positions'})
    curves=f.parent/'curves.jsonl'
    train=[]
    if curves.exists():
        for line in curves.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue  # writer may be appending its final line
            if 'train_ce' in r:train.append(r)
        for r in train:
            for key in ['train_ce','aux','grad_norm']:
                if r[key] is None or not math.isfinite(r[key]):
                    issues.append({'run':f.parent.name,'step':r['step'],'nonfinite':key})
        if train and any(not all(v.values()) for v in train[0]['router_gradient_audit'].values()):
            issues.append({'run':f.parent.name,'first_gradient_issue':True})
    done=(f.parent/'result.json').exists()
    if done:
        result=json.loads((f.parent/'result.json').read_text())
        if any(not v['changed'] or v['l2_delta']<=0 for v in result['router_update_audit'].values()):
            issues.append({'run':f.parent.name,'parameter_update_issue':True})
    rows.append({'run':f.parent.name,'completed':done,'latest_step':train[-1]['step'] if train else 0})
execution=p/'execution.jsonl'
if execution.exists():
    for line in execution.read_text().splitlines():
        event=json.loads(line)
        if event['event']=='end' and event['exit_code']:
            issues.append(event)
record={'time':datetime.now().astimezone().isoformat(),'source_frozen_verified':not any('source_changed' in x for x in issues),
        'new_runs_checked':len(rows),'new_completed':sum(r['completed'] for r in rows),'issues':issues,'runs':rows}
(root/'live_health.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='runs'},indent=2))
raise SystemExit(1 if issues else 0)
