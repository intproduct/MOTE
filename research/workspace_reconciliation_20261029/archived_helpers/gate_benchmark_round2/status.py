"""Compact status without consulting final evaluation scores for selection."""
import json
import math
from pathlib import Path
from datetime import datetime

root=Path(__file__).resolve().parent
groups=[]
for path in sorted(root.glob('*/*/config.json')):
    c=json.loads(path.read_text())
    if not {'gates','seeds','mlp','bert'} <= c.keys():
        continue
    expected=sum(len(c[s]['tasks']) for s in ['mlp','bert'])*len(c['gates'])*len(c['seeds'])
    complete, active, issues = [], [], []
    reused=0
    for stage in ['mlp','bert']:
        for task in c[stage]['tasks']:
            for seed in c['seeds']:
                for gate in c['gates']:
                    d=path.parent/f'{stage}_{task}_{gate}_seed{seed}'
                    if (d/'result.json').exists():
                        r=json.loads((d/'result.json').read_text())
                        complete.append(d.name)
                        reused+=int((d/'reference.json').exists())
                        bad=[n for n,v in r.get('router_update_audit',{}).items() if not v['changed']]
                        if bad:issues.append({'run':d.name,'unchanged':bad})
                    elif d.exists():
                        tr=[]
                        if (d/'curves.jsonl').exists():
                            for line in (d/'curves.jsonl').read_text().splitlines():
                                try:
                                    row=json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if 'train_ce' in row:tr.append(row)
                        active.append({'run':d.name,'step':tr[-1]['step'] if tr else 0,
                                       'planned_steps':c[stage]['steps'],
                                       'ce':tr[-1]['train_ce'] if tr else None})
                        if tr and any(not v['finite'] for v in tr[0].get('router_gradient_audit',{}).values()):
                            issues.append({'run':d.name,'error':'nonfinite/absent router gradient'})
    execution=path.parent/'execution.jsonl'
    interruptions=[]
    if execution.exists():
        events=[json.loads(line) for line in execution.read_text().splitlines()]
        issues += [e for e in events if e['event']=='end' and e['exit_code'] != 0]
        interruptions=[e for e in events if e['event']=='interrupted']
    groups.append({'group':str(path.parent),'completed':len(complete),'expected':expected,
                   'completed_new':len(complete)-reused,'reused_historical':reused,
                   'active':active,'issues':issues,'interrupted_attempts':len(interruptions),
                   'interrupted_runs_recovered':sum((path.parent/e['run']/'result.json').exists() for e in interruptions),
                   'execution_complete':(path.parent/'execution_complete.json').exists()})
result={'time':datetime.now().astimezone().isoformat(),'groups':groups}
(root/'STATUS.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
