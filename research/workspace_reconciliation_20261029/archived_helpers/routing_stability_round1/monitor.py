"""Read-only CPU monitoring: never launches/stops GPU work or rewrites runs."""
from pathlib import Path
import json,hashlib,datetime
root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
active=read(root/'ACTIVE_PROCESS.json')
group=Path(active['group']);cfg=read(group/'config.json') if (group/'config.json').exists() else read(Path(active['config']))
done=[];pending=[];bad=[];latest=None
for seed in cfg['seeds']:
    for variant in cfg['gates']:
        path=group/'runs'/f'{variant}_s{seed}'
        if (path/'result.json').exists():
            r=read(path/'result.json')
            if r['finite_steps']!=cfg['train']['steps'] or r['nonzero_router_grad_steps']!=r['finite_steps']:
                bad.append(path.name)
            done.append(path.name)
        else:
            pending.append(path.name)
            if (path/'curves.jsonl').exists():
                lines=(path/'curves.jsonl').read_text().splitlines()
                records=[json.loads(x) for x in lines]
                latest={'run':path.name,'last':records[-1] if records else None}
        for filename in ['curves.jsonl','routing.jsonl']:
            file=path/filename
            if file.exists() and any(t in file.read_text() for t in ['NaN','Infinity']):bad.append(str(file))
frozen=read(group/'frozen_sources.json')
changed=[f for f,h in frozen.items() if hashlib.sha256((root.parents[1]/f).read_bytes()).hexdigest()!=h]
result={'time':datetime.datetime.now().astimezone().isoformat(),'active':active,'status':read(root/'STATUS.json'),
        'complete':len(done),'expected':len(done)+len(pending),'completed':done,'pending':pending,'latest':latest,
        'finite_problems':bad,'changed_frozen_sources':changed,
        'evaluations':len(list((group/'runs').glob('*/evaluation.json'))),
        'stability_probes':len(list((group/'runs').glob('*/stability.json')))}
(root/'MONITOR_STATUS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
