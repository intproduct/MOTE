"""One-off sequential continuation; no overlapping GPU workers or retries."""
from pathlib import Path
import json,os,subprocess,sys,time,datetime
repo=Path(__file__).resolve().parents[2];root=repo/'outputs/routing_stability_round1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
record={'pid':os.getpid(),'started':datetime.datetime.now().astimezone().isoformat(),
        'scope':'wait for pilot_v2, probe v2, pilot_v3, probe v3; never starts formal'}
(root/'VALIDATION_QUEUE.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
deadline=time.time()+600
while (root/'PIPELINE_LOCK.json').exists():
    if time.time()>deadline:raise RuntimeError('pilot_v2 did not finish within wait budget; inspect, do not duplicate')
    time.sleep(5)
status=read(root/'STATUS.json')
assert status['state']=='pilot_passed' and Path(status['group']).name=='2c58817d07ec',status
for module,config in [('probe','pilot_v2.json'),('pipeline','pilot_v3.json'),('probe','pilot_v3.json')]:
    print(datetime.datetime.now().astimezone().isoformat(),module,config,flush=True)
    subprocess.run([sys.executable,'-u','-m',f'fitmotn.experiments.routing_stability.{module}',
                    '--config',f'experiments/routing_stability/configs/{config}'],cwd=repo,check=True)
record.update(state='complete',finished=datetime.datetime.now().astimezone().isoformat())
(root/'VALIDATION_QUEUE.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('VALIDATION QUEUE COMPLETE',flush=True)
