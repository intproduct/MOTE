"""CPU contract fixtures only; no fabricated experiment observations."""
import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import diagnostic_plan_v14 as p
import run_diagnostics_v14 as s
import formal_diagnostic_budget_v14 as b

c=p.core

@pytest.fixture(scope='module')
def rows():return c.matrix_rows()[0]


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8');return path


@pytest.fixture
def trained(tmp_path,monkeypatch,rows):
    row=copy.deepcopy(rows[0]);row['output']=str(tmp_path/'run')
    monkeypatch.setattr(c,'DIAGNOSTICS',tmp_path/'diagnostics')
    monkeypatch.setattr(p,'TRAINING',tmp_path/'supervisor')
    result=dump(Path(row['output'])/'RESULT.json',dict(cpu_contract_fixture=True))
    retained=[]
    for token in p.TOKENS:
        path=Path(row['output'])/'checkpoints'/f'checkpoint-{token:010d}'
        marker=dump(path/'complete.json',dict(cpu_contract_fixture=True))
        payload=path/'training.pt';payload.write_bytes(b'CPU contract fixture, not a model')
        retained.append(dict(tokens=token,checkpoint=str(path),marker=c.bound(marker),payload=c.bound(payload)))
    audit=dict(passed=True,phase='training_complete_audited',name=row['name'],seed=row['seed'],method=row['method'],cohort=row['cohort'],
               counts=dict(optimizer_updates=1000),primary_tokens=32768000,result=c.bound(result),retained=retained)
    audit_path=dump(p.TRAINING/(row['name']+'.TRAINING_AUDIT.json'),audit)
    return row,audit,audit_path


def cell_fixture(path,row,tokens,kind,checkpoint=None):
    value=dict(cpu_contract_fixture=True,passed=True,phase='complete',method=row['method'],seed=row['seed'],release_sha256=c.RELEASE_HASH,
               protocol_sha256=c.PROTOCOL_HASH,native_precision='torch.bfloat16' if row['cohort']=='bf16' else 'torch.float16',
               physical_device=dict(uuid=row['gpu_uuid']),checkpoint_tokens=tokens,gpu_used=True,native_arithmetic_measured=True)
    if checkpoint:value.update(checkpoint=checkpoint['checkpoint'],checkpoint_payload_sha256=checkpoint['payload']['sha256'])
    result=dump(path/'RESULT.json',value);dump(path/'AUDIT.json',dict(cpu_contract_fixture=True,passed=True,result_sha256=c.sha(result)))
    return p.measurement_cell(path,tokens,kind)


def test_checkpoint_plan_and_atomic_barrier_matches_real_v12_verifier(trained,tmp_path):
    row,audit,audit_path=trained;plan=p.checkpoint_plan(row,audit)
    assert len(plan)==9 and {(v['tokens'],v['kind']) for v in plan}=={(t,k) for t in p.TOKENS for k in p.KINDS}
    cells=[cell_fixture(tmp_path/k,row,0,k) for k in ('full_model','local_precision')]
    cells += [cell_fixture(Path(item['output']),row,item['tokens'],item['kind'],item['checkpoint']) for item in plan]
    receipt=p.publish_barrier(row,audit_path,cells)
    assert receipt and c.read(receipt['path'])['training_audit_sha256']==c.sha(audit_path)
    with pytest.raises(ValueError,match='already'):p.publish_barrier(row,audit_path,cells)


@pytest.mark.parametrize('bad',['method','seed','cohort','tokens','updates','result','missing','duplicate','path','member','payload'])
def test_training_or_checkpoint_mismatch_refused(trained,bad,tmp_path):
    row,audit,_=trained;audit=copy.deepcopy(audit)
    if bad in ('method','cohort'):audit[bad]='different'
    elif bad=='seed':audit['seed']=43
    elif bad=='tokens':audit['primary_tokens']-=1
    elif bad=='updates':audit['counts']['optimizer_updates']=999
    elif bad=='result':audit['result']=c.bound(dump(tmp_path/'foreign.json',{}))
    elif bad=='missing':audit['retained'].pop()
    elif bad=='duplicate':audit['retained'].append(audit['retained'][0])
    elif bad=='path':audit['retained'][0]['checkpoint']=str(tmp_path)
    elif bad=='member':audit['retained'][0]['marker']['path']=str(tmp_path/'foreign.json')
    elif bad=='payload':Path(audit['retained'][0]['payload']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):p.checkpoint_plan(row,audit)


@pytest.mark.parametrize('bad',['seed','method','protocol','precision','gpu','tokens','checkpoint','arithmetic','audit'])
def test_diagnostic_drift_refused(trained,tmp_path,bad):
    row,audit,_=trained;cp=audit['retained'][0];cell=cell_fixture(tmp_path/'cell',row,cp['tokens'],'local_precision',cp)
    result=c.read(cell['result'])
    if bad in ('seed','method'):result[bad]='wrong'
    elif bad=='protocol':result['protocol_sha256']='wrong'
    elif bad=='precision':result['native_precision']='torch.float16'
    elif bad=='gpu':result['physical_device']['uuid']=c.GPUS[1]
    elif bad=='tokens':result['checkpoint_tokens']=0
    elif bad=='checkpoint':result['checkpoint_payload_sha256']='wrong'
    elif bad=='arithmetic':result['native_arithmetic_measured']=False
    dump(Path(cell['result']),result);cell['result_sha256']=c.sha(cell['result'])
    audited=c.read(cell['audit']);audited['result_sha256']=cell['result_sha256']
    if bad=='audit':audited['passed']=False
    dump(Path(cell['audit']),audited);cell['audit_sha256']=c.sha(cell['audit'])
    with pytest.raises(ValueError):p.validate_cell(row,cell,cp)


def ready_state(row,audit_path):
    return dict(time=100,phase='running',failed=False,acceptance=dict(sha256='fixture'),jobs={row['name']:dict(phase='waiting_for_formal_diagnostics',
        training_audit=c.bound(audit_path),exit_codes=[dict(pid=i,exit_code=0,command=row['launch']['argv']) for i in (10,11)])})


def test_training_ready_requires_audit_and_real_command_exits(trained,monkeypatch):
    row,_,path=trained;state=ready_state(row,path);monkeypatch.setattr(s,'matching_run_processes',lambda r:[])
    assert s.training_ready(row,state,'fixture',101)==c.bound(path)
    state['jobs'][row['name']]['phase']='running';assert s.training_ready(row,state,'fixture',101) is None


@pytest.mark.parametrize('bad',['stale','future','failed','stopped','acceptance','native_exit','command','live','audit_path'])
def test_training_handoff_fail_closed(trained,monkeypatch,bad,tmp_path):
    row,_,path=trained;state=ready_state(row,path);monkeypatch.setattr(s,'matching_run_processes',lambda r:[])
    job=state['jobs'][row['name']]
    if bad=='stale':state['time']=-200
    elif bad=='future':state['time']=110
    elif bad=='failed':state['failed']=True
    elif bad=='stopped':state['phase']='stopped_preserved'
    elif bad=='acceptance':state['acceptance']['sha256']='other'
    elif bad=='native_exit':job['exit_codes'][1]['exit_code']=None
    elif bad=='command':job['exit_codes'][1]['command']=['python.exe','different.py']
    elif bad=='live':monkeypatch.setattr(s,'matching_run_processes',lambda r:[{'pid':100}])
    elif bad=='audit_path':job['training_audit']['path']=str(tmp_path/'other.json')
    with pytest.raises(ValueError):s.training_ready(row,state,'fixture',101)


def test_missing_initial_diagnostics_cannot_be_bypassed(rows,monkeypatch,tmp_path):
    monkeypatch.setattr(p,'INITIAL',tmp_path)
    row=next(r for r in rows if r['cohort']=='fp16')
    with pytest.raises(FileNotFoundError):p.initial_cells(row)


def test_all_fifteen_bf16_initial_full_local_references_bound(rows):
    actual=[p.initial_cells(r) for r in rows if r['cohort']=='bf16']
    assert len(actual)==15 and all(len(cells)==2 for cells in actual)


@pytest.mark.parametrize('bad',['gpu','output','checkpoint','nonprotected','config','duplicate_flag'])
def test_formal_resource_context_refuses_cross_run_inputs(rows,bad):
    row=rows[0];config=c.MATRIX.parent/(row['name']+'.json')
    words=['worker.py','--gpu',row['gpu_uuid'],'--output',str(c.DIAGNOSTICS/row['name']/'0327680000-timing'),
           '--checkpoint',str(Path(row['output'])/'checkpoints/checkpoint-0032768000')]
    assert b.bind_context(rows,config,words)[0]==row
    if bad=='gpu':words[2]=c.GPUS[1]
    elif bad=='output':words[4]=str(c.DIAGNOSTICS/'foreign'/'cell')
    elif bad=='checkpoint':words[6]=str(Path(rows[-1]['output'])/'checkpoints/checkpoint-0032768000')
    elif bad=='nonprotected':words[6]=str(Path(row['output'])/'checkpoints/checkpoint-0006553600')
    elif bad=='config':config=c.ROOT/'not-a-configuration.json'
    elif bad=='duplicate_flag':words+=['--gpu',row['gpu_uuid']]
    with pytest.raises(ValueError):b.bind_context(rows,config,words)


def test_formal_adapter_requires_real_acceptance():
    with pytest.raises(ValueError,match='not published'):b.check()


def test_resource_adapter_preserves_interface_and_global_not_legacy_allowances(monkeypatch,tmp_path,rows):
    worker=tmp_path/'production_resource_preflight_v3.py';worker.write_text('fixture')
    monkeypatch.setattr(b,'__file__',str(worker));monkeypatch.setattr(c,'bound',lambda *a,**k:dict(sha256='fixture'))
    monkeypatch.setattr(c,'read',lambda *a:dict(files={}))
    approval={'formal_diagnostic_worker_manifest_sha256':'fixture'}
    monkeypatch.setattr(c,'validate_acceptance',lambda:(rows,{},approval,{'sha256':'fixture'}))
    monkeypatch.setattr(b,'bind_context',lambda *a:(rows[0],tmp_path/'out',tmp_path/'checkpoint'))
    snapshot=dict(fits=True,processing_actual_plus_reserved_bytes=19_000_000_000,free_bytes=8_000_000_000)
    monkeypatch.setattr(c,'capacity_snapshot',lambda *a:snapshot)
    value=b.check();assert value['acceptance_launch_fits'] and value['global_capacity']==snapshot
    assert value['free_bytes']==8_000_000_000 and value['policy']=='formal-v14-global-reservation'


def controller_fixture(monkeypatch,tmp_path,rows,mode):
    selected=[r for r in rows if r['cohort']=='fp16'][:1]
    ctl=s.Diagnostics(mode,selected,{},s.initial_budget(),None if mode=='initial' else {'sha256':'fixture'})
    ctl.out=tmp_path/'controller';ctl.lock=tmp_path/'controller.lock'
    monkeypatch.setattr(ctl,'stopped',lambda:False)
    return ctl


def test_existing_output_and_lock_never_restart(rows,tmp_path,monkeypatch):
    # Constructor's manifest provenance is injected only for this CPU fixture.
    original=c.bound;monkeypatch.setattr(c,'bound',lambda path,*a: {'path':str(path),'sha256':'fixture'} if str(path).endswith('MANIFEST.json') else original(path,*a))
    ctl=controller_fixture(monkeypatch,tmp_path,rows,'initial');ctl.out.mkdir()
    with pytest.raises(ValueError,match='Existing'):ctl.run()
    assert not ctl.lock.exists()


def test_initial_worker_runs_one_missing_cell_once(rows,tmp_path,monkeypatch):
    original=c.bound;monkeypatch.setattr(c,'bound',lambda path,*a: {'path':str(path),'sha256':'fixture'} if str(path).endswith('MANIFEST.json') else original(path,*a))
    ctl=controller_fixture(monkeypatch,tmp_path,rows,'initial');calls=[]
    monkeypatch.setattr(ctl,'cell',lambda *args:calls.append(args) or {'fixture':True})
    ctl.worker(ctl.rows[0]['gpu_uuid']);assert len(calls)==1 and calls[0][1]=='local_precision'


def test_formal_worker_consumes_nine_cells_then_publishes(trained,rows,tmp_path,monkeypatch):
    row,audit,path=trained
    original=c.bound;monkeypatch.setattr(c,'bound',lambda path,*a: {'path':str(path),'sha256':'fixture'} if str(path).endswith('MANIFEST.json') else original(path,*a))
    ctl=controller_fixture(monkeypatch,tmp_path,rows,'formal');ctl.rows=[row];calls=[];barriers=[]
    monkeypatch.setattr(ctl,'wait_training',lambda r:c.bound(path));monkeypatch.setattr(p,'initial_cells',lambda r:[{'initial':1},{'initial':2}])
    monkeypatch.setattr(ctl,'cell',lambda r,k,o,cp:calls.append((cp['tokens'],k)) or {'fixture':True})
    monkeypatch.setattr(p,'publish_barrier',lambda r,a,cells:barriers.append(cells) or {'fixture':True})
    ctl.worker(row['gpu_uuid']);assert len(calls)==9 and len(barriers)==1 and len(barriers[0])==11
    assert ctl.state['jobs'][row['name']]['phase']=='complete_audited'


def test_formal_diagnostic_failure_stops_future_training_no_retry(trained,rows,tmp_path,monkeypatch):
    row,_,path=trained
    original=c.bound;monkeypatch.setattr(c,'bound',lambda path,*a: {'path':str(path),'sha256':'fixture'} if str(path).endswith('MANIFEST.json') else original(path,*a))
    ctl=controller_fixture(monkeypatch,tmp_path,rows,'formal');ctl.rows=[row];ctl.out.mkdir();attempts=[]
    monkeypatch.setattr(ctl,'wait_training',lambda r:c.bound(path));monkeypatch.setattr(p,'initial_cells',lambda r:[])
    def failed(*args):attempts.append(1);raise ValueError('CPU fault fixture')
    monkeypatch.setattr(ctl,'cell',failed);ctl.worker(row['gpu_uuid'])
    assert len(attempts)==1 and ctl.drain.is_set() and ctl.state['failed']
    assert (p.TRAINING/'STOP').exists() and not (c.ROOT/'STOP').exists()
    first=(p.TRAINING/'STOP').read_bytes();ctl.stop_future_training('later');assert (p.TRAINING/'STOP').read_bytes()==first


def test_cpu_tests_never_initialize_cuda():
    import torch
    assert not torch.cuda.is_initialized()


def test_stable_idle_is_reset_by_busy_observation(rows,tmp_path,monkeypatch):
    original=c.bound;monkeypatch.setattr(c,'bound',lambda path,*a: {'path':str(path),'sha256':'fixture'} if str(path).endswith('MANIFEST.json') else original(path,*a))
    ctl=controller_fixture(monkeypatch,tmp_path,rows,'initial');clock=[0];calls=[]
    from fitmotn.experiments.qwen_joint import resources
    def snapshot(gpu):
        calls.append(clock[0]);return dict(idle=clock[0]!=30,epoch=clock[0],uuid=gpu)
    monkeypatch.setattr(resources,'gpu_snapshot',snapshot)
    monkeypatch.setattr(s,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda delay:clock.__setitem__(0,clock[0]+delay)))
    idle=ctl.wait_idle(ctl.rows[0],'fixture')
    assert idle['first']['epoch']==35 and idle['last']['epoch']==95 and idle['separation_seconds']==60


def test_actual_native_cpu_exit_observer_can_be_reused_without_training_files(rows,tmp_path,monkeypatch):
    original=c.bound;monkeypatch.setattr(c,'bound',lambda path,*a: {'path':str(path),'sha256':'fixture'} if str(path).endswith('MANIFEST.json') else original(path,*a))
    ctl=controller_fixture(monkeypatch,tmp_path,rows,'initial')
    script=tmp_path/'cpu_fixture.py';script.write_text('import time\ntime.sleep(3)\n')
    command=[str(c.ROOT/'envs/qwen-native-py312-cu128/Scripts/python.exe'),str(script)]
    process=s.subprocess.Popen(command,creationflags=s.subprocess.CREATE_NO_WINDOW)
    row=dict(name='fixture',output=str(tmp_path/'not-training'),launch=dict(argv=command))
    try:exits=s.TrainingSupervisor.observe_training(ctl,row,process)
    finally:process.wait(timeout=20)
    assert len(exits)==2 and all(v['exit_code']==0 for v in exits) and not ctl.drain.is_set()


def test_complete_initial_cohort_writes_index_once(rows,tmp_path,monkeypatch):
    original=c.bound;monkeypatch.setattr(c,'bound',lambda path,*a: {'path':str(path),'sha256':'fixture'} if str(path).endswith('MANIFEST.json') else original(path,*a))
    ctl=controller_fixture(monkeypatch,tmp_path,rows,'initial')
    def cell(row,kind,output,checkpoint=None):
        value=cell_fixture(tmp_path/'raw-fixture',row,0,kind)
        ctl.update(row['name'],phase='complete_audited',cell=value);return value
    monkeypatch.setattr(ctl,'cell',cell);ctl.run()
    assert ctl.state['phase']=='complete_audited' and not ctl.lock.exists()
    index=c.read(ctl.out/'INITIAL_LOCAL_INDEX.json');assert len(index['measurements'])==1


def test_formal_command_only_uses_retained_checkpoint(trained):
    row,audit,_=trained;item=p.checkpoint_plan(row,audit)[0]
    command=p.worker_command(row,item['kind'],item['output'],item['checkpoint'])
    assert command[2]==str(p.WORKERS/p.ENTRIES[item['kind']])
    assert '--initial' not in command and command[command.index('--checkpoint')+1]==item['checkpoint']['checkpoint']
