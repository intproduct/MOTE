import copy
import json
from pathlib import Path
import subprocess
import time

import psutil
import pytest
import torch

import formal_runtime_v12 as c
import run_formal_matrix_v12 as s
from watch_pro_product_exit_v9 import HeldProcess,atomic_json


@pytest.fixture(scope='module')
def rows():return c.matrix_rows()[0]


def test_missing_final_acceptance_refuses_before_any_effect(tmp_path,monkeypatch):
    missing=tmp_path/'INTEGRATION_ACCEPTANCE.json';monkeypatch.setattr(c,'ACCEPTANCE',missing)
    with pytest.raises(ValueError,match='not published'):c.validate_acceptance(missing)
    assert list(tmp_path.iterdir())==[] and not torch.cuda.is_initialized()


def test_preparation_is_not_formal_acceptance(tmp_path,monkeypatch):
    p=tmp_path/'INTEGRATION_ACCEPTANCE.json';p.write_text(json.dumps({'formal_approved':False}))
    monkeypatch.setattr(c,'ACCEPTANCE',p)
    with pytest.raises(ValueError,match='incomplete'):c.validate_acceptance(p)


def test_approval_without_bound_matrix_refused(tmp_path,monkeypatch):
    p=tmp_path/'INTEGRATION_ACCEPTANCE.json';p.write_text(json.dumps({'formal_approved':True}))
    monkeypatch.setattr(c,'ACCEPTANCE',p)
    with pytest.raises(ValueError,match='matrix'):c.validate_acceptance(p)


def test_shared_capacity_does_not_double_count_consumed_formal_bytes():
    a=c.capacity_math(6_000_000_000,0,0,115_000_000_000,101_000_000_000,8_500_000_000,1_000_000_000,1_800_000_000)
    b=c.capacity_math(7_000_000_000,1_000_000_000,10_000_000_000,104_000_000_000,101_000_000_000,8_500_000_000,1_000_000_000,1_800_000_000)
    assert a['fits'] and b['fits']
    assert a['processing_actual_plus_reserved_bytes']==b['processing_actual_plus_reserved_bytes']
    assert a['free_bytes']-a['pending_all_categories_plus_headroom']==b['free_bytes']-b['pending_all_categories_plus_headroom']


@pytest.mark.parametrize('bad',[-1,1.5,True])
def test_invalid_capacity_refused(bad):
    with pytest.raises(ValueError):c.capacity_math(bad,0,0,100,50,10,0,0,0)


def test_foreign_disk_consumption_blocks_next_launch():
    r=c.capacity_math(1_000_000,0,0,5_000_000,20_000_000,1_000_000,0,0,0)
    assert not r['fits'] and not r['disk_fits']


def test_processing_cap_counts_actual_plus_remaining():
    r=c.capacity_math(19_000_000_000,0,0,100_000_000_000,1,2_000_000_000,0,0,0)
    assert not r['fits'] and not r['processing_fits']


def test_failure_or_log_envelope_exhaustion_refuses():
    with pytest.raises(ValueError,match='checkpoint'):c.capacity_math(10,0,1001,10000,1000,10,0,0,0)
    with pytest.raises(ValueError,match='log/diagnostic'):c.capacity_math(1001,1001,0,10000,1000,1000,0,0,0)


def test_base_interpreter_wrapper_redirect_keeps_exact_arguments(rows):
    row=rows[0];words=list(row['launch']['argv']);words[0]='C:/Python312/python.exe'
    s.verify_worker(words,row)


@pytest.mark.parametrize('flag,new',[('--seed','29'),('--gpu',c.GPUS[1]),('--method','dense'),('--output','E:/AI/MOTE-quantum/elsewhere')])
def test_pid_command_identity_mismatch_refused(rows,flag,new):
    row=rows[0];words=list(row['launch']['argv']);words[words.index(flag)+1]=new
    with pytest.raises(ValueError):s.verify_worker(words,row)


@pytest.mark.parametrize('records',[
    [{'pid':1,'exit_code':0}],
    [{'pid':1,'exit_code':0},{'pid':2,'exit_code':None}],
    [{'pid':1,'exit_code':0},{'pid':2,'exit_code':1}],
    [{'pid':1,'exit_code':0},{'pid':1,'exit_code':0}],
])
def test_pid_alone_or_failed_exit_is_not_completion(records):
    with pytest.raises(ValueError):s.verify_exits(records)


def test_windows_held_handles_observe_both_real_cpu_process_exits(tmp_path):
    script=tmp_path/'cpu_exit_fixture.py';script.write_text('import time\ntime.sleep(2)\n')
    p=subprocess.Popen([str(c.ROOT/'envs/qwen-native-py312-cu128/Scripts/python.exe'),str(script)],creationflags=subprocess.CREATE_NO_WINDOW)
    held=[]
    try:
        held.append(HeldProcess(p.pid));deadline=time.monotonic()+1.5
        while len(held)<2 and time.monotonic()<deadline:
            for child in psutil.Process(p.pid).children(recursive=True):
                if child.name().lower()=='python.exe':held.append(HeldProcess(child.pid));break
            time.sleep(.02)
        assert len(held)==2
        p.wait(timeout=10)
        for h in held:
            deadline=time.monotonic()+2
            while h.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert s.verify_exits([dict(**h.record,exit_code=h.poll()) for h in held])
    finally:
        p.wait(timeout=10)
        for h in held:h.close()


def test_status_permission_conflict_retries_without_losing_complete_file(tmp_path):
    p=tmp_path/'CONTROLLER.json';p.write_text('old');attempts=[]
    def replace(src,dest):
        attempts.append(str(src))
        if len(attempts)<3:raise PermissionError('reader sharing conflict')
        Path(src).replace(dest)
    atomic_json(p,{'phase':'running'},replace=replace,delay=0)
    assert c.read(p)=={'phase':'running'} and len(attempts)==3


@pytest.fixture(scope='module')
def seed17_initial_bindings(rows):
    fixtures=[]
    for row in rows:
        if row['seed']!=17:continue
        cohort='pro4000' if row['cohort']=='bf16' else 'v100b'
        run=c.ROOT/f'runs/production-acceptance-v3-{cohort}-20261031/pilot-{row["method"]}'
        ident=c.read(run/'RUN_CONFIG.json');ident.update(stage='formal',target_tokens=32768000,acceptance_sha256='fixture')
        # The completed FP16 pilots used V100 B; formal seed17 is assigned A.
        # This witness fixture checks the real pilot UUID without rewriting data.
        row=copy.deepcopy(row);row['gpu_uuid']=ident['gpu']
        if row['cohort']=='bf16':ref=c.ROOT/f'diagnostics/bf16-followup-v6r1-pro4000-20261031/initial-seed17-{row["method"]}-initial_full_model/RESULT.json'
        else:
            ctl=c.read(c.ROOT/'diagnostics/initial-full-model-v5-v100a-20261031/CONTROLLER.json')
            job=next(j for j in ctl['jobs'].values() if j['seed']==17 and j['method']==row['method'])
            ref=Path(job['output'])/'RESULT.json'
        initial=c.read(ref)['initialization']
        meta=None if row['method']=='dense' else c.read(Path(row['initialization']['common']['path']).parent/'COMMON_INITIALIZATION.json')
        fixtures.append((row,ident,c.read(run/'PARAMETERS.json'),c.read(run/'ROUTER_PLACEMENT.json'),initial,meta,'fixture'))
    return fixtures


def test_real_ten_pilot_initialization_witnesses_agree_with_diagnostic_references(seed17_initial_bindings):
    # Identity stage/target are fixture substitutions; existing pilot artifacts stay untouched.
    assert len(seed17_initial_bindings)==10
    for args in seed17_initial_bindings:
        result=c.initialization_checks(*args)
        assert result['passed'] and result['full_training_start_tensor_hash_directly_observed'] is False


@pytest.mark.parametrize('bad',['router','seed','common','precision','layers','acceptance','device'])
def test_initialization_drift_refused(seed17_initial_bindings,bad):
    row,ident,params,placement,initial,meta,h=copy.deepcopy(seed17_initial_bindings[0])
    if bad=='router':placement['after']='0'*64
    elif bad=='seed':params['adapter']['seed']=29
    elif bad=='common':meta['checkpoint_sha256']='0'*64
    elif bad=='precision':initial['native_precision']='torch.float16'
    elif bad=='layers':params['layers']=[20,21,22,23,24,25,26]
    elif bad=='acceptance':ident['acceptance_sha256']='different'
    elif bad=='device':ident['cuda_device']['uuid']=c.GPUS[1]
    with pytest.raises(ValueError):c.initialization_checks(row,ident,params,placement,initial,meta,h)


def synthetic_updates():
    updates=[];samples=[]
    for i in range(1,1001):
        updates.append(dict(tokens=i*32768,update_tokens=32768,ce=3.,grad_norm=.1,overflow_retries=0,
            optimizer_update=dict(updated=True,groups={'experts':dict(unscaled_grad_norm=.1,model_update_norm=0. if i==1000 else .01)})))
        samples.append(dict(tokens=i*32768,samples=[dict(valid_targets=1024)]*32))
    return updates,samples


def test_full_budget_and_terminal_zero_update():
    r=c.update_checks(*synthetic_updates());assert r['optimizer_updates']==1000 and r['parameter_changing_updates']==999


@pytest.mark.parametrize('bad',['missing','tokens','nonfinite','negative','terminal_change','first_frozen','target_count'])
def test_failed_or_mismatched_full_run_refused(bad):
    u,samples=synthetic_updates()
    if bad=='missing':u.pop()
    elif bad=='tokens':u[4]['tokens']=1
    elif bad=='nonfinite':u[5]['ce']=float('nan')
    elif bad=='negative':u[5]['optimizer_update']['groups']['experts']['model_update_norm']=-.1
    elif bad=='terminal_change':u[-1]['optimizer_update']['groups']['experts']['model_update_norm']=.1
    elif bad=='first_frozen':u[0]['optimizer_update']['groups']['experts']['model_update_norm']=0.
    elif bad=='target_count':samples[4]['samples']=[dict(valid_targets=1)]
    with pytest.raises(ValueError):c.update_checks(u,samples)


def test_extra_zero_change_is_preserved_as_result_not_filtered():
    u,samples=synthetic_updates();u[500]['optimizer_update']['groups']['experts']['model_update_norm']=0.
    r=c.update_checks(u,samples);assert r['zero_change_updates']==[501,1000] and not r['matches_expected_999_parameter_changes']


def test_no_diagnostic_receipt_keeps_barrier_waiting(rows,tmp_path,monkeypatch):
    monkeypatch.setattr(c,'DIAGNOSTICS',tmp_path)
    assert s.verify_diagnostic_barrier(rows[0],tmp_path/'unused') is None


def test_incomplete_diagnostic_receipt_cannot_release_next_training(rows,tmp_path,monkeypatch):
    monkeypatch.setattr(c,'DIAGNOSTICS',tmp_path);out=tmp_path/rows[0]['name'];out.mkdir()
    audit=tmp_path/'training_audit.json';audit.write_text('{}')
    (out/'DIAGNOSTICS_COMPLETE.json').write_text(json.dumps(dict(schema='qwen-formal-diagnostics-v12',passed=True,name=rows[0]['name'],training_audit_sha256=c.sha(audit),measurements=[])))
    with pytest.raises(ValueError,match='Incomplete'):s.verify_diagnostic_barrier(rows[0],audit)


@pytest.mark.parametrize('failure',['none','nonzero_exit','observer_command_failure'])
def test_full_supervisor_observes_or_drains_real_cpu_workers(rows,tmp_path,monkeypatch,failure):
    row=copy.deepcopy(rows[0]);row['output']=str(tmp_path/'never_training')
    script=tmp_path/'cpu_only_process.py'
    script.write_text('import time,sys\ntime.sleep(3)\nsys.exit('+('7' if failure=='nonzero_exit' else '0')+')\n')
    row['launch']['argv']=[str(c.ROOT/'envs/qwen-native-py312-cu128/Scripts/python.exe'),'-u',str(script),'--output',row['output']]
    supervisor=s.Supervisor([row],{}, {},{'sha256':'fixture'})
    if failure=='observer_command_failure':
        def reject(*args):raise ValueError('Injected observation failure')
        monkeypatch.setattr(s,'verify_worker',reject)
    p=subprocess.Popen(row['launch']['argv'],creationflags=subprocess.CREATE_NO_WINDOW)
    if failure=='none':
        assert s.verify_exits(supervisor.observe_training(row,p))
    else:
        with pytest.raises(ValueError):supervisor.observe_training(row,p)
        assert supervisor.drain.is_set()
        assert supervisor.state['jobs'][row['name']]['phase']=='observer_failure_workers_exited'
    assert p.poll()==(7 if failure=='nonzero_exit' else 0)
    assert not s.matching_run_processes(row)
    assert not torch.cuda.is_initialized()


def test_all15_real_fp16_initial_references_are_hash_bound(rows):
    selected=[r for r in rows if r['cohort']=='fp16']
    assert len(selected)==15
    for row in selected:
        initial,binding=c.initial_reference(row)
        assert initial['native_precision']=='torch.float16' and initial['zero_training_updates']
        assert binding['sha256'] and len(binding['sha256'])==64


@pytest.fixture
def barrier_from_real_worker_schemas(rows,tmp_path,monkeypatch):
    """Copy real diagnostic JSON schemas, substitute only formal identities.

    This exercises the handoff contract, not a new raw-array audit; original
    experiment files remain untouched and fixtures are not acceptance evidence.
    """
    row=next(r for r in rows if r['cohort']=='fp16' and r['seed']==17 and r['method']=='mlp')
    monkeypatch.setattr(c,'DIAGNOSTICS',tmp_path/'diagnostics')
    output=c.DIAGNOSTICS/row['name'];output.mkdir(parents=True)
    audit_path=tmp_path/'training_audit.json'
    retained=[dict(tokens=t,checkpoint=str(tmp_path/f'checkpoint-{t}'),payload=dict(sha256=str(t).zfill(64))) for t in (3276800,16384000,32768000)]
    atomic_json(audit_path,dict(retained=retained))
    measurements=[]
    for token in (0,3276800,16384000,32768000):
        for kind in ('full_model','local_precision') if token==0 else ('timing','local_precision','full_model'):
            original=c.ROOT/f'diagnostics/pilot-endpoint-v5-v100a-20261031/mlp-{kind}/RESULT.json'
            if token==0 and kind=='full_model':original=c.ROOT/'diagnostics/initial-full-v5-validation-fp16-seed17-mlp-20261031/RESULT.json'
            value=c.read(original)
            if kind=='local_precision':
                value['binding']['tokens']=token
                if token:
                    checkpoint=next(r for r in retained if r['tokens']==token)
                    value['binding'].update(checkpoint=checkpoint['checkpoint'],payload_sha256=checkpoint['payload']['sha256'])
                else:
                    value['binding'].pop('checkpoint');value['binding'].pop('payload_sha256');value['binding'].pop('marker_sha256')
            else:
                value['checkpoint_tokens']=token
                if token:
                    checkpoint=next(r for r in retained if r['tokens']==token)
                    value.update(checkpoint=checkpoint['checkpoint'],checkpoint_payload_sha256=checkpoint['payload']['sha256'])
            rp=output/f'{token}-{kind}-RESULT.json';ap=output/f'{token}-{kind}-AUDIT.json'
            atomic_json(rp,value);atomic_json(ap,dict(passed=True,result_sha256=c.sha(rp)))
            measurements.append(dict(tokens=token,kind=kind,result=str(rp),audit=str(ap),result_sha256=c.sha(rp),audit_sha256=c.sha(ap)))
    receipt=output/'DIAGNOSTICS_COMPLETE.json'
    value=dict(schema='qwen-formal-diagnostics-v12',passed=True,name=row['name'],training_audit_sha256=c.sha(audit_path),measurements=measurements)
    atomic_json(receipt,value)
    return row,audit_path,receipt,value


def test_real_diagnostic_schemas_satisfy_complete_handoff(barrier_from_real_worker_schemas):
    row,audit_path,receipt,_=barrier_from_real_worker_schemas
    assert s.verify_diagnostic_barrier(row,audit_path)['sha256']==c.sha(receipt)


@pytest.mark.parametrize('bad',['precision','device','seed','protocol','payload','path','incomplete_phase','initial_trained','unbound_audit','missing_state'])
def test_cross_run_or_incomplete_diagnostics_cannot_release_queue(barrier_from_real_worker_schemas,bad):
    row,audit_path,receipt,value=barrier_from_real_worker_schemas
    cell=value['measurements'][0 if bad=='initial_trained' else -1]
    result=c.read(cell['result'])
    if bad=='precision':result['native_precision']='torch.bfloat16'
    elif bad=='device':result['physical_device']['uuid']=c.PRO
    elif bad=='seed':result['seed']=29
    elif bad=='protocol':result['protocol_sha256']='0'*64
    elif bad=='payload':result['checkpoint_payload_sha256']='0'*64
    elif bad=='path':result['checkpoint']='E:/AI/MOTE-quantum/unrelated'
    elif bad=='incomplete_phase':result['phase']='running'
    elif bad=='initial_trained':result['checkpoint']='E:/AI/MOTE-quantum/trained'
    elif bad=='missing_state':value['measurements'].pop()
    atomic_json(cell['result'],result);cell['result_sha256']=c.sha(cell['result'])
    atomic_json(cell['audit'],dict(passed=True,result_sha256='0'*64 if bad=='unbound_audit' else cell['result_sha256']))
    cell['audit_sha256']=c.sha(cell['audit']);atomic_json(receipt,value)
    with pytest.raises(ValueError):s.verify_diagnostic_barrier(row,audit_path)


@pytest.fixture
def formal_audit_fixture(rows,tmp_path):
    """CPU-only synthetic final run, using the real frozen sample stream and retention API."""
    from fitmotn.experiments.qwen_joint.retention import Retention,FORMAT
    row=copy.deepcopy(next(r for r in rows if r['cohort']=='fp16' and r['seed']==17 and r['method']=='mlp'))
    output=tmp_path/'formal-fixture';output.mkdir();row['output']=str(output)
    pilot=c.ROOT/'runs/production-acceptance-v3-v100b-20261031/pilot-mlp'
    identity=c.read(pilot/'RUN_CONFIG.json');identity.update(stage='formal',target_tokens=32768000,acceptance_sha256='fixture',gpu=row['gpu_uuid'])
    identity['cuda_device']['uuid']=row['gpu_uuid'].removeprefix('GPU-')
    atomic_json(output/'RUN_CONFIG.json',identity)
    for n in ('PARAMETERS.json','ROUTER_PLACEMENT.json','SOURCE_EXECUTION.json'):(output/n).write_bytes((pilot/n).read_bytes())
    frozen=c.ROOT/'initialization/seed-release-v2-20261031/train-seed17-samples.jsonl'
    (output/'samples.jsonl').write_bytes(frozen.read_text().replace('\n','\r\n').encode())
    updates,_=synthetic_updates();updates[-1]['cursor']=row['data']['expected_final_cursor']
    for u in updates:
        u['optimizer_update']['groups']={k:copy.deepcopy(u['optimizer_update']['groups']['experts']) for k in ('experts','residual_tail_or_dense','router')}
    (output/'updates.jsonl').write_text(''.join(json.dumps(u)+'\n' for u in updates))
    retention=Retention.create(output,output/'checkpoints',identity,32768000)
    for i,t in enumerate(row['evaluation']['save_and_dev_tokens'],1):
        dev=c.read(pilot/'dev-0001048576.json');dev['ce']=2.0 if i==3 else 3.0+i/100
        atomic_json(output/f'dev-{t:010d}.json',dev)
        cp=retention.path/f'checkpoint-{t:010d}';cp.mkdir();(cp/'training.pt').write_bytes(f'CPU fixture, no model weights, token {t}'.encode())
        atomic_json(cp/'complete.json',dict(format=FORMAT,identity=identity,tokens=t,files={'training.pt':dict(sha256=c.sha(cp/'training.pt'),bytes=(cp/'training.pt').stat().st_size)}))
        retention.register(cp.name,dev['ce']);retention.prune()
    result=c.read(pilot/'RESULT.json');result.update(stage='formal',tokens=32768000,target=32768000,start_tokens=0,identity=identity,
        samples_sha256=c.sha(output/'samples.jsonl'),checkpoint=str(cp),internal_dev=dev)
    atomic_json(output/'RESULT.json',result)
    return row,output


def test_end_to_end_formal_audit_on_cpu_retention_fixture(formal_audit_fixture):
    row,output=formal_audit_fixture
    result=c.audit_training(row,'fixture')
    assert result['passed'] and result['counts']['optimizer_updates']==1000
    assert result['best_dev_tokens']==9830400 and len(result['retained'])==4
    assert result['samples_sha256']==row['data']['formal_samples_windows_crlf_sha256']


@pytest.mark.parametrize('bad',['payload','registered_marker','extra_checkpoint_file','extra_checkpoint_directory','retention_owner','sample_log','dev','cursor','missing_router'])
def test_formal_audit_preserves_and_rejects_changed_artifacts(formal_audit_fixture,bad):
    row,output=formal_audit_fixture
    cp=output/'checkpoints/checkpoint-0032768000'
    if bad=='payload':(cp/'training.pt').write_bytes(b'corrupted')
    elif bad=='registered_marker':
        marker=c.read(cp/'complete.json');marker['unexpected_change']=True;atomic_json(cp/'complete.json',marker)
    elif bad=='extra_checkpoint_file':(cp/'unexpected.txt').write_text('preserve')
    elif bad=='extra_checkpoint_directory':(cp.parent/'unexpected').mkdir()
    elif bad=='retention_owner':
        owner=c.read(cp.parent/'RETENTION_OWNER.json');owner['run']=str(output);atomic_json(cp.parent/'RETENTION_OWNER.json',owner)
    elif bad=='sample_log':
        with (output/'samples.jsonl').open('ab') as f:f.write(b'\r\n')
    elif bad=='dev':
        p=output/'dev-0003276800.json';value=c.read(p);value['ce']=float('nan');atomic_json(p,value)
    elif bad=='cursor':
        p=output/'updates.jsonl';u=[json.loads(line) for line in p.read_text().splitlines()];u[-1]['cursor']['offset']+=1;p.write_text(''.join(json.dumps(v)+'\n' for v in u))
    elif bad=='missing_router':
        p=output/'updates.jsonl';u=[json.loads(line) for line in p.read_text().splitlines()];u[0]['optimizer_update']['groups'].pop('router');p.write_text(''.join(json.dumps(v)+'\n' for v in u))
    with pytest.raises(ValueError):c.audit_training(row,'fixture')
    assert cp.exists() and (cp/'training.pt').exists()
