import copy
import pytest
import formal_health_boundaries_v18 as h

def fixture():
    name='fp16-seed17-mlp'
    c=h.read(h.ROOT/'runtime/formal-supervision-v12-20261031/CONTROLLER.json')
    j=c['jobs'][name]
    d=h.ROOT/'runs/formal-matrix-v11'/name
    return name,copy.deepcopy(j),h.read(d/'RESULT.json'),h.read(j['training_audit']['path']),h.read(d/'RUN_CONFIG.json')

def test_completed_real_run():
    args=fixture()
    assert h.terminal_contract(*args)==[]

@pytest.mark.parametrize('fault', ['nonzero_exit','missing_exit','wrong_tokens','stopped','missing_dev','changed_samples','reserved_test','wrong_acceptance','missing_checkpoint'])
def test_incomplete_or_invalid_completion_rejected(fault):
    name,j,r,a,c=fixture()
    if fault=='nonzero_exit': j['exit_codes'][0]['exit_code']=1
    elif fault=='missing_exit': j['exit_codes']=[]
    elif fault=='wrong_tokens': r['tokens']-=1
    elif fault=='stopped': r['phase']='stopped'
    elif fault=='missing_dev': a['internal_dev'].pop(0)
    elif fault=='changed_samples': r['samples_sha256']='invalid'
    elif fault=='reserved_test': a['reserved_test_accessed']=True
    elif fault=='wrong_acceptance': c['acceptance_sha256']='invalid'
    elif fault=='missing_checkpoint': a['retained']=[]
    assert h.terminal_contract(name,j,r,a,c)

def test_real_retained_metadata_and_exit_observation():
    name,j,*_=fixture()
    errors=[]
    result=h.finished_training(name,j,lambda ok,msg: errors.append(msg) if not ok else None)
    assert not errors
    assert result['observed_tokens']==32768000
    assert not result['actual_processes']
    assert all(not x['payload_hashes_recomputed_this_observation'] for x in result['checkpoint_observations'])
