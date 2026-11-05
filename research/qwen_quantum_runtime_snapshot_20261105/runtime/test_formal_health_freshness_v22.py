import copy,json
from pathlib import Path
import pytest
from formal_health_freshness_v22 import post_evaluation_evidence
BASE=json.loads(Path(__file__).with_name('formal-health-v22-boundary-fixture.json').read_text(encoding='utf-8'))['input']

def test_real_recorded_boundary():
    answer=post_evaluation_evidence(**copy.deepcopy(BASE))
    assert answer and answer['tokens']==6553600 and answer['dev_age_seconds']<60

@pytest.mark.parametrize('case', ['expired','future_dev','stopped','wrong_target','non_boundary','wrong_checkpoint','wrong_cursor','wrong_acceptance','wrong_method','wrong_seed','wrong_gpu','wrong_sample','wrong_count','wrong_eval_seed','nan_ce','wrong_nll','missing_eval','reversed_eval','thermal_pause','thermal_event','wrong_guard_stage','stale_guard','future_guard','missing_checkpoint'])
def test_rejects_invalid_boundary(case):
    x=copy.deepcopy(BASE)
    if case=='expired':x['now']=x['dev_mtime']+60
    elif case=='future_dev':x['dev_mtime']=x['now']+1
    elif case=='stopped':x['status']['phase']='stopped'
    elif case=='wrong_target':x['status']['target']=1000
    elif case=='non_boundary':x['status']['tokens']+=32768
    elif case=='wrong_checkpoint':x['checkpoint']['tokens']-=3276800
    elif case=='wrong_cursor':x['checkpoint']['cursor']['offset']+=1
    elif case=='wrong_acceptance':x['checkpoint']['identity']['acceptance_sha256']='bad'
    elif case in ('wrong_method','wrong_seed','wrong_gpu'):x['checkpoint']['identity'][case[6:]]='bad'
    elif case=='wrong_sample':x['dev']['sample_hash']='bad'
    elif case=='wrong_count':x['dev']['valid_targets']-=1
    elif case=='wrong_eval_seed':x['dev']['evaluation_order_seed']=29
    elif case=='nan_ce':x['dev']['ce']=float('nan')
    elif case=='wrong_nll':x['dev']['nll_sum']+=1000
    elif case=='missing_eval':x['thermal_rows']=x['thermal_rows'][-1:]
    elif case=='reversed_eval':x['thermal_rows'][-2]['time']=x['thermal_rows'][-3]['time']
    elif case=='thermal_pause':x['thermal_rows'][-1]['wait_seconds']=1
    elif case=='thermal_event':x['thermal_rows'][-1]['event']='thermal_pause'
    elif case=='wrong_guard_stage':x['thermal_rows'][-1]['stage']='evaluation'
    elif case=='stale_guard':x['thermal_rows'][-1]['time']=x['now']-61
    elif case=='future_guard':x['thermal_rows'][-1]['time']=x['now']+1
    elif case=='missing_checkpoint':x['checkpoint']={}
    assert post_evaluation_evidence(**x) is None
