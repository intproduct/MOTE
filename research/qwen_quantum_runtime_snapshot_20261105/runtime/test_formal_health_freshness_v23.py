import copy,json
from pathlib import Path
import pytest
from formal_health_freshness_v23 import active_evaluation_evidence
FIXTURE=json.loads(Path(__file__).with_name('formal-health-v23-boundary-fixture.json').read_text(encoding='utf-8'))
BASE=FIXTURE['input']


def test_real_append_after_pass_clock():
    x=copy.deepcopy(BASE)
    x['observed_epoch']=FIXTURE['original_pass_clock']
    assert active_evaluation_evidence(**x) is None
    x['observed_epoch']=BASE['observed_epoch']
    assert active_evaluation_evidence(**x)['tokens']==29491200


def test_repeated_observation_cannot_extend_freshness():
    x=copy.deepcopy(BASE)
    x['observed_epoch']=x['thermal_rows'][-1]['time']+59.999
    assert active_evaluation_evidence(**x)
    x['observed_epoch']+=.0011
    assert active_evaluation_evidence(**x) is None


@pytest.mark.parametrize('case',['stopped','wrong_target','zero_tokens','over_budget','non_boundary','noninteger_tokens','no_checkpoint','dev_present','empty_log','single_guard','later_update','reversed_times','stale_guard','future_guard','nan_clock','infinite_guard','nan_status','thermal_pause','thermal_event','before_status'])
def test_rejects_invalid_active_evaluation(case):
    x=copy.deepcopy(BASE)
    if case=='stopped':x['status']['phase']='stopped'
    elif case=='wrong_target':x['status']['target']=1000
    elif case=='zero_tokens':x['status']['tokens']=0
    elif case=='over_budget':x['status']['tokens']=36044800
    elif case=='non_boundary':x['status']['tokens']+=32768
    elif case=='noninteger_tokens':x['status']['tokens']=float(x['status']['tokens'])
    elif case=='no_checkpoint':x['checkpoint_exists']=False
    elif case=='dev_present':x['dev_exists']=True
    elif case=='empty_log':x['thermal_rows']=[]
    elif case=='single_guard':x['thermal_rows']=x['thermal_rows'][-1:]
    elif case=='later_update':x['thermal_rows'].append(dict(x['thermal_rows'][-1],stage='update'))
    elif case=='reversed_times':x['thermal_rows'][-1]['time']=x['thermal_rows'][-2]['time']
    elif case=='stale_guard':x['observed_epoch']=x['thermal_rows'][-1]['time']+60
    elif case=='future_guard':x['thermal_rows'][-1]['time']=x['observed_epoch']+1
    elif case=='nan_clock':x['observed_epoch']=float('nan')
    elif case=='infinite_guard':x['thermal_rows'][-1]['time']=float('inf')
    elif case=='nan_status':x['status']['time']=float('nan')
    elif case=='thermal_pause':x['thermal_rows'][-1]['wait_seconds']=1
    elif case=='thermal_event':x['thermal_rows'][-1]['event']='thermal_pause'
    elif case=='before_status':x['status']['time']=x['thermal_rows'][-2]['time']
    assert active_evaluation_evidence(**x) is None
