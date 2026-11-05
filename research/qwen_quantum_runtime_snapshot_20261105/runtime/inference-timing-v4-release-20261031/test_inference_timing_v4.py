import math
from types import SimpleNamespace

import pytest
import torch

from inference_timing_core_v4 import (assert_owned_observation, assert_plain_evaluation,
    positive_integer, serial_measure, summarize)


def test_boundaries_exclude_observers_and_warmup():
    current=[0.]; events=[]; durations=iter([3.,5.,1.,2.,4.]); saved=[]
    def clock(): events.append('clock'); return current[0]
    def sync(): events.append('sync')
    def forward(): events.append('forward'); current[0] += next(durations)
    def before(phase,index):
        events.append('before'); current[0] += 100.; return (phase,index)
    def after(row):
        events.append('after'); current[0] += 1000.; saved.append(dict(row)); return 'observed'
    rows=serial_measure(forward,sync,clock,before,after,warmup=2,repeats=3)
    assert [row['seconds'] for row in rows] == [3.,5.,1.,2.,4.]
    assert events == ['before','sync','clock','forward','sync','clock','after']*5
    assert len(saved)==5 and all(row['observation_after']=='observed' for row in rows)
    measured=[row['seconds'] for row in rows if row['phase']=='measured']
    result=summarize(measured,2048)
    assert result['total_input_tokens']==6144 and result['total_measured_seconds']==7
    assert result['input_tokens_per_second']==6144/7
    assert result['input_tokens_per_second'] != sum(2048/x for x in measured)/3


@pytest.mark.parametrize('value',[0,-1,True,1.5,math.inf])
def test_invalid_repeat_counts(value):
    with pytest.raises(ValueError): positive_integer(value,'repeats')


@pytest.mark.parametrize('seconds',[[],[0.],[math.nan],[math.inf],[-1.]])
def test_invalid_measured_durations(seconds):
    with pytest.raises(ValueError): summarize(seconds,2048)


def test_observation_failure_stops_before_next_model_call():
    calls=[]; ticks=iter([0.,1.])
    def after(row):
        calls.append('persist_failure')
        raise RuntimeError('foreign job observed')
    with pytest.raises(RuntimeError,match='foreign'):
        serial_measure(lambda:calls.append('forward'),lambda:None,lambda:next(ticks),
            lambda phase,index:None,after,warmup=1,repeats=10)
    assert calls==['forward','persist_failure']


def tiny_model():
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,5)).eval()
    model.config=SimpleNamespace(use_cache=False)
    model.is_gradient_checkpointing=False
    return model


def test_real_cpu_model_forward_and_hook_refusal():
    model=tiny_model(); x=torch.tensor([[1.,2.,3.]])
    assert_plain_evaluation(model)
    with torch.inference_mode(): assert model(x).shape==(1,5)
    handle=model[1].register_forward_hook(lambda *args:None)
    try:
        with pytest.raises(ValueError,match='Instrumented'): assert_plain_evaluation(model)
    finally: handle.remove()
    assert_plain_evaluation(model)


@pytest.mark.parametrize('mutation,message',[
    (lambda m:m[0].train(),'Training'),
    (lambda m:setattr(m[0],'capture_metrics',True),'metrics'),
    (lambda m:setattr(m,'is_gradient_checkpointing',True),'checkpointing'),
    (lambda m:setattr(m.config,'use_cache',True),'KV cache')])
def test_instrumentation_and_mode_refusals(mutation,message):
    model=tiny_model(); mutation(model)
    with pytest.raises(ValueError,match=message): assert_plain_evaluation(model)


def observation():
    snap=dict(uuid='GPU-test',compute_apps=[['GPU-test','123','python.exe']],display_apps=[['GPU-test','77','desktop.exe']])
    lease=dict(uuid='GPU-test',pid=123,token='nonce')
    return snap,lease,dict(lease)


def test_owned_process_allows_recorded_desktop_sharing():
    snap,lease,lock=observation()
    value=assert_owned_observation(snap,lease,lock,123)
    assert not value['foreign_compute_observed'] and value['desktop_processes_observed']


@pytest.mark.parametrize('change',[
    lambda s,l,k:s['compute_apps'].append(['GPU-test','456','python.exe']),
    lambda s,l,k:s['compute_apps'].append(['GPU-test','unknown','process']),
    lambda s,l,k:k.update(token='other'),
    lambda s,l,k:k.update(pid=456),
    lambda s,l,k:s.update(uuid='GPU-other')])
def test_foreign_or_unknown_process_and_ownership_refused(change):
    snap,lease,lock=observation(); change(snap,lease,lock)
    with pytest.raises(RuntimeError): assert_owned_observation(snap,lease,lock,123)
