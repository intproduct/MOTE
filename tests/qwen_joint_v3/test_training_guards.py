import copy
import json
import random
import numpy as np
import pytest
import torch
from fitmotn.experiments.qwen_joint import train, task, validation
from fitmotn.experiments.qwen_joint.reproducibility import tree_digest
from fitmotn.experiments.qwen_joint.state import rng_state, digest


@pytest.mark.parametrize('fails',[False,True])
@pytest.mark.parametrize('initial_mode',[False,True])
def test_evaluation_preserves_rng_and_model_mode_on_success_or_failure(monkeypatch,fails,initial_mode):
    model=torch.nn.Linear(2,2)
    model.train(initial_mode)
    class Stream:
        def __init__(self,*args):assert args[1:] == ('internal_dev',17)
        def next(self,n):return np.array([1,2]),np.array([1,2]),dict(valid_targets=2,index=9)
    def loss(model,ids,labels):
        assert not model.training
        random.random();np.random.rand();torch.rand(3)
        return torch.tensor(float('nan') if fails else 3.),2,torch.tensor(0.)
    monkeypatch.setattr(train,'TargetWindows',Stream)
    monkeypatch.setattr(task,'next_target_loss',loss)
    before=tree_digest(rng_state())
    if fails:
        with pytest.raises(FloatingPointError):train.evaluate(model,'fixture',43,torch.float16,target=4,device='cpu')
    else:
        result=train.evaluate(model,'fixture',43,torch.float16,target=4,device='cpu')
        assert result['valid_targets']==4 and result['ce']==1.5 and result['evaluation_order_seed']==17
    assert tree_digest(rng_state())==before and model.training==initial_mode


@pytest.fixture
def formal(tmp_path):
    evidence=tmp_path/'acceptance-evidence.json';evidence.write_text('{"passed":true}')
    acceptance=dict(schema='qwen-integration-acceptance-v3',formal_approved=True,code={'a':'b'},
        data_sha256='data',model_manifest_sha256='model',config_sha256='config',methods=list(validation.METHODS),
        seeds=[17,29,43],common_sha256_by_seed={'17':'seed17','29':'seed29','43':'seed43'},
        accepted_gpu_uuids=['test-gpu'],formal_tokens=32768000,
        resources=dict(approved=True,new_data_cache_reserved_total_bytes=19999999999,maximum_new_run_peak_bytes=1),
        required_evidence=[dict(path=str(evidence),sha256=digest(evidence))])
    return acceptance,tmp_path


def validate(acceptance,root):
    return validation.formal_budget(acceptance,{'a':'b'},'data','model','config','mlp',17,'test-gpu','seed17',root)


def test_formal_acceptance_requires_bound_complete_matrix(formal):
    acceptance,root=formal
    assert validate(acceptance,root)==32768000


@pytest.mark.parametrize('field,value',[
    ('formal_approved',False),('data_sha256','other'),('model_manifest_sha256','other'),
    ('config_sha256','other'),('code',{}),('methods',['mlp']),('seeds',[17]),
    ('formal_tokens',32768001),('accepted_gpu_uuids',[]),('required_evidence',[]),
    ('common_sha256_by_seed',{'17':'wrong'}),
    ('resources',dict(approved=True,new_data_cache_reserved_total_bytes=20000000001,maximum_new_run_peak_bytes=1)),
    ('resources',dict(approved=True,new_data_cache_reserved_total_bytes=1,maximum_new_run_peak_bytes=10**20)),
])
def test_formal_rejects_incomplete_or_changed_release(formal,field,value):
    acceptance,root=formal
    acceptance[field]=value
    with pytest.raises(ValueError):validate(acceptance,root)


def test_formal_rejects_changed_acceptance_evidence(formal):
    acceptance,root=formal
    (root/'acceptance-evidence.json').write_text('{"passed":false}')
    with pytest.raises(ValueError,match='evidence changed'):validate(acceptance,root)
