import torch
from fitmotn.experiments.qwen_joint.model import AdapterConfig,JointProjection
from risk_probe_candidate_v2 import replace_selected_output,target_logprobs


def test_controlled_replacement_matches_independent_dense_mixture_and_retains_tail():
    torch.manual_seed(29)
    module=JointProjection(torch.nn.Linear(128,192,bias=False),'mlp',AdapterConfig(residual_rank=4,tail_rank=4),100)
    x=torch.randn(1,11,128)
    positions=torch.tensor([0,4,9])
    captured={}
    def router_hook(module,inputs,output):captured['route']=output
    def expert_hook(eid):
        def hook(module,inputs,output):captured[eid]=output.clone()
        return hook
    handles=[module.router.register_forward_hook(router_hook)]
    handles.extend(expert.register_forward_hook(expert_hook(eid)) for eid,expert in enumerate(module.experts))
    with torch.no_grad():
        baseline=module(x)
        for h in handles:h.remove()
        actual=captured['route'];original_top1=torch.empty(len(positions),module.main_out)
        for i,pos in enumerate(positions):
            eid=int(actual.indices[pos,0]);token,_=torch.where(actual.indices==eid)
            original_top1[i]=captured[eid][torch.where(token==pos)[0][0]]
        for farthest in (False,True):
            changed,record=replace_selected_output(module,x,baseline,positions,actual,original_top1,farthest)
            assert torch.equal(torch.tensor(record['weights']),actual.weights[positions])
            assert torch.equal(record['old_output'],original_top1)
            route=module.router(x[0]);experts=torch.stack([e(x[0]) for e in module.experts])
            expected=baseline.clone()
            for row,pos in enumerate(positions.tolist()):
                original=int(route.indices[pos,0]);replacement=record['replacement'][row]
                candidates=[i for i in range(16) if i not in route.indices[pos].tolist()]
                distances=[float((experts[i,pos]-experts[original,pos]).square().sum()) for i in candidates]
                expected_candidate=candidates[distances.index(max(distances) if farthest else min(distances))]
                assert replacement==expected_candidate
                mixture=torch.zeros(module.main_out)
                for slot in range(4):
                    eid=replacement if slot==0 else route.indices[pos,slot]
                    mixture+=route.weights[pos,slot]*experts[eid,pos]
                expected[0,pos,:module.main_out]=mixture+module.residual(x[0,pos:pos+1])[0]
            torch.testing.assert_close(changed,expected,atol=2e-7,rtol=2e-6)
            assert torch.equal(changed[0,[1,2,3,5,6,7,8,10]],baseline[0,[1,2,3,5,6,7,8,10]])
            assert torch.equal(changed[:,:,128:],baseline[:,:,128:])


def test_logprob_chunking_matches_direct_explicit_targets():
    torch.manual_seed(43)
    class Model:
        lm_head=torch.nn.Linear(7,31,bias=False)
    model=Model();hidden=torch.randn(2,11,7);labels=torch.randint(0,31,(2,11));labels[:,0]=-100
    with torch.no_grad():
        actual=torch.from_numpy(target_logprobs(model,hidden,labels,chunk=3))
        logits=model.lm_head(hidden).flatten(0,1)
        expected=-torch.nn.functional.cross_entropy(logits,labels.flatten(),ignore_index=-100,reduction='none')
    torch.testing.assert_close(actual,expected)
    assert actual[0]==0 and actual[11]==0
