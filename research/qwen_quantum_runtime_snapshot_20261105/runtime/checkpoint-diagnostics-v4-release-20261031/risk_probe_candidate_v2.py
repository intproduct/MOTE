"""Evaluation-only controlled expert replacement and aligned full-model probes."""
import numpy as np
import torch
from fitmotn.experiments.qwen_joint.routers import hard_topk
from mechanism_metrics_candidate_v1 import replacement_experts


def replace_selected_output(module,inputs,output,positions,original_route,original_top1,farthest=False):
    """Replace top-1's contribution at selected positions, retaining its weight."""
    assert inputs.ndim==3 and inputs.shape[0]==1 and output.dtype==torch.float32
    flat=inputs[0]
    positions=positions.to(device=inputs.device,dtype=torch.long)
    x=flat.index_select(0,positions)
    selected=original_route.indices.index_select(0,positions)
    weights=original_route.weights.index_select(0,positions)
    values=torch.stack([e(x[:,:module.main_in]) for e in module.experts])
    distance=(values.transpose(0,1)-original_top1[:,None]).square().sum(-1)
    distance.scatter_(1,selected,-torch.inf if farthest else torch.inf)
    chosen=distance.argmax(-1) if farthest else distance.argmin(-1)
    token=torch.arange(len(x),device=x.device)
    old=selected[:,0]
    delta=weights[:,0,None]*(values[chosen,token]-original_top1)
    replaced=output.clone()
    replaced[0,positions,:module.main_out]+=delta
    unchanged=torch.ones(len(flat),device=x.device,dtype=torch.bool);unchanged[positions]=False
    assert torch.equal(replaced[0,unchanged],output[0,unchanged])
    assert torch.equal(replaced[:,:,module.main_out:],output[:,:,module.main_out:])
    assert not (chosen[:,None]==selected).any()
    return replaced,dict(positions=positions.cpu().tolist(),selected=selected.cpu().tolist(),
        weights=weights.cpu().tolist(),replaced=old.cpu().tolist(),replacement=chosen.cpu().tolist(),
        original_contribution_source='actual_sparse_router_and_expert_forward_hooks',
        numerical_update='original_projection_output_plus_weight_times_new_minus_actual_old_expert',
        recomputed_old_output_max_abs_difference=float((values[old,token]-original_top1).abs().max()),
        delta_rms=delta.double().square().mean(-1).sqrt().cpu().tolist(),
        selected_input=x.cpu().clone(),old_output=original_top1.cpu().clone(),
        new_output=values[chosen,token].cpu().clone(),delta=delta.cpu().clone())


def target_logprobs(model,hidden,labels,chunk=64):
    hidden=hidden.flatten(0,1);labels=labels.flatten()
    values=[]
    for start in range(0,len(labels),chunk):
        logits=model.lm_head(hidden[start:start+chunk]).float()
        y=labels[start:start+chunk]
        loss=torch.nn.functional.cross_entropy(logits,y,ignore_index=-100,reduction='none')
        values.append(-loss)
    return torch.cat(values).float().cpu().numpy()


def aligned_probe(model,arrays,projection_names,variant,check,replace_projection=None,farthest=False):
    """Return complete aligned logprobs and top-k IDs, with bounded logit memory."""
    captures={name:[] for name in projection_names if hasattr(model.get_submodule(name),'router')}
    records=[];handles=[];current_window=None;actual={}
    def capture(name):
        def hook(module,inputs,output):
            captures[name].append(output.indices.detach().cpu().clone())
            if name==replace_projection:
                actual['route']=output
                actual['positions']=torch.from_numpy(arrays[f'positions_{current_window}']).to(output.weights.device)
                actual['old']=output.weights.new_empty(len(actual['positions']),model.get_submodule(name).main_out)
                actual['written']=torch.zeros(len(actual['positions']),device=output.weights.device,dtype=torch.bool)
        return hook
    def capture_old(eid):
        def hook(module,inputs,output):
            # This hook also sees later diagnostic expert calls; capture only the
            # original sparse forward, before the projection's post-forward hook.
            if actual.get('diagnostic',False):return
            route=actual['route'];positions=actual['positions']
            token,slot=torch.where(route.indices==eid)
            mask=route.indices[positions,0]==eid
            if mask.any():
                at=torch.searchsorted(token,positions[mask])
                assert torch.equal(token[at],positions[mask]) and output.shape[0]==len(token)
                actual['old'][mask]=output[at]
                actual['written'][mask]=True
        return hook
    def roundtrip(module,inputs):
        x=inputs[0]
        if variant=='input_bfloat16_roundtrip':
            # Explicit CPU conversion on V100 is quantization, not native BF16 compute.
            changed=x.cpu().to(torch.bfloat16).float().to(x.device)
        else:changed=x.to(torch.float16).float()
        return (changed,*inputs[1:])
    def replacement(module,inputs,output):
        assert actual['written'].all()
        actual['diagnostic']=True
        changed,record=replace_selected_output(module,inputs[0],output,
            torch.from_numpy(arrays[f'positions_{current_window}']),actual['route'],actual['old'],farthest)
        actual['diagnostic']=False
        record['window_index']=current_window
        records.append(record)
        return changed
    try:
        for name in captures:handles.append(model.get_submodule(name).router.register_forward_hook(capture(name)))
        if variant.startswith('input_'):
            for name in projection_names:handles.append(model.get_submodule(name).register_forward_pre_hook(roundtrip))
        if replace_projection is not None:
            for eid,expert in enumerate(model.get_submodule(replace_projection).experts):
                handles.append(expert.register_forward_hook(capture_old(eid)))
            handles.append(model.get_submodule(replace_projection).register_forward_hook(replacement))
        logprobs=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=variant=='native_float16'):
            groups=[(0,1)] if variant=='batch_two' else [(0,),(1,)]
            for group in groups:
                check();current_window=group[0]
                ids=torch.from_numpy(np.stack([arrays[f'ids_{i}'] for i in group])).to('cuda')
                labels=torch.from_numpy(np.stack([arrays[f'labels_{i}'] for i in group])).to('cuda')
                hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state
                assert torch.isfinite(hidden).all()
                logprobs.append(target_logprobs(model,hidden,labels))
        result=np.concatenate(logprobs)
        topk=np.stack([torch.cat(captures[n]).numpy() for n in captures]) if captures else np.empty((0,len(result),4),dtype=np.int64)
        assert result.shape==(2048,) and topk.shape==(len(captures),2048,4) and np.isfinite(result).all()
        return dict(logprobs=result,topk=topk,router_names=list(captures)),records
    finally:
        for handle in handles:handle.remove()
