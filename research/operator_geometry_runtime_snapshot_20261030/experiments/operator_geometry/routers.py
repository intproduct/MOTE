"""Controlled small-state probes of expert-to-basis geometry.

These are adaptation baselines, not reproductions of published end-to-end runs.
All state-vector work runs on a classical GPU. Product Born is algebraically a
classical product-Bernoulli selector and is explicitly included as that control.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...quantum_gate import apply_ry
from ..routing_stability.routers import SharedBornRouter


def cube_edges(experts):
    q=experts.bit_length()-1
    if 2**q != experts:raise ValueError('power-of-two expert count required')
    return np.array([(i,i^(1<<bit)) for i in range(experts) for bit in range(q) if i < (i^(1<<bit))])


def optimize_assignment(distance,seed,restarts=4,steps=3000):
    """Train-only hill-climbing assignment: nearby basis states, similar outputs."""
    n=len(distance);edges=cube_edges(n);rng=np.random.default_rng(seed)
    def cost(p):return float(distance[p[edges[:,0]],p[edges[:,1]]].mean())
    identity=np.arange(n);best=identity.copy();best_cost=cost(best)
    random_costs=[cost(rng.permutation(n)) for _ in range(128)]
    for restart in range(restarts):
        p=identity.copy() if restart==0 else rng.permutation(n);current=cost(p)
        for _ in range(steps):
            i,j=rng.choice(n,2,replace=False);p[i],p[j]=p[j],p[i]
            candidate=cost(p)
            if candidate<current:current=candidate
            else:p[i],p[j]=p[j],p[i]
        if current<best_cost:best,best_cost=p.copy(),current
    return best,{'identity_cost':cost(identity),'functional_cost':best_cost,
                 'random_mean_cost':float(np.mean(random_costs)),'random_sd_cost':float(np.std(random_costs,ddof=1)),
                 'expert_for_basis':best.tolist(),'objective':'mean train output squared distance along hypercube edges'}


def spectral_codes(distance,width=4):
    """Classical MDS feature coordinates use exactly the same train-only matrix."""
    n=len(distance);center=np.eye(n)-np.ones((n,n))/n
    gram=-.5*center@distance@center
    values,vectors=np.linalg.eigh(gram);order=np.argsort(values)[::-1][:width]
    out=vectors[:,order]*np.sqrt(np.maximum(values[order],0))[None,:]
    # Deterministic sign convention; fixed codes are not learned parameters.
    for j in range(width):
        if out[np.argmax(abs(out[:,j])),j]<0:out[:,j]*=-1
    return out/(np.sqrt(np.mean(out**2))+1e-12)


class ControlledRouter(nn.Module):
    def __init__(self,dim,experts,rank,method,assignment,codes,random_perm):
        super().__init__();self.method=method;self.experts=experts;self.q=experts.bit_length()-1
        self.norm=nn.LayerNorm(dim);self.input_proj=nn.Linear(dim,rank,bias=False)
        if method=='mlp':
            self.output=nn.Linear(rank,experts)
        elif method=='shared_born':
            self.shared=SharedBornRouter(dim,experts,rank)
            del self.norm,self.input_proj
        else:
            self.output=nn.Linear(rank,self.q)
            if method.startswith('spectral'):
                code=codes if method=='spectral_functional' else codes[random_perm]
                self.register_buffer('codes',torch.tensor(code,dtype=torch.float32))
            else:
                self.rotation=nn.Parameter(torch.zeros(2,self.q))
                basis=torch.arange(experts)
                shifts=torch.arange(self.q-1,-1,-1)
                bits=(basis[:,None]>>shifts)&1
                self.register_buffer('signs',(1-2*bits).float())
                phase=torch.ones(experts)
                for j in range(self.q-1):phase*=1-2*(bits[:,j]*bits[:,j+1])
                self.register_buffer('cz_phase',phase)
                mapping=assignment if method in ('angle_functional','product_functional') else (random_perm if method=='angle_random' else np.arange(experts))
                self.register_buffer('inverse_assignment',torch.tensor(np.argsort(mapping)))
                if method=='exp_adapter':self.adapter=nn.Linear(self.q,experts)
        if hasattr(self,'output'):
            nn.init.normal_(self.output.weight,std=.01);nn.init.zeros_(self.output.bias)

    def state(self,angles):
        state=angles.new_zeros(len(angles),self.experts);state[:,0]=1
        for wire in range(self.q):state=apply_ry(state,angles[:,wire],wire,self.q)
        for wire in range(self.q):state=apply_ry(state,self.rotation[0,wire],wire,self.q)
        if self.method!='product_functional':state=state*self.cz_phase
        for wire in range(self.q):state=apply_ry(state,self.rotation[1,wire],wire,self.q)
        return state

    def forward(self,x):
        if self.method=='shared_born':return self.shared(x)
        code=torch.tanh(self.input_proj(self.norm(x)))
        value=self.output(code)
        if self.method=='mlp':return value.softmax(-1)
        if self.method.startswith('spectral'):return (value@self.codes.T/math.sqrt(self.q)).softmax(-1)
        angles=math.pi/2+(math.pi/2)*torch.tanh(value)
        p=self.state(angles).square();p=p/p.sum(-1,keepdim=True)
        if self.method=='exp_adapter':return self.adapter(p@self.signs).softmax(-1)
        p=p[:,self.inverse_assignment]
        return (p+1e-6)/(1+self.experts*1e-6)


def sparse_weights(p,k,ste=False):
    mask=torch.zeros_like(p).scatter(1,p.topk(k,-1).indices,1)
    w=p*mask;w=w/(w.sum(-1,keepdim=True)+1e-12)
    # A dense counterfactual gradient is available from the offline expert bank.
    # All candidate routers use this same explicit estimator.
    return w.detach()+p-p.detach() if ste else w
