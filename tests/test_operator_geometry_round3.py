import numpy as np
import torch
from fitmotn.experiments.operator_geometry.round3_capacity import ReencodingPredictor
from fitmotn.experiments.operator_geometry.round3_metric import analytic_metric
from fitmotn.experiments.operator_geometry.round3_replay_v2 import reweight


def test_pre_softmax_indices_survive_probability_rounding_and_calibration():
    # Recovered baseline case: two unequal logits become the same FP32 p.
    p=torch.tensor([[.4,.3,.2,.0070159900933504105,.0070159900933504105,.001]])
    idx=torch.tensor([[0,1,2,4]])
    for beta in [.1,1.,3.]:
        w=reweight(p,beta,idx)
        assert torch.equal(torch.where(w[0]>0)[0],torch.tensor([0,1,2,4]))
        torch.testing.assert_close(w.sum(1),torch.ones(1))
        torch.testing.assert_close(w[0,0]/w[0,1],(p[0,0]/p[0,1])**beta)


def test_interleaved_metric_formula_and_product_null_directions():
    torch.manual_seed(123)
    for method in ['reencode_cz','reencode_product']:
        m=ReencodingPredictor(12,8,method,np.arange(16)).double()
        a=torch.rand(8,dtype=torch.float64)+.4
        fn=lambda x:m.state(x[None])[0]
        psi=fn(a);j=torch.autograd.functional.jacobian(fn,a)
        qfi=4*(j.T@j-(j.T@psi[:,None])@(psi[None,:]@j))
        torch.testing.assert_close(qfi,analytic_metric(a[:4],method=='reencode_cz'),atol=1e-12,rtol=1e-12)
        assert int(torch.linalg.matrix_rank(j,atol=1e-8))==(8 if method=='reencode_cz' else 4)
        if method=='reencode_product':
            v=torch.cat([torch.ones(4,dtype=torch.float64),-torch.ones(4,dtype=torch.float64)])
            torch.testing.assert_close(j@v,torch.zeros(16,dtype=torch.float64),atol=1e-12,rtol=0)


def test_full_probability_geometry_at_graph_state_encoding():
    a=torch.full((4,),torch.pi/2,dtype=torch.float64)
    torch.testing.assert_close(analytic_metric(a),torch.eye(8,dtype=torch.float64),atol=1e-12,rtol=0)
