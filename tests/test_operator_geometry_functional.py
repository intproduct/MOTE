import torch
from types import SimpleNamespace
from fitmotn.experiments.operator_geometry.round3_functional import gram_matrices,quadratic
from fitmotn.experiments.operator_geometry.metrics import head

def test_fisher_pullback_matches_exact_hessian_and_common_shift():
    torch.manual_seed(7)
    lm=SimpleNamespace(norm=torch.nn.LayerNorm(6).double(),embedding=torch.nn.Embedding(13,6).double())
    h=torch.randn(2,6,dtype=torch.float64);v=torch.randn(2,4,6,dtype=torch.float64)
    g,f=gram_matrices(lm,h,v)
    gs,fs=gram_matrices(lm,h,v+torch.randn(2,1,6,dtype=torch.float64))
    torch.testing.assert_close(g,gs);torch.testing.assert_close(f,fs)
    for i in range(2):
        lp=head(lm,h[i]).log_softmax(-1)
        def kl(d):
            q=head(lm,h[i]+d@(v[i]-v[i].mean(0))).log_softmax(-1)
            return (lp.exp()*(lp-q)).sum()
        H=torch.autograd.functional.hessian(kl,torch.zeros(4,dtype=torch.float64))
        torch.testing.assert_close(f[i],H/2,atol=1e-12,rtol=1e-10)
        assert torch.linalg.eigvalsh(f[i]).min()>-1e-12
    d=torch.randn(2,4,dtype=torch.float64);d-=d.mean(1,keepdim=True)
    torch.testing.assert_close(quadratic(d,g),((d[:,:,None]*v).sum(1)).square().mean(1))
