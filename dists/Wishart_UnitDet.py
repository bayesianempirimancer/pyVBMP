import torch
from .Wishart_eigh import Wishart_eigh

class Wishart_UnitDet(Wishart_eigh):

    def log_mvdigamma_prime(self,nu):
        return (nu.unsqueeze(-1) - torch.arange(self.dim)/2.0).polygamma(1).sum(-1)

    def ss_update(self,SExx,n,lr=1.0,beta=None,iters=4):
        super().ss_update(SExx,n,lr=lr,beta=beta)
        log_mvdigamma_target = -self.dim*torch.log(torch.tensor(2,requires_grad=False)) + self.logdet_invU
        lognu = (log_mvdigamma_target/self.dim)
        for k in range(iters):
            lognu = lognu + (log_mvdigamma_target-self.log_mvdigamma(lognu.exp()))/self.log_mvdigamma_prime(lognu.exp())*(-lognu).exp()
        self.nu = 2.0*lognu.exp()
