import torch
from .Wishart_eigh import Wishart_eigh

class Wishart_UnitTrace(Wishart_eigh):

    def suminv_d_plus_x(self,x):
        return self.nu*(1.0/(self.d+x)).sum(-1)

    def suminv_d_plus_x_prime(self,x):
        return -self.nu*(1.0/(self.d+x)**2).sum(-1)

    def ss_update(self,SExx,n,lr=1.0,beta=None,iters=8):
        super().ss_update(SExx,n,lr=lr,beta=beta)
        x=self.d.mean(-1)
        for i in range(iters):
            x = x + (self.dim-self.suminv_d_plus_x(x))/self.suminv_d_plus_x_prime(x)
            x[x<-self.d.min()] = -self.d.min()+1e-6  # ensure positive definite
        self.d = self.d+x

