# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 

import torch
from .DiagonalWishart import DiagonalWishart

class DiagonalWishart_UnitTrace(DiagonalWishart):
    # Despite the name this enforces a trace constraint of TR(EinvSigma)=D where D is the dimension of the matrix
    # so that the average eigenvalue of EinvSigma is 1.

    def suminv_d_plus_x(self,x):
        return (self.gamma.alpha/(self.gamma.beta+x)).sum(-1,True)

    def suminv_d_plus_x_prime(self,x):
        return -(self.gamma.alpha/(self.gamma.beta+x)**2).sum(-1,True)

    def ss_update(self,SExx,N,lr=1.0,beta=None,iters=10):
        super().ss_update(SExx,N,lr=lr,beta=beta)
#        x=self.gamma.alpha.sum(-1,True)
        x = torch.zeros(self.gamma.beta.shape[:-1]+(1,),requires_grad=False)
        for i in range(iters):
            x = x + (self.dim-self.suminv_d_plus_x(x))/self.suminv_d_plus_x_prime(x)
            idx = x<-self.gamma.beta.min(-1,True)[0]
            x = x*(~idx) + (-self.gamma.beta.min(-1,True)[0]+1e-4)*idx  # ensure positive definite

        self.rescale =  1+x/self.gamma.beta
        self.gamma.beta = self.gamma.beta+x
        

