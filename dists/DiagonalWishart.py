# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 

import torch
from .Gamma import Gamma

class DiagonalWishart():

    def __init__(self, event_shape, batch_shape=(),
                 prior_parms={'nu':torch.tensor(2.0),
                              'U':torch.tensor(0.5)}, scale = 1.0):
            # here nu_0 and U_0 are same shape and EinvSigma = U*nu is the prior precision mean on the diagonal 

        self.dim = event_shape[-1]
        self.event_dim = len(event_shape)
        self.event_shape = event_shape
        self.batch_dim = len(batch_shape)
        self.batch_shape = batch_shape
        self.gamma = Gamma(event_shape,batch_shape,prior_parms={'alpha':prior_parms['nu'],'beta':scale**2/prior_parms['U']}) 

    def to_event(self,n):
        if n==0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape 
        self.batch_shape = self.batch_shape[:-n]
        self.gamma.to_event(n)
        return self

    def ss_update(self,SExx,N,lr=1.0,beta=None):   # assumes SExx is the diagonal of a covariance matrix
        assert(SExx.ndim == self.batch_dim + self.event_dim)
        assert(N.ndim == self.batch_dim + self.event_dim)
#        idx = n>1
#        SExx = SExx*(idx).unsqueeze(-1)
        self.gamma.ss_update(N/2.0,SExx/2.0,lr,beta)

    def KLqprior(self):
        return self.gamma.KLqprior()

    def logZ(self):
        return self.gamma.logZ()

    # These expectations return Matrices with diagonal elements
    # generally one should avoid using these function and instead
    # use self.gamma.mean(), self.gamma.meaninv(), self.gamma.loggeomean()
    def ESigma(self):
        return self.tensor_diag(self.gamma.meaninv())

    def EinvSigma(self):
        return self.tensor_diag(self.gamma.mean())

    def ElogdetinvSigma(self):
        return self.gamma.loggeomean().sum(-1)

    def logdetEinvSigma(self):
        return self.gamma.mean().log().sum(-1)

    def mean(self):
        return self.tensor_diag(self.gamma.mean())

    def tensor_diag(self,A):
        return A.unsqueeze(-1)*torch.eye(A.shape[-1],requires_grad=False)

    def tensor_extract_diag(self,A):
        return A.diagonal(dim=-2,dim1=-1)
        
        

