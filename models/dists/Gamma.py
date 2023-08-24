# Gamma distribution as conjugate prior for Poisson distribution
# raw update assumes Poisson observation model

import torch
import numpy as np

class Gamma():
    def __init__(self,alpha,beta):
        self.event_dim = 0
        self.event_shape = ()
        self.batch_dim = alpha.ndim
        self.batch_shape = alpha.shape
        self.alpha_0 = alpha
        self.beta_0 = beta
        self.alpha = alpha + torch.rand(alpha.shape,requires_grad=False)
        self.beta = beta + torch.rand(alpha.shape,requires_grad=False)
        self.SEx = 0.0
        self.SElogx = 0.0
    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape 
        self.batch_shape = self.batch_shape[:-n]
        return self

    def ss_update(self,SElogx,SEx,lr=1.0, beta=None):
        assert(SElogx.ndim == self.batch_dim + self.event_dim)
        assert(SEx.ndim == self.batch_dim + self.event_dim)
        if beta is not None:
            self.SEx = beta*self.SEx + SEx
            self.SElogx = beta*self.SElogx + SElogx
            SEx = self.SEx
            SElogx = self.SElogx
        # while SEx.ndim > self.event_dim + self.batch_dim:
        #     SEx = SEx.sum(0)
        #     SElogx = SElogx.sum(0)
        self.alpha = (self.alpha_0 + SElogx)*lr + self.alpha*(1-lr)
        self.beta = (self.beta_0 + SEx)*lr + self.beta*(1-lr)

    def update(self,pX,p=None,lr=1.0,beta=None):
        if p is None: 
            # assumes X is sample x batch x event
            sample_shape = pX.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape+self.event_shape)
            SEx=pX.mean().sum(0)
            for i in range(len(sample_shape)-1):
                SEx = SEx.sum(0)
        else:
            n=p.view(p.shape + (1,)*self.event_dim)  # now p is sample x batch x event
            SEx = pX.mean()*n
            while SEx.ndim>self.event_dim + self.batch_dim:
                SEx = SEx.sum(0)
                n = n.sum(0)

        self.ss_update(SEx,n,lr=lr,beta=beta)
        

    def raw_update(self,X,p=None,lr=1.0,beta=None):

        if p is None: 
            # assumes X is sample x batch x event
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape+self.event_shape)
            SEx=X
            for i in range(len(sample_shape)):
                SEx = SEx.sum(0)
        else:
            n=p.view(p.shape + (1,)*self.event_dim)  # now p is sample x batch x event
            SEx = X*n
            while SEx.ndim>self.event_dim + self.batch_dim:
                SEx = SEx.sum(0)
                n = n.sum(0)

        self.ss_update(SEx,n,lr=lr,beta=beta)

    def Elog_like(self,X):   # ASSUMES POISSON OBSERVATION MODEL
        ELL = X*self.loggeomean()- (X+1).lgamma() - self.mean()
        for i in range(self.event_dim):
            ELL = ELL.sum(-1)
        return ELL

    def mean(self):
        return self.alpha/self.beta

    def var(self):
        return self.alpha/self.beta**2

    def meaninv(self):
        return self.beta/(self.alpha-1)

    def ElogX(self):
        return self.alpha.digamma() - self.beta.log()
    
    def loggeomean(self):
        return self.alpha.log() - self.beta.log()

    def entropy(self):
        return self.alpha.log() - self.beta.log() + self.alpha.lgamma() + (1-self.alpha)*self.alpha.digamma()
        
    def logZ(self): 
        return -self.alpha*self.beta.log() + self.alpha.lgamma()

    def logZprior(self):
        return -self.alpha_0*self.beta_0.log() + self.alpha_0.lgamma()

    def KLqprior(self):
        KL = (self.alpha-self.alpha_0)*self.alpha.digamma() - self.alpha.lgamma() + self.alpha_0.lgamma() + self.alpha_0*(self.beta.log()-self.beta_0.log()) + self.alpha*(self.beta_0/self.beta-1)
        for i in range(self.event_dim):
            KL = KL.sum(-1)
        return KL


