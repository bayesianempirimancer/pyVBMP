# Gamma distribution as conjugate prior for Poisson distribution
# raw update assumes Poisson observation model

import torch

class Gamma():
    def __init__(self,event_shape=(),batch_shape=(),
                 prior_parms={'alpha':torch.tensor(1.0,requires_grad=False),
                              'beta':torch.tensor(1.0,requires_grad=False)}):

        self.event_dim = len(event_shape)
        self.event_shape = event_shape
        self.batch_dim = len(batch_shape)
        self.batch_shape = batch_shape
        self.nat_parms_0 = prior_parms
        
        self.alpha_0 = prior_parms['alpha'].expand(batch_shape + event_shape)
        self.beta_0 = prior_parms['beta'].expand(batch_shape + event_shape)

        self.alpha = self.alpha_0 + torch.rand(self.alpha_0.shape,requires_grad=False)
        self.beta = self.beta_0 + torch.rand(self.beta_0.shape,requires_grad=False)
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
        sample_shape = pX.shape[:-self.event_dim-self.batch_dim]
        if p is None: 
            # assumes X is sample x batch x event
            N = torch.prod(torch.tensor(sample_shape),requires_grad=False)
            N = N.expand(self.batch_shape+self.event_shape)
            SEx=pX.mean().sum(list(range(len(sample_shape))))
        else:
            p=p.view(p.shape + (1,)*self.event_dim)  # now p is sample x batch x event
            SEx = (pX.mean()*p).sum(list(range(len(sample_shape))))
            N = p.sum(list(range(len(sample_shape))))

        self.ss_update(SEx,N,lr=lr,beta=beta)
        
    def raw_update(self,X,p=None,lr=1.0,beta=None):
        sample_shape = X.shape[:-self.event_dim-self.batch_dim]
        if p is None: 
            # assumes X is sample x batch x event
            N = torch.prod(torch.tensor(sample_shape,requires_grad=False))
            N = N.expand(self.batch_shape+self.event_shape)
            SEx=X.sum(list(range(len(sample_shape))))
        else:
            p=p.view(p.shape + (1,)*self.event_dim)  # now p is sample x batch x event
            SEx = (X*p).sum(list(range(len(sample_shape))))
            N = p.sum(list(range(len(sample_shape))))

        self.ss_update(SEx,N,lr=lr,beta=beta)

    def Elog_like(self,X):   # ASSUMES POISSON OBSERVATION MODEL
        return (X*self.loggeomean()- (X+1).lgamma() - self.mean()).sum(list(range(-self.event_dim,0)))

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
        return KL.sum(list(range(-self.event_dim,0)))


