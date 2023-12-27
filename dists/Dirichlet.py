import torch

class Dirichlet():
    def __init__(self,event_shape,batch_shape=(),prior_parms={'alpha':torch.tensor(0.5)}):
        self.event_dim = len(event_shape)
        self.batch_dim = len(batch_shape)
        self.event_shape = event_shape
        self.batch_shape = batch_shape
        self.alpha_0 = prior_parms['alpha'].expand(batch_shape + event_shape)
        self.alpha = self.alpha_0*(1.0 + torch.rand(self.alpha_0.shape,requires_grad=False))
        self.NA = 0.0

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]        
        return self

    def ss_update(self,NA,lr=1.0, beta=None):
        assert(NA.shape == self.batch_shape + self.event_shape)
        if beta is not None:
            self.NA = beta*self.NA + NA
        else:
            self.NA = NA
        self.alpha = lr*(self.NA + self.alpha_0) + (1-lr)*self.alpha

    def raw_update(self,X,p=None,lr=1.0,beta=None):
        sample_dim = X.ndim - self.event_dim - self.batch_dim
        if p is None: 
            # assumes X is sample x batch x event
            NA = X.sum(list(range(sample_dim)))
        else:
            # assumes X is batch consistent
            p = p.view(p.shape + (1,)*self.event_dim)
            NA = (X*p).sum(list(range(sample_dim)))
        self.ss_update(NA,lr,beta)

    def update(self,X,p=None,lr=1.0,beta=None):
        self.raw_update(X,p,lr,beta)

    def Elog_like(self,X):
        # assumes multinomial observations with data.shape = samples x batch_shape* x event_shape
        # returns sample shape x batch shape
        return (X*self.loggeomean()).sum(list(range(-self.event_dim,0))) + (1+X.sum(list(range(-self.event_dim,0)))).lgamma() - (1+X).lgamma().sum(list(range(-self.event_dim,0)))

    def mean(self):
        return self.alpha/self.alpha.sum(list(range(-self.event_dim,0)),keepdim=True)

    def loggeomean(self):
        return self.alpha.digamma() - self.alpha.sum(list(range(-self.event_dim,0)),keepdim=True).digamma()

    def ElogX(self):
        return self.alpha.digamma() - self.alpha.sum(list(range(-self.event_dim,0)),keepdim=True).digamma()

    def var(self):
        alpha_sum = self.alpha.sum(list(range(-self.event_dim,0)),keepdim=True)
        mean = self.mean()
        return mean*(1-mean)/(alpha_sum+1)

    def KL_lgamma(self,x):
        out = x.lgamma()
        out[out== torch.inf]=0
        return out

    def KL_digamma(self,x):
        out = x.digamma()
        out[out== -torch.inf]=0
        return out

    def KLqprior(self):
        alpha_sum = self.alpha.sum(list(range(-self.event_dim,0)))
        alpha_0_sum = self.alpha_0.sum(list(range(-self.event_dim,0)))

        KL = alpha_sum.lgamma() - self.KL_lgamma(self.alpha).sum(list(range(-self.event_dim,0)))
        KL = KL  - alpha_0_sum.lgamma() + self.KL_lgamma(self.alpha_0).sum(list(range(-self.event_dim,0)))
        KL = KL + ((self.alpha-self.alpha_0)*(self.KL_digamma(self.alpha)-alpha_sum.digamma().view(alpha_sum.shape + (1,)*self.event_dim))).sum(list(range(-self.event_dim,0)))

        while KL.ndim > self.batch_dim:
            KL = KL.sum(-1)
        return KL

    def logZ(self):
        return self.alpha.lgamma().sum(list(range(-self.event_dim,0))) - self.alpha.sum(list(range(-self.event_dim,0))).lgamma()

