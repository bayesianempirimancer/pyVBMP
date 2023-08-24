# Implements a hierarchal dirichlet distribution with num_levels levels
# The distribution is parameterized by a tensor alpha_0 which has event_shape so that the total number of states in the distribution
# is given by np.prod(alpha.shape).  Ultimately, the only differnce between this and the starndard Dirichlet is that alpha_sum terms are 
# done over the entire event shape rather than just the last dimension.  

import torch

class Dirichlet_Tensor():
    def __init__(self,alpha_0,n_dims=1):
        self.event_dim = n_dims
        self.batch_dim = alpha_0.ndim - n_dims
        self.event_shape = alpha_0.shape[-n_dims:]
        self.batch_shape = alpha_0.shape[:-n_dims]
        self.alpha_0 = alpha_0
        self.alpha = self.alpha_0 + 2.0*torch.rand(self.alpha_0.shape,requires_grad=False)*self.alpha_0 # type: ignore
        self.NA = 0.0
        self.sum_list = list(range(-self.event_dim,0))

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]        
        return self

    def ss_update(self,NA,lr=1.0, beta=None):
        # while NA.ndim > self.event_dim + self.batch_dim:
        #     NA = NA.sum(0)
        assert(NA.ndim == self.batch_dim + self.event_dim)
        if beta is not None:
            self.NA = beta*self.NA + NA
        else:
            self.NA = NA
        self.alpha = lr*(self.NA + self.alpha_0) + (1-lr)*self.alpha

    def raw_update(self,X,p=None,lr=1.0,beta=None):
        if p is None: 
            # assumes X is sample x batch x event
            NA = X
        else:
            # assumes X is batch consistent
            p = p.view(p.shape + (1,)*self.event_dim)
            NA = (X*p).sum(0)
        while NA.ndim > self.event_dim + self.batch_dim:
            NA = NA.sum(0)
        self.ss_update(NA,lr,beta)

    def update(self,X,p=None,lr=1.0,beta=None):
        self.raw_update(X,p,lr,beta)

    def Elog_like(self,X):
        # assumes multinomial observations with data.shape = samples x batch_shape* x event_shape
        # returns sample shape x batch shape
        return (X*self.loggeomean()).sum(self.sum_list) + (1+X.sum(self.sum_list)).lgamma() - (1+X).lgamma().sum(self.sum_list)

    def mean(self):
        return self.alpha/self.alpha.sum(self.sum_list,keepdim=True)

    def loggeomean(self):
        return self.alpha.digamma() - self.alpha.sum(self.sum_list,keepdim=True).digamma()

    def ElogX(self):
        return self.alpha.digamma() - self.alpha.sum(self.sum_list,keepdim=True).digamma()

    def var(self):
        alpha_sum = self.alpha.sum(self.sum_list,keepdim=True)
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
        alpha_sum = self.alpha.sum(self.sum_list)
        alpha_0_sum = self.alpha_0.sum(self.sum_list)

        KL = alpha_sum.lgamma() - self.KL_lgamma(self.alpha).sum(self.sum_list)
        KL = KL  - alpha_0_sum.lgamma() + self.KL_lgamma(self.alpha_0).sum(self.sum_list)
        KL = KL + ((self.alpha-self.alpha_0)*(self.KL_digamma(self.alpha)-alpha_sum.digamma().view(alpha_sum.shape + (1,)*self.event_dim))).sum(self.sum_list)

        while KL.ndim > self.batch_dim:
            KL = KL.sum(-1)
        return KL

    def logZ(self):
        return self.alpha.lgamma().sum(self.sum_list) - self.alpha.sum(self.sum_list).lgamma()

        
