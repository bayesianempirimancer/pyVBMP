import torch
from .Dirichlet import Dirichlet

# This version of the Hierarchical Dirichlet is a parameterizes a joint distribution over 
# 3 or more dimensions that has a conditional independence structure of the form:
# p(x_0,x_1,...,x_{n-1}) = p(x_0)p(x_1|x_0)...p(x_k|x_{k-1})...p(x_{n-1}|x_{n-2})
#
# The log of the code is to construct a list of dirichlet distributions of length n where
#    dist[0] ~ p(x_0)
#       dist[0].batch_shape = ()
#       dist[0].event_shape = event_shape[0] + (1,)*(n-1)
#    dist[1] ~ p(x_1|x_0)
#       dist[1].batch_shape = event_shape[0]
#       dist[1].event_shape = event_shape[1] + (1,)*(n-1)
#    dist[2] ~ p(x_2|x_1)
#       dist[2].batch_shape = (1,) + event_shape[1]
#       dist[2].event_shape = event_shape[2] + (1,)*(n-2)
#    ....
#    dist[k] ~ p(x_k|x_{k-1})
#       dist[k].batch_shape = (1,)*(k-1) + event_shape[k-1]
#       dist[k].event_shape = event_shape[k] + (1,)*(n-k-1) 
#    ....
#    dist[n-1] ~ p(x_{n-1}|x_{n-2})
#       dist[n-1].batch_shape = (1,)*(n-3) + event_shape[n-2]
#       dist[n-1].event_shape = event_shape[n-1] 
#


class Hierarchical_Dirichlet():
    def __init__(self,event_shape,batch_shape=(),prior_parms={'alpha':torch.tensor(0.5)}):
        self.event_dim = len(event_shape)
        self.event_shape = event_shape
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)

        n_dims = self.event_dim
        shape = event_shape[:1]+(1,)*(n_dims-1)
        self.dists = [Dirichlet(event_shape = shape, batch_shape = batch_shape, prior_parms = prior_parms)]
        self.sum_list = [list(range(-n_dims+1,0))]
        for i in range(n_dims-1):            
            shape = event_shape[i+1:i+2] + (1,)*(n_dims-1-i-1)
            batch_shape = self.batch_shape + (1,)*i+event_shape[i:i+1]
            self.dists.append(Dirichlet(event_shape = shape, batch_shape = batch_shape, prior_parms = prior_parms))
            self.sum_list.append(list(range(-n_dims,-n_dims+i))+ list(range(-n_dims+i+2,0)))
        self.NA = 0.0

    def ss_update(self,NA,lr=1.0, beta=None):
        assert(NA.ndim == self.batch_dim + self.event_dim)
        if beta is not None:
            self.NA = beta*self.NA + NA
        else:
            self.NA = NA
        for i in range(self.event_dim):
            self.dists[i].ss_update(self.NA.sum(self.sum_list[i],True),lr=lr,beta=beta)

    def raw_update(self,X,p=None,lr=1.0,beta=None):
        sample_dim = X.ndim - self.batch_dim - self.event_dim
        if p is None: 
            NA = X.sum(list(range(sample_dim)))
        else:
            p = p.view(p.shape + (1,)*self.event_dim)
            NA = (X*p).sum(list(range(sample_dim)))
        self.ss_update(NA,lr,beta)

    def update(self,X,p=None,lr=1.0,beta=None):
        self.raw_update(X,p,lr,beta)

    def marginal(self,idx):
        raise NotImplementedError

    def mean(self):
        p = self.dists[0].mean()
        for i in range(1,self.event_dim):
            p = p*self.dists[i].mean()
        return p

    def loggeomean(self):
        logp = self.dists[0].ElogX()
        for i in range(1,len(self.dists)):
            logp = logp + self.dists[i].ElogX()
        return logp

    def ElogX(self):
        logp = self.dists[0].ElogX()
        for i in range(1,len(self.dists)):
            logp = logp + self.dists[i].ElogX()
        return logp

    def KLqprior(self):

        KL = self.dists[0].KLqprior()
        for i in range(1,len(self.dists)):
            KL = KL + self.dists[i].KLqprior().sum(list(range(-i,0)))
        return KL

        
