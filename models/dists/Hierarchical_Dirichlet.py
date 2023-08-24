# Implements a hierarchal dirichlet distribution with dim = (top_level, level_1, level_2, ..., bottom_level)
# This distribution is encoded as a list of conditional distributions p(level_i | level_i-1) which is a Tensor 
# Dirichlet with shape = (1,...,1,dims(i-1),dims(i),1,..,1).  Here dimesnion i-1 is part of the batch dimension while the event_shape
# is (dims(i),1,...,1).  This choice takes advantage of broadcasting semantics with little effort.  

import torch
from Dirichlet_Tensor import Dirichlet_Tensor

class Hierarchical_Dirichlet():
    def __init__(self,dims,batch_shape=(),alpha_0=torch.tensor(0.5)):
        self.event_dim = len(dims)
        self.event_shape = dims
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)

        n_dims = self.event_dim
        shape = batch_shape + dims[:1]+(1,)*(n_dims-1)
        self.dists = [Dirichlet_Tensor(alpha_0.expand(shape),n_dims)]
        self.sum_list = [list(range(-self.event_dim+1,0))]
        for i in range(n_dims-1):
            shape = batch_shape + (1,)*i + dims[i:i+2] + (1,)*(n_dims-1-i-1)
            self.dists.append(Dirichlet_Tensor(alpha_0.expand(shape),n_dims-i-1))
            self.sum_list.append(list(range(-self.event_dim,-self.event_dim+i)) + list(range(-self.event_dim+i+2,0)))
        self.NA = 0.0

    def ss_update(self,NA,lr=1.0, beta=None):
        assert(NA.ndim == self.batch_dim + self.event_dim)
        if beta is not None:
            self.NA = beta*self.NA + NA
        else:
            self.NA = NA
        for i in range(len(self.dists)):
            self.dists[i].ss_update(NA.sum(self.sum_list[i],True),lr=lr,beta=beta)

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

    def marginal(self,idx):
        p = self.mean().sum(self.sum_list[idx],True)
        if idx == 0 or idx == -self.event_dim:
            return p 
        else:
            return p.sum(idx-1,True)

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

    def Elog_like(self,X):
        # assumes multinomial observations with data.shape = samples x batch_shape* x event_shape
        # returns sample shape x batch shape
        sum_list = list(range(-self.event_dim,0))
        return (X*self.loggeomean()).sum(sum_list) + (1+X.sum(sum_list)).lgamma() - (1+X).lgamma().sum(sum_list)
        
