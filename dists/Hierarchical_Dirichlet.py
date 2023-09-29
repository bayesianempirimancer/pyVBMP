import torch
from .Dirichlet import Dirichlet

class Hierarchical_Dirichlet():
    def __init__(self,event_shape,batch_shape=(),prior_parms={'alpha':torch.tensor(0.5)}):
        print("Hierarchial_Dirichlet in its current form is redundant.  Just use Dirichlet with non-trivial event_shape")
        self.event_dim = len(event_shape)
        self.event_shape = event_shape
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)

        n_dims = self.event_dim
        shape = event_shape[:1]+(1,)*(n_dims-1)
        self.dists = [Dirichlet(event_shape = shape, batch_shape = batch_shape, prior_parms = prior_parms)]
        for i in range(n_dims-1):            
            shape = event_shape[i+1:i+2] + (1,)*(n_dims-1-i-1)
            batch_shape = self.batch_shape + event_shape[0:i+1]
            self.dists.append(Dirichlet(event_shape = shape, batch_shape = batch_shape, prior_parms = prior_parms))
        self.NA = 0.0

    def ss_update(self,NA,lr=1.0, beta=None):
        assert(NA.ndim == self.batch_dim + self.event_dim)
        if beta is not None:
            self.NA = beta*self.NA + NA
        else:
            self.NA = NA
        for i in range(self.event_dim-1):
            self.dists[i].ss_update(self.NA.sum(list(range(-self.event_dim+i+1,0)),True),lr=lr,beta=beta)
        self.dists[-1].ss_update(self.NA,lr=lr,beta=beta)

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

        
