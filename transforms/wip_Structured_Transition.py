

import torch
import dists.Dirichlet as Dirichlet

class Structured_Transition():
    def __init__(self,event_shape,batch_shape=(), prior_parms = None):
        self.event_dim = len(event_shape)
        self.event_shape = event_shape
        self.batch_shape = batch_shape + event_shape
        self.batch_dim = len(self.batch_shape)

        n_dims = self.event_dim
        dims = event_shape
        if prior_parms is None:
            alpha_0 = torch.tensor(0.5)
            alpha_sticky = torch.tensor(1.0)
        else:
            alpha_0 = prior_parms['alpha']
            alpha_sticky = 0.0

        self.dists = []
        self.sum_list = []
        for i in range(0,n_dims):
            shape1 = dims[:i+1] + (1,)*(n_dims-1-i)
            shape2 = (1,)*(i) + dims[i:i+1] + (1,)*(n_dims-1-i)
            alpha = alpha_0.expand(shape1 + shape2) + alpha_sticky*torch.eye(dims[i],requires_grad=False).view(2*shape2)
            self.dists.append(Dirichlet(event_shape = shape2 , batch_shape = batch_shape + shape1, prior_parms = {'alpha':alpha}))  
            sum_list1 = list(range(-2*n_dims+i+1,-n_dims))
            sum_list2 = [x for x in list(range(-n_dims,0)) if x not in (-n_dims+i,)]
            self.sum_list.append(sum_list1 + sum_list2)
        self.NA = 0.0

    def ss_update(self,NA,lr=1.0, beta=None):
        assert(NA.shape == self.batch_shape + self.event_shape)
        if beta is not None:
            self.NA = beta*self.NA + NA
        else:
            self.NA = NA
        for i in range(len(self.dists)):
            self.dists[i].ss_update(NA.sum(self.sum_list[i],True),lr=lr,beta=beta)

    def raw_update(self,X,p=None,lr=1.0,beta=None):
        sample_shape = X.shape[:-self.batch_dim-self.event_dim]
        sample_dims = list(range(len(sample_shape)))
        if p is None: 
            # assumes X is sample x batch x event
            NA = X.sum(sample_dims)
        else:
            # assumes X is batch consistent
            pv = p.view(p.shape + (1,)*self.event_dim)
            NA = (X*pv).sum(sample_dims)
        self.ss_update(NA,lr,beta)

    def update(self,X,p=None,lr=1.0,beta=None):
        self.raw_update(X,p,lr,beta)

    def marginal(self,idx):
        raise NotImplementedError
    
    def mean(self):
        p = self.dists[0].mean()
        for i in range(1,len(self.dists)):
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
        KL = self.dists[0].KLqprior().sum(list(range(-len(self.dists),0)))
        for i in range(1,len(self.dists)):
            KL = KL + self.dists[i].KLqprior().sum(list(range(-len(self.dists),0)))
        return KL

    def Elog_like(self,X):
        # assumes multinomial observations with data.shape = samples x batch_shape* x event_shape
        # returns sample shape x batch shape
        sum_list = list(range(-self.event_dim,0))
        return (X*self.loggeomean()).sum(sum_list) + (1+X.sum(sum_list)).lgamma() - (1+X).lgamma().sum(sum_list)
        
