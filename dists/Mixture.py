
import torch
from .Dirichlet import Dirichlet

class Mixture():
    # This class takes takes in a distribution with non trivial batch shape and 
    # produces a mixture distribution with the number of mixture components equal
    # to the terminal dimension of the batch shape.  The mixture distribution 
    # has batch shape equal to the batch shape of the input distribution minus the final dimension
    #
    # IMPORTANT:  This routine expects data to be sample_shape + dist.batch_shape[:-1] + (1,) + dist.event_shape 
    #             or if running VB batches in parallel:  sample_shape + (1,)*mix.batch_dim  + (1,) + dist.event_shape
    #       when this is the case the observations will not need to be reshaped at any time.  Only p will be reshaped for raw_updates


    def __init__(self, dist, event_shape, prior_parms={'alpha':torch.tensor(0.5)}):
        assert dist.batch_shape[-len(event_shape):] == event_shape
        self.event_shape = event_shape
        self.event_dim = len(event_shape)
        self.batch_shape = dist.batch_shape[:-len(event_shape)]
        self.batch_dim = len(self.batch_shape)
    
        self.pi = Dirichlet(event_shape = event_shape, batch_shape = self.batch_shape, prior_parms = prior_parms)
        self.dist = dist
        self.logZ = torch.tensor(-torch.inf,requires_grad=False)
        self.ELBO_last = torch.tensor(-torch.inf)

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.pi.to_event(n)
        self.dist.to_event(n)
        return self

    def update_assignments(self,X):
            log_p = self.Elog_like(X)
            shift = log_p.max(-1,True)[0]
            logZ = self.stable_logsumexp(log_p,dim = list(range(-self.event_dim,0)),keepdim=False)
            log_p = log_p - logZ.view(logZ.shape + self.event_dim*(1,))
            self.p = log_p.exp()
            self.NA = self.p.sum(0)
            self.logZ = logZ.sum(0)
            while self.NA.ndim > self.event_dim + self.batch_dim:
                self.logZ = self.logZ.sum(0)
                self.NA = self.NA.sum(0)

    def update_parms(self,X,lr=1.0):
        self.pi.ss_update(self.NA,lr=lr)
        self.update_dist(X,lr=lr)

    def raw_update(self,X,iters=1,lr=1.0,verbose=False):
        self.update(X,iters=iters,lr=lr,verbose=verbose)

    def update(self,X,iters=1,lr=1.0,verbose=False):
        for i in range(iters):
            # E-Step
            self.update_assignments(X)
            ELBO = self.ELBO()
            self.update_parms(X,lr)
            if verbose:
                print('Percent Change in ELBO:   ',(ELBO-self.ELBO_last)/self.ELBO_last.abs()*100.0)
            self.ELBO_last = ELBO

    def update_dist(self,X,lr):
        X = X.view(X.shape[:-self.dist.event_dim]+self.event_dim*(1,) + self.dist.event_shape)
        self.dist.raw_update(X,self.p,lr)

    def Elog_like(self,X):
        X = X.view(X.shape[:-self.dist.event_dim]+self.event_dim*(1,) + self.dist.event_shape)
        return self.dist.Elog_like(X) + self.pi.loggeomean()

    def KLqprior(self):
        return self.dist.KLqprior().sum(list(range(-self.event_dim,0))) + self.pi.KLqprior()  

    def ELBO(self):
        return self.logZ - self.KLqprior()

    def assignment_pr(self):
        return self.p

    def assignment(self):
        return self.p.argmax(-1)

    def means(self):
        return self.dist.mean()

    def event_average_f(self,function_string,A=None,keepdim=False):
        if A is None:
            return self.event_average(eval('self.dist.'+function_string)(),keepdim=keepdim)
        else:   
            return self.event_average(eval('self.dist.'+function_string)(A),keepdim=keepdim)

    def average_f(self,function_string,A=None,keepdim=False):
        if A is None:
            return self.average(eval('self.dist.'+function_string)(),keepdim=keepdim)
        else:
            return self.average(eval('self.dist.'+function_string)(A),keepdim=keepdim)

    def average(self,A,keepdim=False):  
        return (A*self.p).sum(-1,keepdim)

    ### Compute special expectations used for VB inference
    def event_average(self,A,keepdim=False):  # returns sample_shape + W.event_shape
        # A is mix_batch_shape + mix_event_shape + event_shape
        out = (A*self.p.view(self.p.shape+(1,)*self.dist.event_dim)).sum(-1-self.dist.event_dim,keepdim)
        for i in range(self.event_dim-1):
            out = out.sum(-self.dist.event_dim-1,keepdim)
        return out

    def stable_logsumexp(self,x,dim=None,keepdim=False):
        if isinstance(dim,int):
            xmax = x.max(dim=dim,keepdim=True)[0]
            if(keepdim):
                return xmax + (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
            else:
                return xmax.squeeze(dim) + (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
        else:
            xmax = x
            for d in dim:
                xmax = xmax.max(dim=d,keepdim=True)[0]
            if(keepdim):
                return xmax + (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
            else:
                x = (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
                for d in dim:
                    xmax = xmax.squeeze(d)
                return xmax + x




