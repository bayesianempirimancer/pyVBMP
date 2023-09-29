import torch
from .Gamma import Gamma

class NormalGamma():
    # Independent Normal-Gamma distribution:  No matrix inversions required
    def __init__(self,event_shape, batch_shape=(), scale=torch.tensor(1.0),
                 prior_parms = {'lambda_mu':torch.tensor(1.0),
                                'mu':torch.tensor(0.0), 
                                'alpha':torch.tensor(2.0),
                                'beta':torch.tensor(2.0)}):

        self.dim = event_shape[-1]
        self.event_dim = 1 
        self.event_shape = event_shape
        self.batch_dim = len(batch_shape)
        self.batch_shape = batch_shape

        self.lambda_mu_0 = prior_parms['lambda_mu'].expand(batch_shape + event_shape[:-1])
        self.lambda_mu = self.lambda_mu_0 + torch.rand_like(self.lambda_mu_0,requires_grad=False)
        self.mu_0 = prior_parms['mu'].expand(batch_shape + event_shape)

        self.gamma = Gamma(event_shape = event_shape, batch_shape = batch_shape, 
                           prior_parms = {'alpha':prior_parms['alpha'],'beta':prior_parms['beta']*scale**2})
        self.mu = self.mu_0 + torch.randn_like(self.mu_0,requires_grad=False)/self.gamma.mean().sqrt()
        self.SExx = 0.0
        self.SEx = 0.0
        self.N = 0.0

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.gamma.to_event(n)
        return self

    def ss_update(self,SExx,SEx,N, lr=1.0, beta=None):
        if beta is not None:
            self.SExx = SExx + beta*self.SExx
            self.SEx = SEx + beta*self.SEx
            self.N = N + beta*self.N
            SExx = self.SExx
            SEx = self.SEx
            N = self.N

        lambda_mu = self.lambda_mu_0 + N
        mu = (self.lambda_mu_0.unsqueeze(-1)*self.mu_0 + SEx)/lambda_mu.unsqueeze(-1)
        SExx = SExx + self.lambda_mu_0.unsqueeze(-1)*self.mu_0**2 - lambda_mu.unsqueeze(-1)*mu**2

        self.lambda_mu = lr*lambda_mu + (1-lr)*self.lambda_mu
        self.mu = lr*mu + (1-lr)*self.mu
        self.gamma.ss_update(0.5*N.unsqueeze(-1),0.5*SExx,lr,beta)

    def raw_update(self,X,p=None,lr=1.0,beta=None):
        sample_shape = X.shape[:-self.event_dim-self.batch_dim]
        sample_dims = list(range(len(sample_shape)))
        if p is None:  # data is sample_shape + batch_shape + event_event_shape 
            SEx = X.sum(sample_dims)
            SExx = (X**2).sum(sample_dims)
            N = torch.prod(torch.tensor(sample_shape,requires_grad=False))
            N = N.expand(self.batch_shape + self.event_shape[:-1])
            self.ss_update(SExx,SEx,N,lr,beta)  # inputs to ss_update must be batch + event consistent

        else:  # data is sample_shape + batch_shape* + event_shape and p is num_samples x batch_shape
                # batch_shape* can be (1,)*batch_dim 
            N = p.sum(sample_dims)
            p=p.view(p.shape + self.event_dim*(1,))
            SEx = (X*p).sum(sample_dims)
            SExx = (X**2*p).sum(sample_dims)
            self.ss_update(SExx,SEx,N,lr,beta)  # inputs to ss_update must be batch + event consistent


    def Elog_like(self,X):
        # X is num_samples x num_dists x dim
        # returns num_samples x num_dists
        # output should be num_samples  

        out = -0.5*(X.pow(2)*self.gamma.mean()).sum(-1) + (X*self.EinvSigmamu()).sum(-1) - 0.5*(self.EXTinvUX())
        out = out + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*torch.log(2*torch.tensor(torch.pi,requires_grad=False))
        out = -0.5*((X - self.mu)**2*self.gamma.mean()).sum(-1) + 0.5*self.gamma.loggeomean().sum(-1) 
        for i in range(self.event_dim-1):
            out = out.sum(-1)
        return out

    def KLqprior(self):

        out = self.lambda_mu_0/2.0*((self.mu-self.mu_0)**2*self.gamma.mean()).sum(-1) 
        out = out + self.dim/2.0*(self.lambda_mu_0/self.lambda_mu - (self.lambda_mu_0/self.lambda_mu).log() -1)
        for i in range(self.event_dim-1):
            out = out.sum(-1)
        return out + self.gamma.KLqprior().sum(-1)

    def mean(self):
        return self.mu

    def Emumu(self):
        return self.mu.unsqueeze(-2)*self.mu.unsqueeze(-1) + self.ESigma()/self.lambda_mu.unsqueeze(-1).unsqueeze(-1)

    def ElogdetinvSigma(self):
        return self.gamma.loggeomean().sum(-1)

    def EmuTinvSigmamu(self):
        return (self.mu**2*self.gamma.mean()).sum(-1) + self.dim/self.lambda_mu

    def EXTinvUX(self):
        return (self.mu**2*self.gamma.mean()).sum(-1) + self.dim/self.lambda_mu

    def EinvSigma(self):
        return self.gamma.mean().unsqueeze(-1)*torch.eye(self.dim,requires_grad=False)
        
    def ESigma(self):
        return self.gamma.meaninv().unsqueeze(-1)*torch.eye(self.dim,requires_grad=False)

    def Res(self):
        return -0.5*self.EXTinvUX() + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*torch.log(2*torch.tensor(torch.pi,requires_grad=False))

    def EinvSigmamu(self):
        return self.gamma.mean()*self.mu
