import torch
import numpy as np
from .Wishart import Wishart

class NormalInverseWishart(): 

    def __init__(self,lambda_mu_0=None,mu_0=None,nu_0=None,invV_0=None):

        self.event_dim = 1
        self.event_shape = mu_0.shape[-1:]
        self.batch_shape = mu_0.shape[:-1]
        self.batch_dim = mu_0.ndim - self.event_dim   
        self.dim = mu_0.shape[-1]
        if lambda_mu_0 is None:
            self.lambda_mu_0 = torch.ones(mu_0.shape[:-1],requires_grad=False)
        else:
            self.lambda_mu_0 = lambda_mu_0
        if nu_0 is None:
            nu_0 = torch.ones(mu_0.shape[:-1],requires_grad=False)*(self.dim+2)
        if invV_0 is None:
            invV_0 = torch.zeros(mu_0.shape+(self.dim,),requires_grad=False)+torch.eye(self.dim,requires_grad=False)

        self.lambda_mu = self.lambda_mu_0
        self.mu_0 = mu_0
        self.mu = mu_0 + torch.randn(mu_0.shape,requires_grad=False)

        self.invU = Wishart(nu_0,invV_0)
        self.SExx = 0.0
        self.SEx = 0.0
        self.N = 0.0

    def mean(self):
        return self.mu

    def EX(self):
        return self.mu
    
    def EXXT(self):
        return self.mu.unsqueeze(-1)*self.mu.unsqueeze(-2) + self.invU.ESigma()/self.lambda_mu.unsqueeze(-1).unsqueeze(-1)

    def ESigma(self):
        return self.invU.ESigma()
        
    def ElogdetinvSigma(self):
        return self.invU.ElogdetinvSigma()

    def EinvSigmamu(self):
        return (self.invU.EinvSigma()*self.mu.unsqueeze(-2)).sum(-1)

    def EinvSigma(self):
        return self.invU.EinvSigma()

    def EinvUX(self):
        return (self.invU.EinvSigma()*self.mu.unsqueeze(-2)).sum(-1)

    def EXTinvUX(self):
        return (self.mu.unsqueeze(-1)*self.invU.EinvSigma()*self.mu.unsqueeze(-2)).sum(-1).sum(-1) + self.dim/self.lambda_mu

    def to_event(self,n):
        if n ==0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.invU.to_event(n)
        return self

    def ss_update(self,SExx,SEx,N, lr=1.0, beta=None):
        # SExx is batch_shape + event_shape + (dim,)
        # SEx  is batch_shape + event_shape
        # n is batch_shape + event_shape[:-1]
        if beta is not None:
            self.SExx = beta*self.SExx + SExx
            self.SEx = beta*self.SEx + SEx
            self.N = beta*self.N + N
            SExx = self.SExx
            SEx = self.SEx
            N = self.N
        lambda_mu = self.lambda_mu_0 + N
        mu = (self.lambda_mu_0.unsqueeze(-1)*self.mu_0 + SEx)/lambda_mu.unsqueeze(-1)
        invV = SExx + self.lambda_mu_0.unsqueeze(-1).unsqueeze(-1)*self.mu_0.unsqueeze(-1)*self.mu_0.unsqueeze(-2) - lambda_mu.unsqueeze(-1).unsqueeze(-1)*mu.unsqueeze(-1)*mu.unsqueeze(-2)

        self.lambda_mu = lr*lambda_mu + (1-lr)*self.lambda_mu
        self.mu = lr*mu + (1-lr)*self.mu
        self.invU.ss_update(invV,N,lr)

    def raw_update(self,X,p=None,lr=1.0,beta=None):
        # assumes data is  num_samples (Times) x batch_shape x evevnt_dim
        # if specified p has shape num_samples x batch_shape
        # the critical manipulation here is that p averages over the batch dimension

        if p is None:  
            SEx = X.sum(0)
            SExx = (X.unsqueeze(-1)*X.unsqueeze(-2)).sum(0)
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape + self.event_shape[:-1])
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
            self.ss_update(SExx,SEx,n,lr,beta)  # inputs to ss_update must be batch + event consistent

        else:  # data is shape sample_shape x batch_shape x event_shape with the first batch dimension having size 1

            p=p.view(p.shape + (1,)*self.event_dim)
            SExx = (X.unsqueeze(-1)*X.unsqueeze(-2)*p.unsqueeze(-1)).sum(0) 
            SEx =  (X*p).sum(0)
            p=p.sum(0)
            
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
                p = p.sum(0)      
            # p now has shape batch_shape + event_shape so it must be squeezed by the default event_shape which is 1
            self.ss_update(SExx,SEx,p.squeeze(-1),lr,beta)  # inputs to ss_update must be batch + event consistent

    def Elog_like(self,X):
        # X is num_samples x batch_shape x event_shape  OR  num_samples x (1,)*batch_dim x event_shape

        out = -0.5*((X.unsqueeze(-1)*self.EinvSigma()).sum(-2)*X).sum(-1) + (X*self.EinvSigmamu()).sum(-1) - 0.5*(self.EXTinvUX())
        out = out + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)

        for i in range(self.event_dim-1):
            out = out.sum(-1)
        return out

    def KLqprior(self):
        KL = 0.5*(self.lambda_mu_0/self.lambda_mu - 1 + (self.lambda_mu/self.lambda_mu_0).log())*self.dim
        KL = KL + 0.5*self.lambda_mu_0*((self.mu-self.mu_0).unsqueeze(-1)*(self.mu-self.mu_0).unsqueeze(-2)*self.invU.mean()).sum(-1).sum(-1)
        for i in range(self.event_dim-1):
            KL = KL.sum(-1)
        KL = KL + self.invU.KLqprior()
        return KL



import torch
import numpy as np
from .Wishart import Wishart

class NormalInverseWishart_vector_format(): 

    def __init__(self,lambda_mu_0=None,mu_0=None,nu_0=None,invV_0=None):

        self.event_dim = 2
        self.event_shape = mu_0.shape[-2:]
        self.batch_shape = mu_0.shape[:-2]
        self.batch_dim = mu_0.ndim - self.event_dim   
        self.dim = mu_0.shape[-2]
        if lambda_mu_0 is None:
            self.lambda_mu_0 = torch.ones(mu_0.shape[:-2],requires_grad=False)
        else:
            self.lambda_mu_0 = lambda_mu_0
        if nu_0 is None:
            nu_0 = torch.ones(mu_0.shape[:-2],requires_grad=False)*(self.dim+2)
        if invV_0 is None:
            invV_0 = torch.zeros(mu_0.shape[:-2]+(self.dim,self.dim),requires_grad=False)+torch.eye(self.dim,requires_grad=False)

        self.lambda_mu = self.lambda_mu_0
        self.mu_0 = mu_0
        self.mu = mu_0 + torch.randn(mu_0.shape,requires_grad=False)

        self.invU = Wishart(nu_0,invV_0)
        self.SExx = 0.0
        self.SEx = 0.0
        self.N = 0.0

    def to_event(self,n):
        if n ==0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.invU.to_event(n)
        return self

    def ss_update(self,SExx,SEx,N, lr=1.0, beta=None):
        # SExx is batch_shape + event_shape[:-2] + (dim,dim)
        # SEx  is batch_shape + event_shape i.e. terminal dimension are (dim,1)
        # n is batch_shape + event_shape[:-2]
        if beta is not None:
            self.SExx = beta*self.SExx + SExx
            self.SEx = beta*self.SEx + SEx
            self.N = beta*self.N + N
            SExx = self.SExx
            SEx = self.SEx
            N = self.N
        lambda_mu = self.lambda_mu_0 + N
        mu = (self.lambda_mu_0.unsqueeze(-1).unsqueeze(-1)*self.mu_0 + SEx)/lambda_mu.unsqueeze(-1).unsqueeze(-1)
#        invV = SExx + self.lambda_mu_0.unsqueeze(-1).unsqueeze(-1)*self.mu_0.unsqueeze(-1)*self.mu_0.unsqueeze(-2) - lambda_mu.unsqueeze(-1).unsqueeze(-1)*mu.unsqueeze(-1)*mu.unsqueeze(-2)
        invV = SExx + self.lambda_mu_0.unsqueeze(-1).unsqueeze(-1)*self.mu_0*self.mu_0.transpose(-2,-1) - lambda_mu.unsqueeze(-1).unsqueeze(-1)*mu*mu.transpose(-2,-1)

        self.lambda_mu = lr*lambda_mu + (1-lr)*self.lambda_mu
        self.mu = lr*mu + (1-lr)*self.mu
        self.invU.ss_update(invV,N,lr)

    def raw_update(self,X,p=None,lr=1.0,beta=None):
        # assumes X is  num_samples (Times) x batch_shape x event_dim
        # if specified p has shape num_samples x batch_shape
        # the critical manipulation here is that p averages over the batch dimension
        sample_shape = X.shape[:-self.event_dim-self.batch_dim]
        sample_dims = tuple(range(len(sample_shape)))

        if p is None:  
            SEx = X.sum(sample_dims)
            SExx = (X*X.transpose(-2,-1)).sum(sample_dims)
            N = torch.tensor(np.prod(sample_shape),requires_grad=False)
            N = N.expand(self.batch_shape + self.event_shape[:-2])

        else:  # assumes p is sample x batch_shape
            N = p.sum(sample_dims)
            p=p.view(p.shape + (1,)*self.event_dim)
            SExx = (X*X.transpose(-2,-1)*p).sum(sample_dims) 
            SEx =  (X*p).sum(sample_dims)
            
        self.ss_update(SExx,SEx,N,lr,beta)  # inputs to ss_update must be batch + event consistent

    def Elog_like(self,X):
        # X is num_samples x batch_shape x event_shape  OR  num_samples x (1,)*batch_dim x event_shape

        out = -0.5*(X.transpose(-2,-1)@self.EinvSigma()@X).squeeze(-1).squeeze(-1) + (X*self.EinvSigmamu()).sum(-2).squeeze(-1) - 0.5*(self.EXTinvUX())
        out = out + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)

        for i in range(self.event_dim-1):
            out = out.sum(-1)
        return out

    def KLqprior(self):
        KL = 0.5*(self.lambda_mu_0/self.lambda_mu - 1 + (self.lambda_mu/self.lambda_mu_0).log())*self.dim
        KL = KL + 0.5*self.lambda_mu_0*((self.mu-self.mu_0).transpose(-1,-2)@self.invU.mean()@(self.mu-self.mu_0)).squeeze(-1).squeeze(-1)
        for i in range(self.event_dim-1):
            KL = KL.sum(-1)
        KL = KL + self.invU.KLqprior()
        return KL

    def mean(self):
        return self.mu

    def EX(self):
        return self.mu
    
    def EXXT(self):
        return self.mu*self.mu.transpose(-2,-1) + self.invU.ESigma()/self.lambda_mu.unsqueeze(-1).unsqueeze(-1)

    def ESigma(self):
        return self.invU.ESigma()
        
    def ElogdetinvSigma(self):
        return self.invU.ElogdetinvSigma()

    def EinvSigmamu(self):
        return self.invU.EinvSigma()@self.mu

    def EinvSigma(self):
        return self.invU.EinvSigma()

    def EinvUX(self):
        return self.invU.EinvSigma()@self.mu

    def EXTinvUX(self):
        return self.mu.transpose(-2,-1)@self.invU.EinvSigma()@self.mu + self.dim/self.lambda_mu



