import torch
from .Wishart import Wishart

class NormalInverseWishart_vector_format(): 

    def __init__(self,event_shape, batch_shape=(), scale = torch.tensor(1.0,requires_grad=False), fixed_precision = False,
                 prior_parms = {'lambda_mu' : torch.tensor(1.0,requires_grad=False), 
                                'mu' : torch.tensor(0.0,requires_grad=False),
                                'nu' : None,
                                'invU' : None}):

        assert(event_shape[-1] == 1)
        assert(len(event_shape) >= 2)
        self.dim = event_shape[-2]
        self.event_shape = event_shape
        self.event_dim = len(event_shape)
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.fixed_precision = fixed_precision

        self.lambda_mu_0 = prior_parms['lambda_mu'].expand(self.batch_shape + (self.event_dim-2)*(1,))
        self.mu_0 = prior_parms['mu'].expand(self.batch_shape + self.event_shape)
        
        self.lambda_mu = self.lambda_mu_0
        self.mu = self.mu_0 + scale*torch.randn_like(self.mu_0,requires_grad=False)

        self.invU = Wishart(event_shape = event_shape[:-2] + (self.dim,self.dim), batch_shape = batch_shape, scale=scale)
        if prior_parms['invU'] is not None and prior_parms['nu'] is not None:
            if self.invU.invU_0.shape == prior_parms['invU'].shape:
                self.invU.invU_0 = prior_parms['invU']
            else:
                print('Warning: NormalInverseWishart prior invU shape does not match Wishart invU_0 shape.  Using default.')
            if self.invU.nu_0.shape == prior_parms['nu'].shape:
                self.invU.nu_0 = prior_parms['nu']
            else:
                print('Warning: NormalInverseWishart prior nu shape does not match Wishart nu_0 shape.  Using default.')

        self.SExx = torch.tensor(0.0)
        self.SEx = torch.tensor(0.0)
        self.N = torch.tensor(0.0)

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
        invU = SExx + self.lambda_mu_0.unsqueeze(-1).unsqueeze(-1)*self.mu_0*self.mu_0.transpose(-2,-1) - lambda_mu.unsqueeze(-1).unsqueeze(-1)*mu*mu.transpose(-2,-1)

        self.lambda_mu = lr*lambda_mu + (1-lr)*self.lambda_mu
        self.mu = lr*mu + (1-lr)*self.mu
        if self.fixed_precision is False:
            self.invU.ss_update(invU,N,lr)

    def raw_update(self,X,p=None,lr=1.0,beta=None):
        sample_shape = X.shape[:-self.event_dim-self.batch_dim]
        sample_dims = tuple(range(len(sample_shape)))

        if p is None:  
            SEx = X.sum(sample_dims)
            SExx = (X*X.transpose(-2,-1)).sum(sample_dims)
            N = torch.prod(torch.tensor(sample_shape,requires_grad=False))
            N = N.expand(self.batch_shape + self.event_shape[:-2])
        else:
            N = p.sum(sample_dims)
            N = N.view(N.shape + (1,)*(self.event_dim-1))
            pv=p.view(p.shape + (1,)*self.event_dim)
            SExx = (X*X.transpose(-2,-1)*pv).sum(sample_dims) 
            SEx =  (X*pv).sum(sample_dims)
            
        self.ss_update(SExx,SEx,N,lr,beta)  # inputs to ss_update must be batch + event consistent

    def update(self,pX,p=None,lr=1.0,beta=None):
        pass

    def Elog_like(self,X):
        # X is num_samples x batch_shape x event_shape  OR  num_samples x (1,)*batch_dim x event_shape
        out = -0.5*(X.transpose(-2,-1)@self.EinvSigma()@X).squeeze(-1).squeeze(-1) + (X*self.EinvSigmamu()).sum(-2).squeeze(-1) - 0.5*(self.EXTinvUX())
        out = out + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*torch.log(2*torch.tensor(torch.pi))
        for i in range(self.event_dim-1):
            out = out.sum(-1)
        return out

    def KLqprior(self):
        KL = 0.5*(self.lambda_mu_0/self.lambda_mu - 1 + (self.lambda_mu/self.lambda_mu_0).log())*self.dim
        KL = KL + 0.5*self.lambda_mu_0*((self.mu-self.mu_0).transpose(-1,-2)@self.invU.mean()@(self.mu-self.mu_0)).squeeze(-1).squeeze(-1)
        for i in range(self.event_dim-2):
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



