# __author__ = "Jeff Beck"
# __copyright__ = "Copyright 2023, Neurotics Inc."
# __credits__ = ["Jeff Beck"]
# __license__ = "GPL"
# __version__ = "0.0.1"
# __maintainer__ = "Jeff Beck"
# __email__ = "bayesian.empirimancer@gmail.com"
# __status__ = "Development"
# 
# Discription:
# Multivariate Normal distribution with ARD prior on the precision matrix
#   Tested Routines:
#       ss_update
#       raw_update 
#       KLqprior
#
#   Missing Routines:
#       update
#       sample       

import torch
import numpy as np
from .Gamma import Gamma

class MVN_ard():
    def __init__(self,dim,batch_shape=(),scale=1):

        self.dim = dim
        self.event_dim = 2
        self.event_shape = (dim,1)
        self.batch_shape = batch_shape
        self.batch_dim = len(self.batch_shape)
        self.mu = torch.randn(batch_shape + (dim,1),requires_grad=False)*scale
        self.invSigma = torch.zeros(batch_shape + (dim,dim),requires_grad=False) + torch.eye(dim,requires_grad=False)/scale**2
        self.Sigma = self.invSigma
        self.logdetinvSigma = self.invSigma.logdet()
        self.invSigmamu = self.invSigma@self.mu
        self.alpha = Gamma(torch.ones(batch_shape+(dim,),requires_grad=False),torch.ones(batch_shape+(dim,),requires_grad=False)*scale**2)
        self.SEx = 0.0
        self.SExx = 0.0

    def to_event(self,n):
        if n == 0: 
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        return self

    def ss_update(self,SExx,SEx, iters = 1, lr=1.0, beta=None):
        if beta is not None:
            self.SExx = self.SExx*beta + SExx
            self.SEx = self.SEx*beta + SEx
            SExx = self.SExx
            SEx = self.SEx

        invSigmamu = SEx
        invSigma =  SExx + self.alpha.mean().unsqueeze(-1)*torch.eye(self.dim,requires_grad=False) + 1e-6*torch.eye(self.dim,requires_grad=False)
        Sigma = invSigma.inverse()
        mu = Sigma@self.invSigmamu
        for i in range(iters):
            EXXT = Sigma.diagonal(dim1=-1,dim2=-2) + (mu**2).squeeze(-1)
            self.alpha.ss_update(torch.tensor(0.5).expand(self.alpha.batch_shape + self.alpha.event_shape),
                                 0.5*EXXT,lr=lr,beta=beta)
            invSigma =  SExx + self.alpha.mean().unsqueeze(-1)*torch.eye(self.dim,requires_grad=False)
            Sigma = self.invSigma.inverse()
            mu = self.Sigma@self.invSigmamu
        
        self.invSigma = (1-lr)*self.invSigma + lr*invSigma
        self.invSigmamu = (1-lr)*self.invSigmamu + lr*invSigmamu
        self.Sigma = self.invSigma.inverse()
        self.mu = self.Sigma@self.invSigmamu

        self.logdetinvSigma = self.invSigma.logdet()

    def raw_update(self,X,p=None,lr=1.0,beta=None):  # assumes X is a vector and p is sample x batch 

        if p is None:  
            SEx = X
            SExx = X@X.transpose(-2,-1)
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape + self.event_shape[:-2])
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
            self.ss_update(SExx,SEx,n,lr,beta)  # inputs to ss_update must be batch + event consistent

        else:  # data is shape sample_shape x batch_shape x event_shape with the first batch dimension having size 1

            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SExx = X@X.transpose(-2,-1)*p
            SEx =  X*p
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
                p = p.sum(0)      
            self.ss_update(SExx,SEx,p.squeeze(-1).squeeze(-1),lr,beta)  # inputs to ss_update must be batch + event consistent
            # p now has shape batch_shape + event_shape so it must be squeezed by the default event_shape which is 1

    def KLqprior(self):
        KL = 0.5*(self.mu.pow(2).squeeze(-1)*self.alpha.mean()).sum(-1) - 0.5*self.alpha.loggeomean().sum(-1) + 0.5*self.ElogdetinvSigma()
        KL = KL + self.alpha.KLqprior().sum(-1)        
        for i in range(self.event_dim-2):
            KL = KL.sum(-1)
        return KL

    def mean(self):
        return self.mu
    
    def ESigma(self):
        return self.Sigma

    def EinvSigma(self):
        return self.invSigma
    
    def EinvSigmamu(self):
        return self.invSigmamu

    def ElogdetinvSigma(self):
        return self.logdetinvSigma

    def EX(self):
        return self.mean()

    def EXXT(self):
        return self.ESigma() + self.mean()@self.mean().transpose(-2,-1)

    def EXTX(self):
        return self.ESigma().sum(-1).sum(-1) + self.mean().pow(2).sum(-2).squeeze(-1)

    def EXTinvUX(self):
        return (self.mean().transpose(-2,-1)@self.EinvSigma()@self.mean()).squeeze(-1).squeeze(-1)

    def Res(self):
        return - 0.5*(self.mean()*self.EinvSigmamu()).sum(-1).sum(-1) + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)


