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
from .Gamma import Gamma
class MVN_ard():
    def __init__(self,event_shape,batch_shape=(),scale=1.0, pad_X = False):
        print('MVN_ard does not use tau parameter,  need to implement GammaGlobalLocal distribution')
        assert event_shape[-1] == 1
        self.dim = event_shape[-2]
        self.event_dim = len(event_shape)
        self.event_shape = event_shape
        self.batch_shape = batch_shape
        self.batch_dim = len(self.batch_shape)
        self.mu = torch.randn(batch_shape + event_shape,requires_grad=False)*scale
        self.invSigma = torch.zeros(batch_shape + event_shape[:-1] + (self.dim,),requires_grad=False) + torch.eye(self.dim,requires_grad=False)/scale**2
        self.Sigma = self.invSigma
        self.logdetinvSigma = self.invSigma.logdet()
        self.invSigmamu = self.invSigma@self.mu
        self.alpha = Gamma(event_shape, batch_shape, prior_parms = {'alpha' : torch.tensor(0.5), 'beta' : torch.tensor(0.5*scale**2)})
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

    def ss_update(self, SExx, SEx, iters = 2, lr=1.0, beta=None):
        if beta is not None:
            self.SExx = self.SExx*beta + SExx
            self.SEx = self.SEx*beta + SEx
            SExx = self.SExx
            SEx = self.SEx

        invSigmamu = SEx
        invSigma =  SExx + self.alpha.mean()*torch.eye(self.dim,requires_grad=False) + 1e-6*torch.eye(self.dim,requires_grad=False)
        Sigma = invSigma.inverse()
        mu = Sigma@self.invSigmamu
        for i in range(iters):
            EXXT = Sigma.diagonal(dim1=-1,dim2=-2).unsqueeze(-1) + (mu**2)
            self.alpha.ss_update(torch.tensor(0.5).expand(self.alpha.batch_shape+self.alpha.event_shape), 0.5*EXXT,lr=lr,beta=beta)
            invSigma =  SExx + self.alpha.mean()*torch.eye(self.dim,requires_grad=False)
            Sigma = invSigma.inverse()
            mu = Sigma@invSigmamu
        
        self.invSigma = (1-lr)*self.invSigma + lr*invSigma
        self.invSigmamu = (1-lr)*self.invSigmamu + lr*invSigmamu
        self.Sigma = self.invSigma.inverse()
        self.mu = self.Sigma@self.invSigmamu

        self.logdetinvSigma = self.invSigma.logdet()

    def KLqprior(self):
        KL = 0.5*(self.mu.pow(2)*self.alpha.mean()).sum(list(range(-self.event_dim,0))) 
        KL = KL - 0.5*self.alpha.loggeomean().sum(list(range(-self.event_dim,0)))  + 0.5*self.ElogdetinvSigma().sum(list(range(2-self.event_dim,0))) 
        KL = KL + (self.Sigma.diagonal(dim1=-1,dim2=-2)*self.alpha.mean().squeeze(-1)).sum(list(range(1-self.event_dim,0))) 
        KL = KL + self.alpha.KLqprior()       
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
        print('MVN_ard:  Res function does not include effects of alpha')
        return - 0.5*(self.mean()*self.EinvSigmamu()).sum(-1).sum(-1) + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*torch.log(2*torch.tensor(torch.pi,requires_grad=False))


