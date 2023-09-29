# Implements Wishart distribution and associated natural parameter updates.  This could be made more memory efficient by
# using the eigenvalue decomposition for all calculation instead so simultaneously storing invU and U.  Which is to say that 
# currently it uses 3x more memory than is really needed.  We could fix this by replacing invU and U with @property methods 
# that compute them using invU = self.v@(self.d.unsqueeze(-1)*self.v.transpose(-2,-1)) and U = self.v@(1.0/self.d.unsqueeze(-1)*self.v.transpose(-2,-1))

import torch
class Wishart():

    def __init__(self,event_shape, batch_shape = (), scale = torch.tensor(1.0,requires_grad=False)):
        assert event_shape[-1]==event_shape[-2]
        self.dim = event_shape[-1]
        self.event_shape = event_shape
        self.event_dim = len(event_shape)
        self.batch_dim = len(batch_shape)
        self.batch_shape = batch_shape

        self.invU_0 = (scale**2*torch.eye(self.dim,requires_grad=False)).expand(batch_shape + event_shape)
        self.nu_0 = torch.tensor(self.dim+2.0).expand(batch_shape + event_shape[:-2])

        self.logdet_invU_0 = self.invU_0.logdet()
        self.invU = self.invU_0
        self.U = self.invU.inverse()
        self.nu = self.nu_0
        self.logdet_invU = self.invU.logdet()
        self.SExx = 0.0
        self.N = 0.0

    def to_event(self,n):
        if n ==0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape  
        self.batch_shape = self.batch_shape[:-n]
        return self

    def log_mvgamma(self,nu):
        return (nu.unsqueeze(-1) - torch.arange(self.dim)/2.0).lgamma().sum(-1)

    def log_mvdigamma(self,nu):
        return (nu.unsqueeze(-1) - torch.arange(self.dim)/2.0).digamma().sum(-1)

    def ss_update(self,SExx,N,lr=1.0,beta=None):
        assert(SExx.ndim == self.batch_dim + self.event_dim)
        assert(N.ndim == self.batch_dim + self.event_dim - 2)
        if beta is not None:
            self.SExx = SExx + beta*self.SExx
            self.N = N + beta*self.N
            SExx = self.SExx
            N = self.N
#        idx = N>0.5
#        SExx = SExx*(idx).unsqueeze(-1).unsqueeze(-1)
        self.invU = lr*(self.invU_0 + SExx) + (1.0-lr)*self.invU
        self.nu = lr*(self.nu_0 + N) + (1.0-lr)*self.nu
        self.U = self.invU.inverse()
        self.logdet_invU = self.invU.logdet()

        # idx = ~(self.logdet_invU>self.logdet_invU_0)
        # if idx.sum()>0:
        #     print('Wishart ss_update hack triggered at',idx.sum(),'locations')
        #     print(idx)
        #     self.invU[idx] = self.invU_0[idx]
        #     self.U[idx] = self.invU_0[idx].inverse()
        #     self.nu[idx] = self.nu_0[idx]
        #     self.logdet_invU[idx] = self.logdet_invU_0[idx]

    def mean(self):
        return self.U*self.nu.view(self.nu.shape + (1,1))
    
    def meaninv(self):
        return self.invU/(self.nu.view(self.nu.shape + (1,1)) - self.dim - 1)

    def ESigma(self):
        return self.invU/(self.nu.view(self.nu.shape + (1,1)) - self.dim - 1)

    def EinvSigma(self):
        return self.U*self.nu.view(self.nu.shape + (1,1))

    def ElogdetinvSigma(self):
        return self.dim*torch.log(torch.tensor(2,requires_grad=False)) - self.logdet_invU + self.log_mvdigamma(self.nu/2.0)

    def KLqprior(self):
        out = self.nu_0/2.0*(self.logdet_invU-self.logdet_invU_0) + self.nu/2.0*(self.invU_0*self.U).sum(-1).sum(-1) - self.nu*self.dim/2.0
        out = out + self.log_mvgamma(self.nu_0/2.0) - self.log_mvgamma(self.nu/2.0) + (self.nu - self.nu_0)/2.0*self.log_mvdigamma(self.nu/2.0) 

        for i in range(self.event_dim -2):
            out = out.sum(-1)
        return out

    def logZ(self):
        return self.log_mvgamma(self.nu/2.0) + 0.5*self.nu*self.dim*torch.log(torch.tensor(2,requires_grad=False)) - 0.5*self.nu*self.logdet_invU
