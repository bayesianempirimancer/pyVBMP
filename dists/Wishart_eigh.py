import torch

class Wishart_eigh():

    def __init__(self,event_shape, batch_shape = ()):
        assert event_shape[-1]==event_shape[-2]
        self.dim = event_shape[-1]
        self.event_dim = len(event_shape)
        self.batch_dim = len(batch_shape)
        self.batch_shape = batch_shape

        self.invU_0 = (torch.eye(self.dim,requires_grad=False)).expand(batch_shape + event_shape)
        self.nu_0 = torch.tensor(self.dim+2.0).expand(batch_shape + event_shape[:-2])

        self.d, self.v = torch.linalg.eigh(self.invU_0)
        self.logdet_invU_0 = self.d.log().sum(-1)
        self.nu = self.nu_0*(1.0+torch.rand_like(self.nu_0))

    @property
    def U(self):
        return self.v@(1.0/self.d.unsqueeze(-1)*self.v.transpose(-2,-1))

    @property
    def invU(self):
        return self.v@(self.d.unsqueeze(-1)*self.v.transpose(-2,-1))

    @property
    def logdet_invU(self):
        return self.d.log().sum(-1)

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

    def ss_update(self,SExx,n,lr=1.0,beta=None):
        if beta is None:
            beta = 1.0-lr
        idx = n>1
        SExx = SExx*(idx).unsqueeze(-1).unsqueeze(-1)
        invU = (self.invU_0 + SExx)*lr + beta*self.invU
        self.nu = (self.nu_0 + n)*lr + beta*self.nu
        self.d, self.v = torch.linalg.eigh(0.5*invU+0.5*invU.transpose(-2,-1))  # recall v@d@v.transpose(-2,-1) = invU 

    def nat_update(self,nu,invU):
        self.nu = nu
        self.d, self.v = torch.linalg.eigh(0.5*invU+0.5*invU.transpose(-2,-1))  # recall v@d@v.transpose(-2,-1) = invU 
   
    def mean(self):
        return self.U*self.nu.unsqueeze(-1).unsqueeze(-1)
    
    def meaninv(self):
        return self.invU/(self.nu.unsqueeze(-1).unsqueeze(-1) - self.dim - 1)

    def ESigma(self):
        return self.invU/(self.nu.unsqueeze(-1).unsqueeze(-1) - self.dim - 1)

    def EinvSigma(self):
        return self.U*self.nu.unsqueeze(-1).unsqueeze(-1)

    def logdetEinvSigma(self):
        return -self.logdet_invU + self.nu.log()

    def ElogdetinvSigma(self):
        return self.dim*torch.log(torch.tensor(2,requires_grad=False)) - self.logdet_invU + ((self.nu.unsqueeze(-1)  - torch.arange(self.dim))/2.0).digamma().sum(-1)

    def KLqprior(self):
        out = self.nu_0/2.0*(self.logdet_invU-self.logdet_invU_0) + self.nu/2.0*(self.invU_0*self.U).sum(-1).sum(-1) - self.nu*self.dim/2.0
        out = out + self.log_mvgamma(self.nu_0/2.0) - self.log_mvgamma(self.nu/2.0) + (self.nu - self.nu_0)/2.0*self.log_mvdigamma(self.nu/2.0) 

        for i in range(self.event_dim -2):
            out = out.sum(-1)
        return out

    def logZ(self):
        return self.log_mvgamma(self.nu/2.0) + 0.5*self.nu*self.dim*torch.log(torch.tensor(2,requires_grad=False)) - 0.5*self.nu*self.logdet_invU

