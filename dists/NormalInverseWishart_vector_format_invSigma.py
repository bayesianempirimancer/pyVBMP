import torch
from utils.torch_functions import log_mvgamma, mvdigamma
from dists.Wishart import Wishart
class NormalInverseWishart_vector_format(): 

    def __init__(self, event_shape, batch_shape=(), scale=torch.tensor(1.0,requires_grad=False), fixed_precision=False,
                 prior_parms={'lambda': torch.tensor(1.0,requires_grad=False),
                              'lambda_mu' : torch.tensor(0.0,requires_grad=False),
                              'nu' : None,
                              'invU' : None},
                 parms={'lambda': torch.tensor(1.0,requires_grad=False),
                        'lambda_mu' : torch.tensor(0.0,requires_grad=False),
                        'nu' : None,
                        'invU' : None}):

        self.min_event_dim = 2
        self.dim = event_shape[-2]
        self.event_shape = event_shape
        self.event_dim = len(event_shape)
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.fixed_precision = fixed_precision

        self.lmbda = parms['lambda'].expand(self.batch_shape + self.event_shape[:-2] + (1,1))
        self.lmbda_mu = parms['lambda_mu'].expand(self.batch_shape + self.event_shape)
        self.lmbda_0 = prior_parms['lambda'].expand(self.batch_shape + self.event_shape[:-2] + (1,1))
        self.lmbda_mu_0 = prior_parms['lambda_mu'].expand(self.batch_shape + self.event_shape)
        self.invSigma = Wishart(event_shape = event_shape[:-1] + (self.dim,), batch_shape = batch_shape, scale=scale)

# Natural Parameters and associated sufficient statistics for NormalInverseWishart distribution 
# which is the conjugate prior for the multivariate normal distribution p(x|mu,Sigma).
# note that the invU and nu are natural parameters for the associated Wishart Distribution 
# but that the natural parameters for the normal inverse wishart are not quite the same.
# 
#       Natural Parameters          Sufficient Statistics           Expectations of Sufficient Statistics:  invU = xi - lmbda_mu@lmbda_mu^T/lmbda
# 
#       lmbda,                     -0.5*mu^T*invSigma*mu            -0.5*mu^T@U@mu*(nu_star+dim) + 0.5*dim/lmbda
#       lmbda_mu,                   invSigma@mu                     U@mu*(nu_star + dim)
#       nu_star = nu - dim,         0.5*logdet(invSigma)            0.5*(dim*log(2) - logdet(invU) + log_mvdigamma((nu_star+dim)/2))
#       xi = invU + lmbda*mu*mu^T, -0.5*invSigma                    -0.5*U*(nu_star+dim) 
#
#       Res = -logZ = 0.5*dim*log(lmbda) - 0.5*dim*(nu+1)*log(2*pi) + 0.5*nu*logdet(invU) - log(Gamma_dim(nu/2)) 

    @property
    def parms(self):
        return {'lambda' : self.lmbda, 'lambda_mu' : self.lmbda_mu, 'nu' : self.nu, 'invU' : self.invU}

    @property
    def prior_parms(self):
        return {'lambda' : self.lmbda_0, 'lambda_mu' : self.lmbda_mu_0, 'nu' : self.nu, 'xi' : self.invU}

    @property
    def mu_0(self):
        return self.lmbda_mu_0/self.lmbda_0
    
    @property
    def mu(self):  # Posterior mean
        return self.lmbda_mu/self.lmbda

    @property
    def nu(self):
        return self.invSigma.nu

    @property
    def invU(self):
        return self.invSigma.invU

    @property
    def nu_0(self):
        return self.invSigma.nu_0
    
    @property
    def invU_0(self):
        return self.invSigma.invU_0

    @property
    def U(self):
        return self.invSigma.U

    @property 
    def logdet_invU(self):
        return self.invSigma.logdet_invU

    def to_event(self,n):
        if n ==0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        return self

    def ss_update(self,SExx,SEx,N, lr=1.0, beta=0.0):
        # update natural parameters

        if beta>0.0:
            SEx = SEx + beta*(self.lmbda_mu - self.lmbda_mu_0)
            N = N + beta*(self.lmbda - self.lmbda_0)
            if self.fixed_precision is False:
                SExx = SExx + beta*(self.invU-self.invU_0 + self.lmbda_mu@self.mu.transpose(-2,-1) - self.lmbda_mu_0@self.mu_0.transpose(-2,-1))

        self.lmbda = (1-lr)*self.lmbda + lr*(self.lmbda_0 + N)
        self.lmbda_mu = (1-lr)*self.lmbda_mu + lr*(self.lmbda_mu_0 + SEx)
        
        if self.fixed_precision is False:
            SExx = SExx - self.lmbda_mu@self.mu.transpose(-2,-1) + self.lmbda_mu_0@self.mu_0.transpose(-2,-1)
            self.invSigma.ss_update(SExx,N,lr=lr,beta=0.0)

        # update and store computationally expensive expectations bits
#        self.set_expectation_parameters()
#
#    def set_expectation_parameters(self):
#        self.U = self.invU.inverse()
#        self.logdet_invU = -self.U.logdet().unsqueeze(-1).unsqueeze(-1)

    def raw_update(self,X,p=None,lr=1.0,beta=0.0):   # X is sample_shape + batch_shape x event_shape  OR  sample_shape + (1,)*batch_dim + event_shape
                                                     # if specified p is sample_shape + batch_shape
        sample_shape = X.shape[:-self.event_dim-self.batch_dim]
        sample_dims = tuple(range(len(sample_shape)))

        if p is None:  
            SEx = X.sum(sample_dims)
            SExx = (X*X.transpose(-2,-1)).sum(sample_dims)
            N = torch.prod(torch.tensor(sample_shape,requires_grad=False))
            N = N.expand(self.batch_shape + self.event_shape[:-2])
        else:
            p=p.view(p.shape + (1,)*self.event_dim)  # expand p to be batch + event consistent with X
            N = p.sum(sample_dims)
            SExx = (X*X.transpose(-2,-1)*p).sum(sample_dims) 
            SEx =  (X*p).sum(sample_dims)
            
        self.ss_update(SExx,SEx,N,lr,beta)  # inputs to ss_update must be batch + event consistent

    def update(self,pX,p=None,lr=1.0,beta=0.0):
        sample_shape = pX.mean().shape[:-self.event_dim-self.batch_dim]
        sample_dims = tuple(range(len(sample_shape)))

        if p is None:  
            SEx = pX.mean().sum(sample_dims)
            SExx = (pX.EXXT()).sum(sample_dims)
            N = torch.prod(torch.tensor(sample_shape,requires_grad=False))
            N = N.expand(self.batch_shape + self.event_shape[:-2])
        else:
            p=p.view(p.shape + (1,)*self.event_dim)
            N = p.sum(sample_dims)
            SExx = (pX.EXXT()*p).sum(sample_dims) 
            SEx =  (pX.mean()*p).sum(sample_dims)

        self.ss_update(SExx,SEx,N,lr,beta)  # inputs to ss_update must be batch + event consistent

    def Elog_like(self,X):
        # X is num_samples x batch_shape x event_shape  OR  num_samples x (1,)*batch_dim x event_shape
        out = -0.5*(X.transpose(-2,-1)@self.EinvSigma()@X) + (X*self.EinvSigmamu()).sum(-2,True) - 0.5*(self.EXTinvUX())
        out = out + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*torch.log(2*torch.tensor(torch.pi))
        return out.sum(list(range(-self.event_dim,0)))  # eliminate event dimensions

    def KLqprior_Wishart(self):
        return self.invSigma.KLqprior()

    def KLqprior(self):
        KL = 0.5*(self.lmbda_0/self.lmbda - 1 + (self.lmbda/self.lmbda_0).log())*self.dim
        KL = KL + 0.5*self.lmbda_0*((self.mu-self.mu_0).transpose(-1,-2)@self.EinvSigma()@(self.mu-self.mu_0))
        KL = KL + self.KLqprior_Wishart()
        return KL.sum(list(range(-self.event_dim,0)))  # eliminate event dimensions

    def mean(self):
        return self.lmbda_mu/self.lmbda
    
    def EX(self):
        return self.lmbda_mu/self.lmbda
    
    def EXXT(self):
        return self.mu*self.mu.transpose(-2,-1) + self.ESigma()/self.lmbda_mu

    def EinvSigma(self):
        return self.U*self.nu

    def ESigma(self):
        return self.invU/(self.nu - self.dim - 1)
        
    def ElogdetinvSigma(self):
        return self.dim*torch.log(torch.tensor(2,requires_grad=False)) - self.logdet_invU + mvdigamma(self.nu/2.0,self.dim)

    def logdetEinvSigma(self):
        return -self.logdet_invU + self.nu.log()

    def EinvSigmamu(self):
        return self.EinvSigma()@self.mu

    def EinvUX(self):
        return self.EinvSigma()@self.mu

    def EXTinvUX(self):
        return self.mu.transpose(-2,-1)@self.EinvSigma()@self.mu + self.dim/self.lmbda
    
    def EXmMUTinvUXmMU(self):
        return self.dim/self.lmbda

    def logZ(self):
        out = -0.5 * self.dim * self.lmbda.log() + 0.5*self.dim*torch.log(torch.tensor(2 * torch.pi, requires_grad=False))
        out = out + 0.5*self.nu*self.dim*torch.log(torch.tensor(2.0,requires_grad=False)) - 0.5*self.nu*self.logdet_invU
        out = out + log_mvgamma(self.nu / 2.0, self.dim)
        return out.sum(list(range(-self.event_dim, 0)))

    def logZ_p(self):
        out = -0.5 * self.dim * self.lmbda_0.log() + 0.5*self.dim*torch.log(torch.tensor(2 * torch.pi, requires_grad=False))
        out = out + 0.5*self.nu_0*self.dim*torch.log(torch.tensor(2.0,requires_grad=False)) - 0.5*self.nu_0*self.logdet_invU_0
        out = out + log_mvgamma(self.nu_0 / 2.0, self.dim)
        return out.sum(list(range(-self.event_dim, 0)))


import dists.Mixture
class GMM_vector(dists.Mixture):
    def __init__(self,nc,dim):
        parms = {'lmbda' : torch.tensor(1.0,requires_grad=False).expand((nc,1,1)), 
                 'lmbda_mu' : torch.zeros((nc,dim,1),requires_grad=False), 
                 'nu_star' : torch.tensor(2.0,requires_grad=False).expand((nc,1,1)), 
                 'xi' : torch.eye(dim,requires_grad=False).expand((nc,dim,dim))}
        dist = NormalInverseWishart_vector_format(parms)
        super().__init__(dist,(nc,))

    def initialize(self,data):
        idx = torch.randint(data.shape[0],self.dist.batch_shape)
        self.dist.lmbda_mu = data[idx]/self.dist.lmbda


    
