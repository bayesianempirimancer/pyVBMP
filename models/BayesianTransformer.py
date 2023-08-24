import torch
from .dists import Dirichlet, MatrixNormalWishart, MultivariateNormal_vector_format, Delta, MVN_ard
from .dists import MatrixNormalWishart
from .MultiNomialLogisticRegression import *
from .dMixtureofLinearTransforms import *

class ChainedBayesianTransformer():
    # Here the idea is that the different observations form a chain but are out of order
    # so the goal is to determine which (if any of the other observations drives another observation)
    # i.e. the conditional independence relationship is p(y_i|y_k, z_i=k) with teh special case that 
    # z_i = -1 means that y_i is independent of all other observations.  Once again we would like 
    # to have roles so that the relationships between the y_i's can be different.
    # 
    # This principle problem with this approach is circularity.      

    # This is a simple version of the transformer that has a feedforward flavor rather than a generative flavor
    # Here we have observations X and Y where a subset of X's are used to predict a subset of Y's.  The idea is that
        # X has shape sample x num_obs x p
        # Y has shape sample x n

    # but for each sample there is a different relationship betweeen each of the num_obs X's and Y.  This is modeled by defining 
    # roles where each of the num_obs X's.

    def __init__(self, mixture_dim, role_dim, n, p, batch_shape = (), pad_X=False):
        self.obs_dim = n # dimension of the observations (note that the number of observations can vary)
        self.regression_dim = p
        self.event_shape = (mixture_dim,n,p)
        self.event_dim = 3  # 
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.role_dim = role_dim
        self.mixture_dim = mixture_dim # the number of different latents

        self.W = MatrixNormalWishart(mu_0 = torch.zeros(n,n,requires_grad=False))
        self.pi = Dirichlet(torch.ones(2)*0.5)

    def raw_update(self,Y,iters=1,lr=1.0,verbose=False):
        print('Not Working')
        nr = Y.shape[-2]
        log_p = self.W.Elog_like(Y.unsqueeze(-2).unsqueeze(-1),Y.unsqueeze(-3).unsqueeze(-1)) 
        log_p = log_p*(1.0-torch.eye(nr,requires_grad=False)) + torch.eye(nr,requires_grad=False)*self.pi.ElogX()[...,0]
        log_p = log_p + torch.eye(nr,requires_grad=False)*(self.pi.ElogX()[...,0]-self.pi.ElogX()[...,1]) 
        log_p = log_p + torch.ones(nr,nr,requires_grad=False)*self.pi.ElogX()[...,1]

        shift = log_p.max(-1,keepdim=True)[0]
        log_p = (log_p-shift)
        self.p = log_p.exp()
        logZ = self.p.sum(-1,keepdim=True)
        self.p = self.p/logZ
        logZ = (logZ.log() + shift).squeeze(-1)

        NA0 = self.p.diagonal(dim1=-2,dim2=-1).sum((0,-1)).unsqueeze(-1)
        NA1 = self.p.sum((-1,-2,0)).unsqueeze(-1) - self.NA0
        self.NA = torch.cat((NA0,NA1),-1)

        self.pi.raw_update(self.NA,lr=lr)
        self.W.raw_update(Y.unsqueeze(-2).unsqueeze(-1),Y.unsqueeze(-3).unsqueeze(-1),p=self.p,lr=lr)





class FocusedGenerativeBayesianTransformer():
    # Here the idea is that roles should be selected based upon the hidden latent, i.e. p(role|x_k,z=k) is MNLR
    # This is accomplished by simply doing a mixture of directed mixture of linear transforms with the latent x_k's computed 
    # by averaging over the observations given the assignments

    def __init__(self, mixture_dim, role_dim, obs_dim, hidden_dim, batch_shape = (), pad_X=False):
        self.obs_dim = obs_dim # dimension of the observations (note that the number of observations can vary)
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 3  # 
        self.role_dim = role_dim
        self.hidden_dim = hidden_dim
        self.mixture_dim = mixture_dim # the number of different latents

        self.W = dMixtureofLinearTransforms(obs_dim,hidden_dim,role_dim,batch_shape = (mixture_dim,),pad_X=pad_X)
        self.pi = Dirichlet(torch.ones(mixture_dim,requires_grad=False))        

        self.p = None
        self.pX = None
        self.ELBO_last = -torch.tensor(torch.inf)

    def update_assignments(self,Y):
        # Here self.p will be the assignment probability for each observation Y to a single latent X
        # This is ditinct from the generative version where self.p also includes the role assignment
        if self.pX is None:
            self.pX = MultivariateNormal_vector_format(invSigma = torch.eye(self.hidden_dim,requires_grad=False), invSigmamu = torch.zeros(1,self.mixture_dim,self.hidden_dim,1,requires_grad=False))

        log_p = self.W.Elog_like_given_pX_pY(self.pX,Delta(Y.unsqueeze(-1).unsqueeze(-3))) + self.pi.ElogX()
        shift = log_p.max(-1,keepdim=True)[0]
        log_p = (log_p-shift)
        logZ = log_p.logsumexp(-1) + shift.squeeze(-1)
        log_p = log_p.exp()
        log_p = log_p/log_p.sum(-1,keepdim=True)
        self.p = log_p 
        self.NA = self.p.sum((0,-2))
#        return logZ.logsumexp(-1)  # returns the ELBO contrib for each data point Y

    def update_latents(self,Y):
        if self.p is None: 
            self.update_assignments(Y)                  
        pX, Res = self.W.postdict(Y.unsqueeze(-2))[0:2]
        pv = self.p.view(self.p.shape+(1,1))
        self.pX = MultivariateNormal_vector_format(invSigma = (pX.EinvSigma()*pv).sum(-4,True) + torch.eye(self.hidden_dim,requires_grad=False),
                                                   invSigmamu = (pX.EinvSigmamu()*pv).sum(-4,True))
        Res = (Res*self.p).sum(-2,True)
        Warning('Doublecheck Res computation for dMixtureofLinearTransforms')
        return Res - self.pX.Res()  # returns the ELBO contrib for each data point Y

    def update_parms(self,Y,lr=1.0):
        self.W.update(self.pX,Delta(Y.unsqueeze(-1).unsqueeze(-3)),p=self.p,lr=lr)
        self.pi.raw_update(self.NA,lr=lr)

    def raw_update(self,Y,iters=1,lr=1.0,verbose=False):
        for i in range(iters):
            self.update_assignments(Y)
            ELBO = self.update_latents(Y).sum()
            idx = self.p>0
            ELBO = ELBO - (self.p[idx]*self.p[idx].log()).sum() + (self.NA*self.pi.ElogX()).sum()- self.KLqprior()            
            if verbose:
                print('Focused Transformer Percent Change in  ELBO: ', (ELBO-self.ELBO_last)/self.ELBO_last.abs()*100)
            self.ELBO_last = ELBO
            self.update_parms(Y,lr=lr)

    def Elog_like(self,Y):
        return self.update_latents(Y).sum(-1)

    def KLqprior(self):
        return self.W.KLqprior().sum(-1) + self.pi.KLqprior()


class GenerativeBayesianTransformer():
    # The logic of the Bayesian Transformer is that observations, Y with size (num_obs,obs_dim), are probabilistically 
    # clustered into mixture_dim groups that have different relationships to the latent, X.  In generative modeling
    # terms, p(y_i|x,z_i) gives the probability of observation x_i given the output y and the latent assignment 
    # z_i \in {1,...,mixture_dim}.  

    def __init__(self, mixture_dim, role_dim, obs_dim, hidden_dim, batch_shape = (), pad_X=False):
        self.obs_dim = obs_dim # dimension of the observations (note that the number of observations can vary)
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 3  # 
        self.role_dim = role_dim
        self.hidden_dim = hidden_dim
        self.mixture_dim = mixture_dim # the number of different latents

        self.A = MatrixNormalWishart(mu_0 = torch.zeros(mixture_dim, role_dim, obs_dim, hidden_dim,requires_grad=False),
            U_0 = torch.eye(obs_dim,requires_grad=False).view(batch_shape + (1,1,obs_dim,obs_dim))*(role_dim*mixture_dim)**2,
            pad_X=pad_X)
        self.pi_role = Dirichlet(torch.ones(mixture_dim, role_dim,requires_grad=False))
        self.pi_mix = Dirichlet(torch.ones(mixture_dim,requires_grad=False))

        self.p = None
        self.pX = None
        self.ELBO_last = -torch.tensor(torch.inf)

    def update_assignments(self,Y):
        if self.pX is None:
            self.pX = MultivariateNormal_vector_format(invSigma = torch.eye(self.hidden_dim,requires_grad=False), invSigmamu = torch.zeros(self.hidden_dim,1,requires_grad=False))

        log_p = self.A.Elog_like_given_pX_pY(self.pX,Delta(Y.unsqueeze(-1).unsqueeze(-3).unsqueeze(-3))) + self.pi_role.ElogX() + self.pi_mix.ElogX().unsqueeze(-1)
        shift = log_p.max(-1,keepdim=True)[0].max(-2,keepdim=True)[0]
        log_p = (log_p-shift)
        logZ = log_p.logsumexp((-1,-2)) + shift.squeeze(-1).squeeze(-1)
        log_p = log_p.exp()
        log_p = log_p/log_p.sum((-1,-2),keepdim=True)
        self.p = log_p
        self.NA = self.p.sum((0,-3))
        return logZ

    def update_latents(self,Y):
        if self.p is None:
            self.update_assignments(Y)            
        invSigma, invSigmamu, Res = self.A.Elog_like_X(Y.unsqueeze(-1).unsqueeze(-3).unsqueeze(-3))
        pv = self.p.view(self.p.shape+(1,1))

        invSigma = (invSigma*pv).sum((-3,-5),True) + torch.eye(self.hidden_dim,requires_grad=False)
        invSigmamu = (invSigmamu*pv).sum((-3,-5),True)
        Res = (Res*self.p).sum((-3,-1),True)

        self.pX = MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu)
        return Res - self.pX.Res()
    
    def update_parms(self,Y,lr=1.0):
        self.A.update(self.pX,Delta(Y.unsqueeze(-1).unsqueeze(-3).unsqueeze(-3)),p=self.p,lr=lr)
        self.pi_role.raw_update(self.NA,lr=lr)
        self.pi_mix.raw_update(self.NA.sum(-1),lr=lr)

    def raw_update(self, Y, iters = 1, latent_iters = 1, lr=1.0,verbose=False):
        for i in range(iters):
            for j in range(latent_iters):
                self.update_assignments(Y)
                ELBO = self.update_latents(Y).sum()
            idx = self.p>0
            ELBO = ELBO - (self.p[idx]*self.p[idx].log()).sum() + (self.NA*(self.pi_mix.ElogX().unsqueeze(-1) + self.pi_role.ElogX())).sum()- self.KLqprior()
            self.update_parms(Y,lr=lr)
            if verbose:
                print('GBT Percent Change in  ELBO: ', (ELBO-self.ELBO_last)/self.ELBO_last.abs())
            self.ELBO_last = ELBO

    def KLqprior(self):
        return self.A.KLqprior().sum((-1,-2)) + self.pi_mix.KLqprior() + self.pi_role.KLqprior().sum(-1)

    def Elog_like(self,Y):
        return self.update_latents(Y).sum(-1)

    def postdict(self,Y,iters=0):
        for i in range(iters):        
            self.update_assignments(Y)
            self.update_latents(Y)
        return self.pX

# mixture_dim = 8
# role_dim = 4
# obs_dim = 2
# hidden_dim = 2

# model = GenerativeBayesianTransformer(mixture_dim, role_dim, obs_dim, hidden_dim, batch_shape = (), pad_X=True)
# self = model
# num_samples = 400
# num_obs = 10
# Y = 4*torch.randn(num_samples, num_obs, obs_dim)*torch.rand(num_samples,num_obs,1)

# model.update(Y,iters=10,lr=1)
