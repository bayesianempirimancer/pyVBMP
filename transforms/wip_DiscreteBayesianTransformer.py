import torch
import dists.Dirichlet as Dirichlet
import dists.Delta as Delta
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format
import dists.NormalInverseWishart_vector_format as NormalInverseWishart_vector_format
from .MatrixNormalWishart import MatrixNormalWishart
from .MatrixNormalGamma import MatrixNormalGamma
from .MultiNomialLogisticRegression import MultiNomialLogisticRegression
from .dMixtureofLinearTransforms import dMixtureofLinearTransforms
from utils.torch_functions import *


class DiscreteBayesianTransformer():
    # Here the idea is that observations are NormalInverseWishart conditioned on a what and where latent as well 
    # as an observation specific location variable.  


    # Here the idea is that roles should be selected based upon the hidden latent, i.e. p(role|x_k,z=k) is MNLR
    # This is accomplished by simply doing a mixture of directed mixture of linear transforms with the latent x_k's computed 
    # by averaging over the observations given the assignments.  
    # This routine expects Y.shape = sample_shape x batch_shape x num_obs x event_shape

    def __init__(self, mixture_dim, role_dim, obs_dim, hidden_dim, batch_shape = (), pad_X=False):
        self.obs_dim = obs_dim # dimension of the observations (note that the number of observations can vary)
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 3  # 
        self.role_dim = role_dim
        self.hidden_dim = hidden_dim
        self.mixture_dim = mixture_dim # the number of different latents

        self.W = dMixtureofLinearTransforms(obs_dim,hidden_dim,role_dim,batch_shape = (mixture_dim,),pad_X=pad_X)
                # sample x mixture_Dim x role_dim x hidden_dim x (obs_dims,1)
                # so that W.Elog_like has shape mixture_dim
        self.pi = Dirichlet((mixture_dim,))        


        self.p = None
        self.pX = None
        self.ELBO_last = -torch.tensor(torch.inf)

    def update_assignments(self,Y):
        # Here self.p will be the assignment probability for each observation Y to a single latent X
        # This is ditinct from the generative version where self.p also includes the role assignment
        if self.pX is None:
            self.pX = MultivariateNormal_vector_format(invSigma = torch.eye(self.hidden_dim,requires_grad=False), invSigmamu = torch.zeros(1,self.mixture_dim,self.hidden_dim,1,requires_grad=False))

        log_p = self.W.Elog_like_given_pX_pY(self.pX,Delta(Y.unsqueeze(-1).unsqueeze(-3))) + self.pi.ElogX()
        # shift = log_p.max(-1,keepdim=True)[0]
        # log_p = (log_p-shift)
        # logZ = log_p.logsumexp(-1) + shift.squeeze(-1)
        # log_p = log_p.exp()
        # log_p = log_p/log_p.sum(-1,keepdim=True)
        # self.p = log_p 
        logZ = stable_logsumexp(log_p,-1,True)
        self.p = (log_p - logZ).exp()
        logZ = logZ.squeeze(-1)
        self.NA = self.p.sum((0,-2))  # The -2 sums over the observations
#        return logZ.logsumexp(-1)  # returns the ELBO contrib for each data point Y

    def update_latents(self,Y):
        if self.p is None: 
            self.update_assignments(Y)                  
        pX, Res = self.W.postdict(Y.unsqueeze(-2))[0:2]
        pv = self.p.view(self.p.shape+(1,1))
        self.pX = MultivariateNormal_vector_format(invSigma = (pX.EinvSigma()*pv).sum(-4,True) + torch.eye(self.hidden_dim,requires_grad=False),
                                                   invSigmamu = (pX.EinvSigmamu()*pv).sum(-4,True))
        # Use of torch.eye in the above line implements a unit normal prior on the X latents...

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


