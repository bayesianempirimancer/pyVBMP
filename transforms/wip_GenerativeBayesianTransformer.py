import torch
import dists.Dirichlet as Dirichlet
import dists.Delta as Delta
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format
import dists.NormalInverseWishart_vector_format as NormalInverseWishart_vector_format
from .MatrixNormalWishart import MatrixNormalWishart
from .MatrixNormalGamma import MatrixNormalGamma
from .MultiNomialLogisticRegression import MultiNomialLogisticRegression
from .dMixtureofLinearTransforms import dMixtureofLinearTransforms

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
