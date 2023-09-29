import torch
import dists.Dirichlet as Dirichlet
import dists.Delta as Delta
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format
import dists.NormalInverseWishart_vector_format as NormalInverseWishart_vector_format
from .MatrixNormalWishart import MatrixNormalWishart
from .MatrixNormalGamma import MatrixNormalGamma
from .MultiNomialLogisticRegression import MultiNomialLogisticRegression
from .dMixtureofLinearTransforms import dMixtureofLinearTransforms

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
        print('Chained Transformer does not batch properly yet')
        self.obs_dim = n # dimension of the observations (note that the number of observations can vary)
        self.regression_dim = p
        self.event_shape = (mixture_dim,n,p)
        self.event_dim = 3  # 
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.role_dim = role_dim
        self.mixture_dim = mixture_dim # the number of different latents

        self.x0 = NormalInverseWishart_vector_format(mu_0 = torch.zeros(n,1,requires_grad=False))
        self.W = MatrixNormalWishart(mu_0 = torch.zeros(n,n,requires_grad=False))
        self.pi = Dirichlet(torch.ones(2)*0.5)  # the idea is that pi.mean()[0] is the probability that the observation is independent of the others

    def raw_update(self,Y,iters=1,lr=1.0,beta=None,verbose=False):  # expects Y to be sample x num_y x n
        # Add another Y to the end of the mixed chain that has mean zero
        NR = Y.shape[-2]
        log_p = self.W.Elog_like(Y.unsqueeze(-2).unsqueeze(-1),Y.unsqueeze(-3).unsqueeze(-1)) + self.pi.ElogX()[1] - torch.tensor(NR-1.0,requires_grad=False).log()
        log_p = log_p*(1.0-torch.eye(NR,requires_grad=False)) + (self.x0.Elog_like(Y.unsqueeze(-1)).unsqueeze(-1)+self.pi.ElogX()[0])*torch.eye(NR,requires_grad=False)

        shift = log_p.max(-1,keepdim=True)[0]
        log_p = (log_p-shift)
        p = log_p.exp()
        logZ = p.sum(-1,keepdim=True)
        p = p/logZ
        logZ = (logZ.log() + shift).squeeze(-1)

        p0 = p.diagonal(dim1=-1,dim2=-2)
        self.x0.raw_update(Y.unsqueeze(-1),p=p0,lr=lr,beta=beta)
        p = p*(1.0-torch.eye(NR,requires_grad=False))

        N = torch.tensor(Y.shape[:-1],requires_grad=False).prod().unsqueeze(-1)
        N0 = p0.sum().unsqueeze(-1)

        self.pi.raw_update(torch.cat((N0,N-N0),dim=-1),lr=lr)
        self.W.raw_update(Y.unsqueeze(-2).unsqueeze(-1),Y.unsqueeze(-3).unsqueeze(-1),p=p,lr=lr)

