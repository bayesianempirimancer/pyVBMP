# Variational Bayesian Expectation Maximization for Tensor Normal Distribution with Wishart Prior on the 
# Covariance matrix, which is assumed to be an outer product of covariance matrices 
# that are tensor dimsnion specific, i.e. cov(X_i,j,k,...;X_i',j',k',... ) = Sigma_i,i' * Sigma_j,j' *
# This is a generalization of the matrix normal distribution to tensors with a significant computational 
# advantage obtained by exploiting the Kronecker product structure of the covariance matrix.  
#
# For example if you have a large vector with dimension D = n1*n2*n3*...n_K then the associated covariance 
# matrix you have to represent (and invert) is D x D = n1**2 * n2**2 * ....
# Tensorizing the vector into a tensor of shape (n1,n2,...,n_k) and then using this class to represent the covariance
# Changes this to K covariance matrices of of shape (n1,n1), (n2,n2), ... (n_k,n_k)
# This changes the computationl cost of inference from O(n1**3 * n2**3 * ...) to O(n1**3 + n2**3 + ... + n_k**3)
#
# Since the full covariance matrix is a product of tensor dimension specific covariance matrices, the scales of the 
# dimension specific covariance matrices are massively conflated.  We resolve this issue by using Langrange multipliers to 
# set the scale of the tensor dimension specifice covarince matrices and then use a gamma prior to set the overall scale.  
# The specific constraint used enforces <logdet(C)> = 0 for the different covariance matrices on the different tensor 
# dimensions.  This requires instantiating a spectral representation of the covariance matrices.  The Wishart_UnitDet 
# class does this automatically hence its use here.  

import torch
import numpy as np
from .Wishart import Wishart_UnitDet as Wishart
from .Gamma import Gamma

class TensorNormalWishart():

    def __init__(self,shape,batch_shape=()):
        self.dims = shape
        self.mu_0 = torch.zeros(batch_shape + shape,requires_grad=False)
        self.mu = torch.randn(batch_shape + shape,requires_grad=False)/np.sqrt(np.prod(self.dims))
        self.event_dim = len(shape)
        self.event_shape = shape
        self.batch_dim = len(batch_shape)
        self.batch_shape = batch_shape

        self.lambda_mu_0 = torch.ones(batch_shape,requires_grad=False)
        self.lambda_mu = torch.ones(batch_shape,requires_grad=False)
        self.invU = ()
        for i in range(len(shape)):
            self.invU = self.invU + (Wishart((shape[i]+2)*torch.ones(batch_shape,requires_grad=False),
                        torch.zeros(batch_shape+(shape[i],shape[i]),requires_grad=False)+torch.eye(shape[i],requires_grad=False)),)

        self.alpha = Gamma(torch.ones(batch_shape,requires_grad=False),torch.ones(batch_shape,requires_grad=False))

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n 
        self.event_shape = self.batch_shape[-n:] + self.event_shape 
        self.batch_shape = self.batch_shape[:-n]
        for invU in self.invU:
            invU.to_event(n)
        return self

    def log_mvdigamma(self,nu,p):
        return (nu.unsqueeze(-1) - torch.arange(p)/2.0).digamma().sum(-1)

    def log_mvdigamma_prime(self,nu,p):
        return (nu.unsqueeze(-1) - torch.arange(p)/2.0).polygamma(1).sum(-1)

    def raw_update(self,X,iters=1,lr=1.0,beta=None):
        for i in range(iters):
            self._raw_update(X,lr=lr,beta=beta)

    def _raw_update(self,X,lr=1.0,beta=None):
        if beta is None:
            beta = 1.0 - lr 
        sample_shape = X.shape[:-self.event_dim - self.batch_dim]
        N = np.prod(sample_shape)*torch.ones(self.batch_shape)
        lambda_mu = self.lambda_mu_0 + N
        mu = (X.sum(list(range(len(sample_shape)))) + self.mu_0*self.lambda_mu_0.view(self.batch_shape+self.event_dim*(1,)))/lambda_mu.view(self.batch_shape+self.event_dim*(1,))
        X = (X - mu)

        # Traces = self.ETraceSigmas()
        # Traces = Traces.prod(-1,True)/Traces

        for i in range(len(self.event_shape)):
            # temp = X.swapaxes(self.batch_dim + i + 1,-1)
            # temp = (temp.unsqueeze(-1)*temp.unsqueeze(-2)).sum(list(range(len(sample_shape))))
            # temp = temp.sum(list(range(self.batch_dim,self.batch_dim+self.event_dim-1)))/Traces[i]/self.alpha.mean().view(self.batch_shape+(1,1))
            # N_temp = N*(np.prod(self.event_shape)/self.event_shape[i])
            # self.invU[i].ss_update(temp,N_temp,lr=lr)

            idx = list(range(0,i))+list(range(i+1,len(self.event_shape)))
            sidx1 = list(range(-2*len(self.event_shape),-2*len(self.event_shape)+i)) + list(range(-2*len(self.event_shape)+i+1,-len(self.event_shape)))
            sidx2 = list(range(-len(self.event_shape),-len(self.event_shape)+i)) + list(range(-len(self.event_shape)+i+1,0))
            temp = (self.EinvSigma(idx)*X.view(X.shape+len(self.event_shape)*(1,)))
            temp = (temp.sum(sidx1)*X.unsqueeze(-len(self.event_shape)-1)).sum(sidx2)
            temp = temp.sum(list(range(0,len(sample_shape))))
            self.invU[i].ss_update(temp,N,lr=lr,beta=beta)


        self.lambda_mu = lr*lambda_mu + beta*self.lambda_mu
        self.mu = lr*mu + beta*self.mu 

        temp = (self.EinvSigma()*X.view(sample_shape + self.batch_shape + self.event_shape + len(self.event_shape)*(1,))*X.view(sample_shape + self.batch_shape + len(self.event_shape)*(1,) + self.event_shape)).sum(list(range(len(sample_shape))))
        temp = temp.sum(list(range(self.batch_dim,self.batch_dim+2*self.event_dim)))/self.alpha.mean()
        self.alpha.ss_update(torch.tensor(np.prod(self.event_shape)*np.prod(sample_shape)/2.0).expand(self.batch_shape).float(), temp/2.0 ,lr=lr,beta=beta)

    def KLqprior(self):
        temp = self.mu - self.mu_0
        KL = (temp.view(self.batch_shape + self.dims + len(self.dims)*(1,))*self.EinvSigma()*temp.view(self.batch_shape + len(self.dims)*(1,) + self.dims)).sum(list(range(-2*len(self.dims),0)))
        KL = 0.5*self.lambda_mu_0*KL + 0.5*(self.lambda_mu_0/self.lambda_mu - 1 + (self.lambda_mu/self.lambda_mu_0).log())*np.prod(self.dims)

        for i in range(len(self.event_shape)):
            KL = KL + self.invU[i].KLqprior()
        return KL + self.alpha.KLqprior()

    def Elog_like(self,X):
        X = X - self.mu
        ELL = -0.5*(self.EinvSigma()*X.view(X.shape+len(self.dims)*(1,))*X.view(X.shape[:-len(self.dims)]+len(self.dims)*(1,)+X.shape[-len(self.dims):])).sum(list(range(-2*len(self.dims),0)))
        ELL = ELL - 0.5*np.prod(self.dims)*np.log(2*np.pi) + 0.5*self.ElogdetinvSigma()
        return ELL

    def mean(self):
        return self.mu

    def var(self):
        raise NotImplementedError
        pass

    def EinvSigma(self,dims=None):
        if dims is None:
            dims = list(range(0,len(self.dims)))
        EinvSigma = self.invU[dims[0]].EinvSigma().view(self.batch_shape+2*(dims[0]*(1,) + (self.dims[dims[0]],) +(len(self.dims)-dims[0]-1)*(1,)))*self.alpha.mean().view(self.batch_shape+2*len(self.dims)*(1,))
        for i in dims[1:]:
            EinvSigma = EinvSigma*self.invU[i].EinvSigma().view(self.batch_shape+2*(i*(1,) + (self.dims[i],) +(len(self.dims)-i-1)*(1,)))
        return EinvSigma

    def ESigma(self,dims=None):
        if dims is None:
            dims = list(range(0,len(self.dims)))
        ESigma = self.invU[dims[0]].ESigma().view(self.batch_shape+2*(dims[0]*(1,) + (self.dims[dims[0]],) +(len(self.dims)-dims[0]-1)*(1,)))*self.alpha.meaninv().view(self.batch_shape+2*len(self.dims)*(1,))
        for i in dims[1:]:
            ESigma = ESigma*self.invU[i].ESigma().view(self.batch_shape+2*(i*(1,) + (self.dims[i],) +(len(self.dims)-i-1)*(1,)))
        return ESigma

    def ETraceinvSigmas(self):
        res=torch.zeros(self.batch_shape + (0,),requires_grad=False)
        for invU in self.invU:
            res = torch.cat((res,invU.ETraceinvSigma().unsqueeze(-1)),dim=-1)
        return res

    def ETraceSigmas(self):
        res=torch.zeros(self.batch_shape + (0,),requires_grad=False)
        for invU in self.invU:
            res = torch.cat((res,invU.ETraceSigma().unsqueeze(-1)),dim=-1)
        return res

    def ElogdetinvSigmas(self):
        res=torch.zeros(self.batch_shape + (0,),requires_grad=False)
        for invU in self.invU:
            res = torch.cat((res,invU.ElogdetinvSigma().unsqueeze(-1)),dim=-1)
        return res

    def ElogdetinvSigma(self):
        res = np.prod(self.dims)*self.alpha.loggeomean()
        for invU in self.invU:
            res = res + invU.ElogdetinvSigma()
        return res

