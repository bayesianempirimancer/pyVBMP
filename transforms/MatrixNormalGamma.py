# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 
import torch
import dists.DiagonalWishart as DiagonalWishart
import dists.DiagonalWishart_UnitTrace as DiagonalWishart_UnitTrace

import utils.matrix_utils as matrix_utils
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format

class MatrixNormalGamma():
    # Conugate prior for linear regression coefficients
    # mu is assumed to be n x p
    # V is assumed to be p x p and plays the role of lambda for the normal wishart prior
    # invU is n x n diagonal matrix with diagonal elements assumed to be gamma distributed
    # i.e.  Y = A @ X + U^{-1/2} @ eps, where A is the random variable represented by the MNW prior
    # When used for linear regression, either zscore X and Y or pad X with a column of ones
    #   mask is a boolean tensor of shape mu_0 that indicates which entries of mu can be non-zero
    #         mask must be the same for all elements of a given batch, i.e. it is n x p 
    #   X_mask is a boolean tensor of shape (mu_0.shape[:-2] + (1,) + mu_shape[-1:]) 
    #           that indicates which entries of X contribute to the prediction 

    def __init__(self, event_shape, batch_shape = (), 
                 prior_parms = {'mu':torch.tensor(0.0)}, scale=1.0, 
                 uniform_precision=False, mask=None, X_mask=None, pad_X=False, fixed_precision=False):
        self.n = torch.tensor(event_shape[-2]).long()
        self.p = torch.tensor(event_shape[-1]).long()
        self.pad_X = pad_X
        self.fixed_precision = fixed_precision
        self.uniform_precision = uniform_precision
        mu_0 = prior_parms['mu']

        if pad_X:
            self.p = self.p+1
            event_shape = event_shape[:-1] + (self.p,)
            if mu_0.ndim!=0:
                mu_0 = torch.cat((mu_0,torch.zeros(mu_0.shape[:-1]+(1,),requires_grad=False)),dim=-1).clone()
        mu_0 = mu_0.expand(batch_shape + event_shape)

        self.event_dim = len(event_shape)
        self.event_shape = event_shape
        self.batch_dim = len(batch_shape)
        self.batch_shape = batch_shape

        self.mask = mask
        self.X_mask = X_mask
        self.mu_0 = mu_0
        self.mu = torch.randn_like(mu_0,requires_grad=False)/self.p.sqrt()
        self.invV_0 = torch.eye(self.p,requires_grad=False).expand(batch_shape + event_shape[:-2] + (self.p,self.p))
        self.invV = self.invV_0
        self.V = self.invV.inverse()

        self.invU = DiagonalWishart(event_shape = event_shape[:-1], batch_shape = batch_shape, scale = scale)
        self.logdetinvV = self.invV.logdet()    
        self.logdetinvV_0=self.invV_0.logdet() 

        self.SEyy = 0.0   
        self.SExx = 0.0
        self.SEyx = 0.0
        self.N = 0.0

        if X_mask is not None:
            if pad_X:
                self.X_mask = torch.cat((X_mask,torch.ones(X_mask.shape[:-1]+(1,),requires_grad=False)>0),dim=-1)
            self.mu_0 = self.mu_0*self.X_mask
            self.mu = self.mu*self.X_mask
            self.V = self.V*self.X_mask*self.X_mask.transpose(-2,-1)
            self.invV = self.invV*self.X_mask*self.X_mask.transpose(-2,-1)

        if mask is not None:
            if pad_X:
                self.mask = torch.cat((self.mask,torch.ones(self.mask.shape[:-1]+(1,),requires_grad=False)>0),dim=-1)
            self.mu_0 = self.mu_0*self.mask
            self.mu = self.mu*self.mask

        self.log2pi = torch.tensor(2*torch.pi,requires_grad=False).log()

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n 
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.invU.to_event(n)
        return self

    def ss_update(self,SExx,SEyx,SEyy,N,lr=1.0,beta=None):
        assert(SExx.ndim==self.batch_dim+self.event_dim)
        assert(SEyx.ndim==self.batch_dim+self.event_dim)
        assert(SEyy.ndim==self.batch_dim+self.event_dim)
        assert(N.ndim == self.batch_dim+self.event_dim - 2)

        if beta is not None:
            self.SExx = beta*self.SExx + SExx
            self.SEyx = beta*self.SEyx + SEyx
            self.SEyy = beta*self.SEyy + SEyy
            self.N = beta*self.N + N
            SExx = self.SExx
            SEyx = self.SEyx
            SEyy = self.SEyy
            N = self.N

        if self.X_mask is not None:
            SExx = SExx*self.X_mask*self.X_mask.transpose(-2,-1)
            SEyx = SEyx*self.X_mask
            invV = self.invV_0 + SExx
            muinvV = self.mu_0@self.invV_0 + SEyx
            mu = muinvV@invV.inverse()
            mu = mu*self.X_mask
        else:
            invV = self.invV_0 + SExx
            muinvV = self.mu_0@self.invV_0 + SEyx
            mu = torch.linalg.solve(invV,muinvV.transpose(-2,-1)).transpose(-2,-1)
            # mu = muinvV@invV.inverse()

        if self.mask is not None:  # Assumes mask is same for whole batch
            V = invV.inverse()
            U = self.invU.EinvSigma().inverse()
            Astar = V.unsqueeze(-3).unsqueeze(-2)*U.unsqueeze(-2).unsqueeze(-1)
            A  = Astar[...,~self.mask,:,:][...,:,~self.mask]            
            b = mu[...,~self.mask]
            gamma = torch.zeros_like(mu)
            gamma[...,~self.mask] = torch.linalg.solve(A,b)
            mu = mu - U@gamma@V
            mu = mu*self.mask

        if self.fixed_precision is False:
            SEyy = SEyy - mu@invV@mu.transpose(-2,-1) + self.mu_0@self.invV_0@self.mu_0.transpose(-2,-1)
            self.invU.ss_update(SEyy.diagonal(dim1=-2,dim2=-1), N.unsqueeze(-1), lr=lr)
            if self.uniform_precision==True:
                self.invU.gamma.alpha = self.invU.gamma.alpha.sum(-1,keepdim=True)  # THIS IS A HACK
        self.invV = lr*invV + (1.0-lr)*self.invV
        self.invV = 0.5*(self.invV + self.invV.transpose(-2,-1))
        self.mu = lr*mu + (1.0-lr)*self.mu
        if(self.mask is not None):
            self.mu = self.mu*self.mask

#        self.invV_d, self.invV_v = torch.linalg.eigh(self.invV) 
#        self.V = self.invV_v@(1.0/self.invV_d.unsqueeze(-1)*self.invV_v.transpose(-2,-1))
#        self.logdetinvV = self.invV_d.log().sum(-1)
        self.V = self.invV.inverse()
        self.logdetinvV = self.invV.logdet()

        if self.X_mask is not None:
#            self.V = self.V * self.X_mask * self.X_mask.transpose(-2,-1)
#            self.invV = self.invV * self.X_mask * self.X_mask.transpose(-2,-1)
            self.mu = self.mu*self.X_mask
#            self.logdetinvV = self.logdetinvV - self.logdetinvV_0*(~self.X_mask).sum(-1).sum(-1)

    def update(self,pX,pY,p=None,lr=1.0,beta=None):
        sample_shape = pX.shape[:-self.event_dim-self.batch_dim]
        sample_dims = tuple(range(len(sample_shape)))
        if p is None:
            SExx = pX.EXXT().sum(sample_dims)
            SEyy = pY.EXXT().sum(sample_dims)
            SEyx = (pY.EX()@pX.EX().transpose(-2,-1)).sum(sample_dims)
            N = torch.prod(torch.tensor(sample_shape,requires_grad=False))
            N = N.expand(self.batch_shape + self.event_shape[:-2])
        else:
            N=p.sum(sample_dims)
            p=p.view(p.shape + self.event_dim*(1,))
            SExx = (pX.EXXT()*p).sum(sample_dims)
            SEyy = (pY.EXXT()*p).sum(sample_dims)
            SEyx = ((pY.EX()@pX.EX().transpose(-2,-1))*p).sum(sample_dims)
        
        if self.pad_X:
            if p is None:
                SEx = pX.EX().sum(sample_dims)
                SEy = pY.EX().sum(sample_dims)
            else: 
                SEx = (pX.EX()*p).sum(sample_dims)
                SEy = (pY.EX()*p).sum(sample_dims)
                    
            SExx = torch.cat((SExx,SEx),dim=-1)
            SEx = torch.cat((SEx,N.view(N.shape+(1,1))),dim=-2)
            SExx = torch.cat((SExx,SEx.transpose(-2,-1)),dim=-2)
            SEyx = torch.cat((SEyx,SEy.expand(SEyx.shape[:-1]+(1,))),dim=-1)

        self.ss_update(SExx,SEyx,SEyy,N,lr=lr,beta=beta)

    def raw_update(self,X,Y,p=None,lr=1.0,beta=None):
        sample_shape = X.shape[:-self.event_dim-self.batch_dim]
        sample_dims = tuple(range(len(sample_shape)))
        if p is None:
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            SExx = (X*X.transpose(-2,-1)).sum(sample_dims)
            SEyy = (Y*Y.transpose(-2,-1)).sum(sample_dims)
            SEyx = (Y*X.transpose(-2,-1)).sum(sample_dims)
            N = torch.prod(torch.tensor(sample_shape,requires_grad=False))
            N = N.expand(self.batch_shape + self.event_shape[:-2])
        else:
            N=p.sum(sample_dims)
            p = p.view(p.shape+self.event_dim*(1,))
            SExx = (X*X.transpose(-2,-1)*p).sum(sample_dims)
            SEyy = (Y*Y.transpose(-2,-1)*p).sum(sample_dims)
            SEyx = (Y*X.transpose(-2,-1)*p).sum(sample_dims)

        if self.pad_X:
            if p is None:
                SEx = X.sum(sample_dims)
                SEy = Y.sum(sample_dims)
            else: 
                SEx = (X*p).sum(sample_dims)
                SEy = (Y*p).sum(sample_dims)

            SExx = torch.cat((SExx,SEx),dim=-1)
            SEx = torch.cat((SEx,N.view(N.shape+(1,1))),dim=-2)
            SExx = torch.cat((SExx,SEx.transpose(-2,-1)),dim=-2)
            SEyx = torch.cat((SEyx,SEy.expand(SEyx.shape[:-1]+(1,))),dim=-1)

        self.ss_update(SExx,SEyx,SEyy,N,lr=lr,beta=beta)

    def KLqprior(self):

        KL = self.n/2.0*self.logdetinvV - self.n/2.0*self.logdetinvV_0 - self.n*self.p/2.0
        if self.X_mask is not None:
            KL = KL + self.n/2.0*self.logdetinvV_0*(self.X_mask).sum(-1).sum(-1)
        KL = KL + 0.5*self.n*(self.invV_0*self.V).sum(-1).sum(-1)
        temp = (self.mu-self.mu_0).transpose(-2,-1)@(self.invU.gamma.mean().unsqueeze(-1)*(self.mu-self.mu_0))
        KL = KL + 0.5*(self.invV_0*temp).sum(-1).sum(-1)
        for i in range(self.event_dim-2):
            KL = KL.sum(-1)
        if self.uniform_precision == True:
            KL = KL + self.invU.KLqprior()/self.n
        else:
            KL = KL + self.invU.KLqprior()
        for i in range(self.event_dim-2):
            KL = KL.sum(-1)
        return KL

    def Elog_like(self,X,Y):  # expects X to be batch_shape* + event_shape[:-1] by p 
                              #     and Y to be batch_shape* + event_shape[:-1] by n
                              # for mixtures batch_shape* = batch_shape[:-1]+(1,)
        ELL = -0.5*(Y.transpose(-2,-1)@self.EinvSigma()@Y).squeeze(-1).squeeze(-1)
        if self.pad_X:
            ELL = ELL + (Y.transpose(-2,-1)@(self.EinvUX()[...,:,:-1]@X + self.EinvUX()[...,:,-1:])).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(X.transpose(-2,-1)@self.EXTinvUX()[...,:-1,:-1]@X + 2*self.EXTinvUX()[...,-1:,:-1]@X +  self.EXTinvUX()[...,-1:,-1:]).squeeze(-1).squeeze(-1)
        else:
            ELL = ELL + (Y.transpose(-2,-1)@self.EinvUX()@X).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(X.transpose(-2,-1)@self.EXTinvUX()@X).squeeze(-1).squeeze(-1)
        ELL = ELL + 0.5*self.ElogdetinvSigma() - 0.5*self.n*self.log2pi
        for i in range(self.event_dim-2):
            ELL = ELL.sum(-1)
        return ELL

    def Elog_like_given_pX_pY(self,pX,pY):  # This assumes that X is a distribution with the ability to produce 
                                       # expectations of the form EXXT, and EX with dimensions matching Y, i.e. EX.shape[-2:] is (p,1)

        ELL = -0.5*(pY.EXXT()*self.EinvSigma()).sum(-1).sum(-1)
        if self.pad_X:        
            ELL = ELL + (pY.mean().transpose(-2,-1)@(self.EinvUX()[...,:,:-1]@pX.mean()+self.EinvUX()[...,:,-1:])).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(pX.EXXT()*self.EXTinvUX()[...,:-1,:-1]).sum(-1).sum(-1)
            ELL = ELL - (self.EXTinvUX()[...,-1:,:-1]@pX.mean()).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(self.EXTinvUX()[...,-1,-1])            
        else:
            ELL = ELL + (pY.mean().transpose(-2,-1)@self.EinvUX()@pX.mean()).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(pX.EXXT()*self.EXTinvUX()).sum(-1).sum(-1)
        ELL = ELL + 0.5*self.invU.ElogdetinvSigma() - 0.5*self.n*self.log2pi
        for i in range(self.event_dim-2):
            ELL = ELL.sum(-1)
        return ELL

    def Elog_like_X(self,Y):
        if self.pad_X:
            invSigma_x_x = self.EXTinvUX()[...,:-1,:-1]
            invSigmamu_x = self.EXTinvU()[...,:-1,:]@Y - self.EXTinvUX()[...,:-1,-1:]   #DOUBLECHECK HERE
            Residual = -0.5*(Y.transpose(-2,-1)@self.EinvSigma()@Y).squeeze(-1).squeeze(-1) - 0.5*self.n*self.log2pi + 0.5*self.ElogdetinvSigma()
            Residual = Residual - 0.5*self.EXTinvUX()[...,-1,-1]
        else:
            invSigma_x_x = self.EXTinvUX()
            invSigmamu_x = self.EXTinvU()@Y
            Residual = -0.5*(Y.transpose(-2,-1)@self.EinvSigma()@Y).squeeze(-1).squeeze(-1) - 0.5*self.n*self.log2pi + 0.5*self.ElogdetinvSigma()
        return invSigma_x_x, invSigmamu_x, Residual

    def Elog_like_X_given_pY(self,pY):
        if self.pad_X:
            PJ_y_y = pY.EinvSigma() + self.EinvSigma()
            PJ_y_x = -self.EinvUX()[...,:,:-1]
            PJ_x_x = self.EXTinvUX()[...,:-1,:-1]
            PmuJ_y = pY.EinvSigmamu() - self.EinvUX()[...,:,-1:]
            PmuJ_x = -self.EXTinvUX()[...,:-1,-1:]
            PJ_1_1 = self.EXTinvUX()[...,-1,-1]
        else:
            PJ_y_y = pY.EinvSigma() + self.EinvSigma()
            PJ_y_x = -self.EinvUX()
            PJ_x_x = self.EXTinvUX()
            PmuJ_y = pY.EinvSigmamu()
            PmuJ_x = torch.zeros(self.p,1)
            PJ_1_1 = torch.tensor(0)

        invSigma_y_y, negBinvD, negCinvA, invSigma_x_x = matrix_utils.block_precision_marginalizer(PJ_y_y, PJ_y_x, PJ_y_x.transpose(-2,-1), PJ_x_x)
        invSigmamu_y = PmuJ_y + negBinvD@PmuJ_x
        invSigmamu_x = PmuJ_x + negCinvA@PmuJ_y

        Sigma_x_x = invSigma_x_x.inverse()
        mu_x = Sigma_x_x@invSigmamu_x

        Res = pY.Res() + 0.5*(invSigmamu_y.transpose(-2,-1)@invSigma_y_y.inverse()@invSigmamu_y).squeeze(-1).squeeze(-1)
        Res = Res - 0.5*invSigma_y_y.logdet() + 0.5*pY.dim*self.log2pi + 0.5*self.ElogdetinvSigma() - 0.5*PJ_1_1
        px = MultivariateNormal_vector_format(invSigma=invSigma_x_x,invSigmamu=invSigmamu_x,mu=mu_x,Sigma=Sigma_x_x)
        return px, Res - px.Res()

    def Eforward(self,pX):
        # This is distinct from forward in that it doesnt marginalize over the joint distribution of X and Y
        # rather it computes the expected value of the log probability of Y given X.  This approximation 
        # fails to propagate uncertainty about X to Y, but is fast
        if self.pad_X:
            invSigma = self.EinvSigma()
            invSigmamu = self.EinvUX()[...,:,:-1]@pX.mean() + self.EinvUX()[...,:,-1:]
        else:
            invSigma = self.EinvSigma()
            invSigmamu = self.EinvUX()[...,:,:-1]@pX.mean()
        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu)

    def forward(self,pX):
        if self.pad_X:
            PJ_y_y = self.EinvSigma()
            PJ_y_x = -self.EinvUX()[...,:,:-1]
            PJ_x_x = self.EXTinvUX()[...,:-1,:-1] + pX.EinvSigma()
            PmuJ_y = self.EinvUX()[...,:,-1:]
            PmuJ_x = pX.EinvSigmamu()-self.EXTinvUX()[...,:-1,-1:]
#            PJ11 = self.EXTinvUX()[...,-1,-1]
        else:
            PJ_y_y = self.EinvSigma()
            PJ_y_x = -self.EinvUX()
            PJ_x_x = self.EXTinvUX() + pX.EinvSigma()
            PmuJ_y = torch.zeros(PJ_y_y.shape[:-1]+(1,))
            PmuJ_x = pX.EinvSigmamu()
#            PJ11 = torch.tensor(0.0)

        invSigma_y_y, negBinvD = matrix_utils.block_precision_marginalizer(PJ_y_y, PJ_y_x, PJ_y_x.transpose(-2,-1), PJ_x_x)[0:2]
        invSigmamu_y = PmuJ_y + negBinvD@PmuJ_x
        return MultivariateNormal_vector_format(invSigma = invSigma_y_y, invSigmamu = invSigmamu_y)

    def backward(self,pY):  
        if self.pad_X:
            PJ_y_y = pY.EinvSigma() + self.EinvSigma()
            PJ_y_x = -self.EinvUX()[...,:,:-1]
            PJ_x_x = self.EXTinvUX()[...,:-1,:-1]
            PmuJ_y = pY.EinvSigmamu() + self.EinvUX()[...,:,-1:]
            PmuJ_x = -self.EXTinvUX()[...,:-1,-1:]
            PJ11 = self.EXTinvUX()[...,-1,-1]
        else:
            PJ_y_y = pY.EinvSigma() + self.EinvSigma()
            PJ_y_x = -self.EinvUX()
            PJ_x_x = self.EXTinvUX()
            PmuJ_y = pY.EinvSigmamu()
            PmuJ_x = torch.zeros(PJ_x_x.shape[:-1] + (1,))
            PJ11 = torch.tensor(0.0)

        invSigma_y_y, negBinvD, negCinvA, invSigma_x_x = matrix_utils.block_precision_marginalizer(PJ_y_y, PJ_y_x, PJ_y_x.transpose(-2,-1), PJ_x_x)
        invSigmamu_y = PmuJ_y + negBinvD@PmuJ_x
        invSigmamu_x = PmuJ_x + negCinvA@PmuJ_y

        pX = MultivariateNormal_vector_format(invSigma = invSigma_x_x, invSigmamu = invSigmamu_x)
        Res = pY.Res() + 0.5*(invSigmamu_y.transpose(-2,-1)@invSigma_y_y.inverse()@invSigmamu_y).squeeze(-1).squeeze(-1) - 0.5*invSigma_y_y.logdet() + 0.5*pY.dim*self.log2pi + 0.5*self.ElogdetinvSigma() - 0.5*PJ11
        return pX, Res - pX.Res()

    def Ebackward(self,pY):
        raise NotImplementedError
        pass

    def predict(self,X):
        if self.pad_X:
            invSigmamu_y = (self.EinvUX()[...,:,:-1]@X + self.EinvUX()[..., :,-1:])
            Res = -0.5*X.transpose(-1,-2)@self.EXTinvUX()[...,:-1,:-1]@X - self.EXTinvUX()[...,-1:,:-1]@X - 0.5*self.EXTinvUX()[...,-1:,-1:]
        else:
            invSigmamu_y = (self.EinvUX()@X)
            Res = -0.5*X.transpose(-1,-2)@self.EXTinvUX()@X
        Res = Res.squeeze(-1).squeeze(-1) + 0.5*self.ElogdetinvSigma() - 0.5*self.n*self.log2pi
        return MultivariateNormal_vector_format(invSigma = self.EinvSigma(), invSigmamu = invSigmamu_y), Res

    def predict_given_pX(self,pX):
        return self.forward(pX)

    def mean(self):
        return self.mu

    def bias(self):
        if self.pad_X is True:
            return self.mu[...,-1:]
        else:
            return torch.tensor(0.0)

    def weights(self): 
        if self.pad_X is True:
            return self.mu[...,:-1]
        else:
            return self.mu

    def var(self):
        return self.ESigma().diagonal(dim1=-1,dim2=-2).unsqueeze(-1)*self.V.diagonal(dim1=-1,dim2=-2).unsqueeze(-2)

    ### Compute special expectations used for VB inference
    def EinvUX(self):
        return self.invU.gamma.mean().unsqueeze(-1)*self.mu

    def EXTinvU(self):
        return self.mu.transpose(-2,-1)@self.invU.EinvSigma()

    def EXTAX(self,A):  # X is n x p, A is p x p
        return self.V*(self.invU.gamma.meaninv()*A.diagonal(dim1=-2,dim2=-1)).sum(-1)  + self.mu.transpose(-2,-1)@A@self.mu

    def EXmMUTAXmMU(self,A):
        return self.V*(self.invU.gamma.meaninv()*A.diagonal(dim1=-2,dim2=-1)).sum(-1).sum(-1)

    def EXAXT(self,A):
        return self.ESigma()*(self.V*A).sum(-1).sum(-1) + self.mu@A@self.mu.transpose(-2,-1) #DOUBLE CHECK HERE

    def EXmMUAXmMUT(self,A):
        return self.ESigma()*(self.V*A).sum(-1).sum(-1)

    def EXTinvUX(self):
        return self.n * self.V + self.mu.transpose(-1,-2)@(self.invU.gamma.mean().unsqueeze(-1)*self.mu)

    def EXinvVXT(self):
        return self.p * self.invU.ESigma() + self.mu@self.invV@self.mu.transpose(-1,-2)

    def EXmMUTinvUXmMU(self): # X minus mu
        return self.n*self.V

    def EXmMUinvVXmMUT(self):
        return self.p*self.invU.ESigma()

    def EXTX(self):
        return self.V * self.invU.gamma.meaninv().sum() + self.mu.transpose(-1,-2)@self.mu

    def EXXT(self):
        return self.V.diagonal().sum() * self.invU.ESigma() + self.mu@self.mu.transpose(-1,-2)

    def ElogdetinvU(self):
        return self.invU.gamma.loggeomean().sum(-1)

    def ElogdetinvSigma(self):
        return self.invU.gamma.loggeomean().sum(-1)

    def EinvSigma(self):  
        return self.invU.mean()

    def ESigma(self):  
        return self.invU.ESigma()
    
class MatrixNormalGamma_UnitTrace(MatrixNormalGamma):

    def __init__(self,mu_0,U_0=None,V_0=None,mask=None,X_mask=None,pad_X=False):
        super().__init__(mu_0,U_0=U_0,V_0=V_0,uniform_precision=False,mask=mask,X_mask=X_mask,pad_X=pad_X)
        alpha = self.invU.gamma.alpha_0
        beta = self.invU.gamma.beta_0
        self.invU = DiagonalWishart_UnitTrace(alpha,beta)

