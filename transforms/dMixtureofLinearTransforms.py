import torch
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format
import dists.MVN_ard as MVN_ard
from .MatrixNormalWishart import MatrixNormalWishart
from .MatrixNormalGamma import MatrixNormalGamma
from .MultiNomialLogisticRegression import MultiNomialLogisticRegression

class dMixtureofLinearTransforms():
    # This basically a mxiture of linear transforms, p(y|x,z) with a mixture components driven by 
    # z ~ p(z|x) which is MNLR.  Component number give the number of different z's, latent_dim gives the dimension of x, and obs_dim gives the dimension
    # of y.  
    
    def __init__(self, n, p, mixture_dim, batch_shape=(),pad_X=True, type = 'Wishart'):
        self.event_shape = (mixture_dim,n,p)
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 3
        self.n = n
        self.p = p
        self.mix_dim = mixture_dim
        self.ELBO_last = -torch.tensor(torch.inf)

        scale = 1.0/mixture_dim**(1.0/n)
        if type == 'Wishart':
            self.A = MatrixNormalWishart(event_shape = (n,p), batch_shape = batch_shape + (mixture_dim,), 
                                         scale = scale, pad_X=pad_X)
        elif type == 'Gamma':
            self.A = MatrixNormalGamma(event_shape = (n,p), batch_shape = batch_shape + (mixture_dim,), 
                                         scale = scale, pad_X=pad_X)
        elif type == 'MVN_ard':
            raise NotImplementedError
        else:
            raise ValueError('type must be either Wishart (default) or Gamma')
        self.pi = MultiNomialLogisticRegression(mixture_dim,p,batch_shape = batch_shape,pad_X=True)
        self.ELBO_save = []

    def raw_update(self,X,Y,p=None,iters=1,lr=1.0,verbose=False):
        AX = X.unsqueeze(-1).unsqueeze(-3)  # make vector and mixture dimension compatible
        AY = Y.unsqueeze(-1).unsqueeze(-3)

        for i in range(iters):
            log_p = self.A.Elog_like(AX,AY) + self.pi.log_predict(X)  # A.Elog_like is sample x batch x component
            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            p_ass = log_p.exp()
            p_ass = p_ass/p_ass.sum(-1,True)
            if verbose:
                ELBO = (shift.squeeze(-1) + log_p.logsumexp(-1)).sum(0) - self.KLqprior()
                print("dMixture Percent Change in ELBO = ",((ELBO-self.ELBO_last)/self.ELBO_last.abs()).data*100)
                self.ELBO_last = ELBO

            self.pi.raw_update(X,p_ass,p=p,lr=lr,verbose=False)
            if p is None:
                self.A.raw_update(AX,AY,p=p_ass,lr=lr)
            else: 
                self.A.raw_update(AX,AY,p=p_ass*p.unsqueeze(-1),lr=lr)

    def postdict(self, Y):

        invSigma, invSigmamu, Res = self.A.Elog_like_X(Y.unsqueeze(-2).unsqueeze(-1))  # Res is sample x batch x component 
#        invSigma, invSigmamu, Res = self.A.Elog_like_X_given_pY(pY.unsqueeze(-3))  # Res is sample x batch x component 
        like_X = MultivariateNormal_vector_format(invSigma = invSigma.unsqueeze(0).movedim(-3,-3-self.batch_dim), invSigmamu = invSigmamu.movedim(-3,-3-self.batch_dim))
        Res = Res.movedim(-1,-1-self.batch_dim)  # This res is just from the A, does not include like_X contribution

        Z = torch.eye(self.mix_dim)
        for i in range(self.batch_dim):
            Z = Z.unsqueeze(-2)
        invSigma, invSigmamu, Sigma, mu, Res_z = self.pi.Elog_like_X(like_X,Z,iters=4)  # Res_z includes input like_X contrib, but not output like_X contrib
        Res = Res + Res_z + 0.5*(mu*invSigmamu).sum(-2).squeeze(-1) - 0.5*invSigma.logdet() + like_X.dim/2.0*torch.log(2*torch.tensor(torch.pi,requires_grad=False))
        logZ = Res.logsumexp(-1-self.batch_dim,True)
        logp = Res - logZ
        logZ = logZ.squeeze(-1)
        p = logp.exp()

        pv = p.view(p.shape+(1,1))
        invSigma = (invSigma*pv).sum(-3-self.batch_dim)
        invSigmamu = (invSigmamu*pv).sum(-3-self.batch_dim)
        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu), logZ.squeeze(-1-self.batch_dim), p

        # Sigma = ((Sigma+mu@mu.transpose(-2,-1))*pv).sum(-3-self.batch_dim)
        # mu = (mu*pv).sum(-3-self.batch_dim)
        # Sigma = Sigma - mu@mu.transpose(-2,-1)
        # return MultivariateNormal_vector_format(Sigma = Sigma, mu = mu), logZ.squeeze(-1-self.batch_dim), p
#        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu), logZ.squeeze(-1-self.batch_dim)

    def predict(self,X):  # update to handle batching
        p=self.pi.predict(X)
        pv=p.view(p.shape+(1,1))
        Xv = X.view(X.shape[:-1]+(1,) + X.shape[-1:] + (1,))
        pY = self.A.predict(Xv)[0]

#        invSigma = (invSigma_y_y*pv).sum(-3)
#        invSigmamu = (invSigmamu_y*pv).sum(-3)
#        Sigma = invSigma.inverse()
#        mu = Sigma@invSigmamu

        Sigma = (pY.EXXT()*pv).sum(-3)
        mu = (pY.mean()*pv).sum(-3)
        Sigma = Sigma - mu@mu.transpose(-2,-1)
        return MultivariateNormal_vector_format(mu = mu, Sigma = Sigma), p

    def update(self,pX,pY,p=None,iters=1,lr=1.0,verbose=False):
        # Expects X and Y to be batch consistent, i.e. X is sample x batch x p
        #                                              Y is sample x batch x n
        pAX = pX.unsqueeze(-3)
        pAY = pY.unsqueeze(-3)
        for i in range(iters):
            log_p = self.A.Elog_like_given_pX_pY(pAX,pAY) + self.pi.log_forward(pX)
            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            self.logZ = shift.squeeze(-1) + log_p.logsumexp(-1)
            p_ass = log_p.exp()
            p_ass = p_ass/p_ass.sum(-1,True)
            self.NA = p_ass.sum(0)

            self.pi.update(pX, p_ass, p=p, lr=lr, verbose=False)
            if p is None:
                self.A.update(pAX,pAY,p=p_ass,lr=lr)
            else: 
                self.A.update(pAX, pAY, p=p_ass*p.unsqueeze(-1), lr=lr)

            ELBO = self.logZ.sum() - self.KLqprior().sum()
            if verbose:
                print('dMixLT Percent Change in ELBO: ', (ELBO-self.ELBO_last)/self.ELBO_last.abs())
            self.ELBO_last = ELBO

    def forward(self,pX):
        p = self.pi.forward(pX)        
        pY = self.A.forward(pX.unsqueeze(-3))
        mu = (pY.mean()*p.view(p.shape+(1,1))).sum(-3)
        Sigma = (pY.EXXT()*p.view(p.shape+(1,1))).sum(-3)-mu@mu.transpose(-2,-1)
        return MultivariateNormal_vector_format(Sigma = Sigma, mu = mu)

    def forward_mix(self,pX):
        return self.A.forward(pX.unsqueeze(-3)), self.pi.forward(pX)     
    
    def backward(self,pY):
        pX, ResA = self.A.backward(pY.unsqueeze(-3))
        pX, Res = self.pi.backward(torch.eye(self.mix_dim),pX)
        log_p = Res + ResA
        p = log_p - log_p.max(-1,True)[0]
        p = p.exp()
        p = p/p.sum(-1,True)
        p = p.unsqueeze(-1).unsqueeze(-1)

        invSigma = (pX.EinvSigma()*p).sum(-3)
        invSigmamu = (pX.EinvSigmamu()*p).sum(-3)

        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu), log_p - log_p.logsumexp(-1,True)

    def backward_mix(self,pY):
        pX, ResA = self.A.backward(pY.unsqueeze(-3))
        pX, Res = self.pi.backward(pX,torch.eye(self.mix_dim))
        log_p = Res + ResA
        shift = log_p.max(-1,True)[0]
        log_p = log_p - shift
        Res = (shift.squeeze(-1) + log_p.logsumexp(-1))
        p = p.exp()
        p = p/p.sum(-1,True)
        Res = Res - pX.Res()
        return pX, p, Res

    def Elog_like_given_pX_pY(self,pX,pY):
        pAX = pX.unsqueeze(-3)  # make mixture dimension compatible
        pAY = pY.unsqueeze(-3)
        log_p = self.A.Elog_like_given_pX_pY(pAX,pAY) + self.pi.log_forward(pX)
        return log_p.logsumexp(-1)
    
    def Elog_like(self,X,Y):
        log_p = self.A.Elog_like(X.unsqueeze(-1).unsqueeze(-3),Y.unsqueeze(-1).unsqueeze(-3)) + self.pi.log_predict(X)
        return log_p.logsumexp(-1)

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.pi.KLqprior() 


