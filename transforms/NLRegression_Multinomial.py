
import torch
import transforms.MatrixNormalWishart as MatrixNormalWishart
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format
import transforms.MultiNomialLogisticRegression as MultiNomialLogisticRegression 

class NLRegression_Multinomial():
    # Generative model of NL regression.  Generative model is:
    #  z_t ~ MNRL(x_t)
    #  y_t|z_t,x_t ~ MatrixNormalWishart
    print("NLRegression has no forward/backward methods, Use dMixtureofLinearTransofrorms instead")
    def __init__(self,n,p,mixture_dim,batch_shape=()):

        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 2
        self.n = n
        self.p = p
        self.mixture_dim = mixture_dim
        self.ELBO_last = -torch.tensor(torch.inf)

        self.A = MatrixNormalWishart((n,p), batch_shape + (mixture_dim,), scale = 1.0/mixture_dim**(1.0/n),pad_X=True)
        self.Z = MultiNomialLogisticRegression(mixture_dim, p, batch_shape = batch_shape, pad_X=True)

    def raw_update(self,X,Y,iters=1.0,lr=1.0,verbose=False):
        AX = X.view(X.shape + (1,))  # make vector
        AY = Y.view(Y.shape + (1,))
        AX = AX.view(AX.shape[:-2] + (self.batch_dim+1)*(1,) + AX.shape[-2:]) # add z dim and batch_dim
        AY = AY.view(AY.shape[:-2] + (self.batch_dim+1)*(1,) + AY.shape[-2:])

        for i in range(int(iters)):
            log_p = self.A.Elog_like(AX,AY) + self.Z.log_predict(X)
            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            self.logZ = shift.squeeze(-1) + log_p.logsumexp(-1)
            p = log_p.exp()
            p = p/p.sum(-1,True)
            self.NA = p.sum(0)

            ELBO = self.logZ.sum() - self.KLqprior()
            if verbose: print("Percent Change in ELBO = ",((ELBO-self.ELBO_last)/self.ELBO_last.abs()).data*100)
            self.ELBO_last = ELBO

            self.A.raw_update(AX,AY,p=p,lr=lr)
            self.Z.raw_update(X,p,lr=lr,verbose=False)

    def Elog_like_X(self,Y):
        AY = Y.view(Y.shape + (1,))
        AY = AY.view(AY.shape[:-2] + (self.batch_dim+1)*(1,) + AY.shape[-2:])
        invSigma,invSigmamu,Res = self.A.Elog_like_X(AY)
        NotImplemented("NLRegression has no Elog_like_X, Use dMixtureofLinearTransofrorms instead")
        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu), Res

    def forward(self,X):
        NotImplemented("NLRegression has no forward/backward methods, Use dMixtureofLinearTransofrorms instead")
        pass

    def backward(self,Y):
        NotImplemented("NLRegression has no forward/backward methods, Use dMixtureofLinearTransofrorms instead")
        pass

    def predict_full(self,X):
        log_p = self.Z.log_predict(X)  
        log_p = log_p - log_p.max(-1,keepdim=True)[0]
        p = log_p.exp()
        p = p/p.sum(-1,True)
        p = p.view(p.shape+(1,1))
        return self.A.predict(X.unsqueeze(-2).unsqueeze(-1)) + (p,)

    def predict(self,X):
        p=self.Z.predict(X)
        pv = p.view(p.shape+(1,1))

        pY  = self.A.predict(X.unsqueeze(-2).unsqueeze(-1))[0]
        mu = (pY.mean()*pv).sum(-3)
        Sigma = (pY.EXXT()*pv).sum(-3) - mu@mu.transpose(-2,-1)

        return MultivariateNormal_vector_format(mu = mu, Sigma = Sigma), p

    def ELBO(self):
        return self.logZ - self.KLqprior()

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.Z.KLqprior()

