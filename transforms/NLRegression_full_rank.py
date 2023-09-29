
import torch
import transforms.MatrixNormalWishart as MatrixNormalWishart
import transforms.MatrixNormalGamma as MatrixNormalGamma
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format
import dists.NormalInverseWishart as NormalInverseWishart
import dists.NormalGamma as NormalGamma
import dists.Dirichlet as Dirichlet

class NLRegression_full_rank():
    # Generative model 2 for NL regression.  Generative model is:
    #  z_t ~ Cat(pi)
    #  x_t|z_t ~ NormalInverseWishart(mu_z_t, Sigma_z_t)
    #  y_t|x_t,z_t ~ Normal(A_z_t x_t, Sigma_yy_z_t)

    def __init__(self,n,p,mixture_dim,batch_shape=(),independent=False):
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.independent = independent

        self.A = MatrixNormalWishart((n,p),batch_shape + (mixture_dim,),scale=mixture_dim**(-1.0/n),pad_X=True)

        if independent == True:
            self.X =  NormalGamma((p,),batch_shape + (mixture_dim,))
        else:            
            self.X =  NormalInverseWishart((p,),batch_shape + (mixture_dim,))
        self.pi = Dirichlet((mixture_dim,),batch_shape)


    def raw_update(self,X,Y,iters=1.0,lr=1.0,verbose=False):
        ELBO = -torch.tensor(torch.inf)
        for i in range(self.batch_dim+1):
            X = X.unsqueeze(-2)
            Y = Y.unsqueeze(-2)
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)

        for i in range(int(iters)):
            log_p = self.A.Elog_like(X,Y) + self.X.Elog_like(X.squeeze(-1)) + self.pi.loggeomean()
            self.logZ = log_p.logsumexp(-1,keepdim=True)
            log_p = log_p - log_p.max(-1,keepdim=True)[0]
            self.p = log_p.exp()
            self.p = self.p/self.p.sum(-1,keepdim=True)
            self.NA = self.p.sum(0)

            ELBO_last = ELBO
            ELBO = self.ELBO().sum()
            if verbose == True:
                print('Percent Change in ELBO = ',((ELBO-ELBO_last)/ELBO_last.abs()).data*100)
            self.pi.ss_update(self.NA,lr)
            self.A.raw_update(X,Y,p=self.p,lr=lr)
            self.X.raw_update(X.squeeze(-1),p=self.p,lr=lr)

    def forward(self,X):
        print('NL Regression full rank: forward not implemented:  using predict instead')
        return self.predict(X)

    def predict(self,X):
        log_p = self.X.Elog_like(X.unsqueeze(-2)) + self.pi.loggeomean()
        log_p = log_p - log_p.max(-1,keepdim=True)[0]
        p = log_p.exp()
        p = p/p.sum(-1,keepdim=True)
        if self.A.pad_X is True:
            invSigmamu_y = self.A.EinvUX()[...,:-1]@X.unsqueeze(-2).unsqueeze(-1) + self.A.EinvUX()[...,-1:]
        else:
            invSigmamu_y = self.A.EinvUX()@X.unsqueeze(-2).unsqueeze(-1) 

        invSigma_y = (self.A.EinvSigma()*p.view(p.shape+(1,1))).sum(-3)
        invSigmamu_y = (invSigmamu_y*p.view(p.shape+(1,1))).sum(-3)
        Sigma_y = invSigma_y.inverse()
        mu_y  = Sigma_y@invSigmamu_y

        return MultivariateNormal_vector_format(mu = mu_y, Sigma = Sigma_y), p

    def ELBO(self):
        return self.logZ.sum(0) - self.KLqprior()

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.X.KLqprior().sum(-1) + self.pi.KLqprior()


