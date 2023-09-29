
import torch
import transforms.MatrixNormalWishart as MatrixNormalWishart
import transforms.MatrixNormalGamma as MatrixNormalGamma
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format
import dists.NormalInverseWishart as NormalInverseWishart
import dists.NormalGamma as NormalGamma
import dists.Dirichlet as Dirichlet

class NLRegression_low_rank():
    # Generative model of low rank NL regression.  When mixture_dim = 1 
    # this performs a cononinical correlation analysis.  
    #  z_t ~ Cat(pi)
    #  u_t|z_t ~ Normal(mu_z_t, Sigma_z_t)
    #  x_t|u_t ~ Normal(W u_t, Sigma_xx)
    #  y_t|u_t,z_t ~ Normal(A_z_t u_t, Sigma_yy_z_t)

    def __init__(self,n,p,hidden_dim,mixture_dim,batch_shape=(),independent=False):
        self.hidden_dim = hidden_dim
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.mixture_dim = mixture_dim
        self.independent = independent
#        self.W = MatrixNormalWishart(torch.zeros(batch_shape + (1,hidden_dim,p)))  # the 1 is because W is same for all clusters on u 
        if independent is True:
            self.W = MatrixNormalGamma((p,hidden_dim), batch_shape + (1,))  # the 1 is because W is same for all clusters on u 
        else:
            self.W = MatrixNormalWishart((p,hidden_dim), batch_shape + (1,))  # the 1 is because W is same for all clusters on u 

        self.A = MatrixNormalWishart((n,hidden_dim+1), batch_shape + (mixture_dim,), scale = mixture_dim**(-1.0/n))
        # self.U =  NormalInverseWishart(torch.ones(batch_shape + (mixture_dim,),requires_grad=False), 
        #        torch.zeros(batch_shape + (mixture_dim,hidden_dim),requires_grad=False), 
        #        (hidden_dim+2)*torch.ones(batch_shape + (mixture_dim,),requires_grad=False),
        #        torch.zeros(batch_shape + (mixture_dim, hidden_dim, hidden_dim),requires_grad=False)+torch.eye(hidden_dim,requires_grad=False)*mixture_dim**2,
        #        )
        self.U = NormalGamma((hidden_dim,), batch_shape + (mixture_dim,))
        # self.U = NormalInverseWishart(mu_0 = torch.zeros(batch_shape + (mixture_dim,hidden_dim,)))
        self.ELBO_last = -torch.tensor(torch.inf)        
        self.pi = Dirichlet((mixture_dim,),batch_shape)

    def raw_update(self,X,Y,iters=1.0,lr=1.0,verbose=False):
        ELBO = -torch.tensor(torch.inf)
        for i in range(self.batch_dim+1):
            X = X.unsqueeze(-2)
            Y = Y.unsqueeze(-2)
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)

        if self.independent is True:
            SExx = (X.pow(2)).sum(0).squeeze(-1)
        else:
            SExx = (X@X.transpose(-1,-2)).sum(0)

        for i in range(int(iters)):
            invSigma_u_u = self.U.EinvSigma() + self.A.EXTinvUX()[...,:-1,:-1] + self.W.EXTinvUX()
            invSigmamu_u = self.U.EinvSigmamu().unsqueeze(-1) + self.A.EXTinvU()[...,:-1,:]@Y - self.A.EXTinvUX()[...,:-1,-1:] + self.W.EXTinvU()@X
            Sigma_u_u = invSigma_u_u.inverse()
            mu_u = Sigma_u_u@invSigmamu_u

            logZ = -0.5*Y.transpose(-1,-2)@self.A.EinvSigma()@Y - 0.5*X.transpose(-1,-2)@self.W.EinvSigma()@X - 0.5*self.A.EXTinvUX()[...,-1:,-1:] + self.A.EXTinvU()[...,-1:,:]@Y + 0.5*mu_u.transpose(-1,-2)@invSigma_u_u@mu_u 
            logZ = logZ.squeeze(-1).squeeze(-1) + 0.5*self.A.ElogdetinvSigma() + 0.5*self.U.ElogdetinvSigma()+ 0.5*self.W.ElogdetinvSigma() - 0.5*invSigma_u_u.logdet() - 0.5*self.U.EXTinvUX()

            log_p = logZ + self.pi.loggeomean()

            shift = log_p.max(-1,keepdim=True)[0]
            self.logZ = (log_p-shift).logsumexp(-1,keepdim=True) + shift

            self.p = (log_p-shift).exp()
            self.p = self.p/self.p.sum(-1,keepdim=True)
            self.logZ = self.logZ.squeeze(-1)

            SEuu = Sigma_u_u + mu_u@mu_u.transpose(-1,-2)
            SEux = mu_u@X.transpose(-1,-2)

            SEu1u1 = torch.cat((SEuu,mu_u),-1)
            mu_u1 = torch.cat((mu_u,torch.ones(mu_u.shape[:-2] + (1,1),requires_grad=False)),-2)
            SEu1u1 = torch.cat((SEu1u1,mu_u1.transpose(-2,-1)),-2)

            SEyy = Y@Y.transpose(-1,-2)
            SEyu1 = Y@mu_u1.transpose(-1,-2)

            self.NA = self.p.sum(0)
            p = self.p.view(self.p.shape + (1,1))
            SEu =  (mu_u*p).sum(0)  # averages over q(u|z)
            SEuu = (SEuu*p).sum(0)
            SEux = (SEux*p).sum(0)

            SEu1u1 = (SEu1u1*p).sum(0)
            SEyy = (SEyy*p).sum(0)
            SEyu1 = (SEyu1*p).sum(0)

            ELBO_last = ELBO
            ELBO = self.ELBO().sum()

            if verbose:
                print('Percent Change in ELBO = ',((ELBO-ELBO_last)/ELBO_last.abs()).data*100)
            self.pi.ss_update(self.NA,lr)
            self.A.ss_update(SEu1u1,SEyu1,SEyy,self.NA,lr)
            self.W.ss_update(SEuu.sum(-3,True),SEux.sum(-3,True).transpose(-1,-2),SExx,self.NA.sum(-1,True),lr)
            self.U.ss_update(SEuu.diagonal(dim1=-1,dim2=-2),SEu.squeeze(-1),self.NA,lr)
#             self.U.ss_update(SEuu,SEu.squeeze(-1),self.NA,lr)

    def forward(self,pX):
        NotImplemented('NL Regression full rank: forward not implemented')

    def predict(self,X):
        for i in range(self.batch_dim+1):
            X = X.unsqueeze(-2)
        X = X.unsqueeze(-1)

        invSigma_u_u = self.U.EinvSigma() + self.W.EXTinvUX()
        invSigmamu_u = self.U.EinvSigmamu().unsqueeze(-1) + self.W.EXTinvU()@X
        Sigma_u_u = invSigma_u_u.inverse()
        mu_u = Sigma_u_u@invSigmamu_u

        logZ = - 0.5*X.transpose(-1,-2)@self.W.EinvSigma()@X + 0.5*mu_u.transpose(-1,-2)@invSigma_u_u@mu_u 
        logZ = logZ.squeeze(-1).squeeze(-1) + 0.5*self.U.ElogdetinvSigma()+ 0.5*self.W.ElogdetinvSigma() - 0.5*invSigma_u_u.logdet() - 0.5*self.U.EXTinvUX()

        log_p = logZ + self.pi.loggeomean()

        shift = log_p.max(-1,keepdim=True)[0]
        logZ = (log_p-shift).logsumexp(-1,keepdim=True) + shift

        log_p = log_p - logZ
        p = log_p.exp()
        mu_u1 = torch.cat((mu_u,torch.ones(mu_u.shape[:-2] + (1,1),requires_grad=False)),-2)

        mu_y = (self.A.mu@mu_u1)
        Sigma_y = self.A.mu[...,:,:-1]@Sigma_u_u@self.A.mu[...,:,:-1].transpose(-1,-2) + self.A.ESigma()
        Sigma_y = ((Sigma_y + mu_y@mu_y.transpose(-2,-1))*p.view(p.shape + (1,1))).sum(-3)
        mu_y = (mu_y*p.view(p.shape+(1,1))).sum(-3)
        Sigma_y = Sigma_y - mu_y@mu_y.transpose(-2,-1)

        # invSigma_y = (self.A.EinvSigma()*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        # invSigmamu_y = ((self.A.EinvUX()@mu_u1)*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        # Sigma_y = invSigma_y.inverse()
        # mu_y  = Sigma_y@invSigmamu_y

        # invSigma_u_u = (invSigma_u_u*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        # invSigmamu_u = (invSigmamu_u*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        # mu_u = invSigma_u_u.inverse()@invSigmamu_u

        return MultivariateNormal_vector_format(mu = mu_y, Sigma = Sigma_y), p, mu_u.squeeze(-1)

    def ELBO(self):
        return self.logZ.sum(0) - self.KLqprior()

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.W.KLqprior().sum(-1) + self.U.KLqprior().sum(-1) + self.pi.KLqprior()
