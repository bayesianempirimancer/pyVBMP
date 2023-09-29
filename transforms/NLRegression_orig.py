
import torch
import transforms.MatrixNormalWishart as MatrixNormalWishart
import transforms.MatrixNormalGamma as MatrixNormalGamma
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format
import dists.NormalInverseWishart as NormalInverseWishart
import dists.NormalGamma as NormalGamma
import dists.Dirichlet as Dirichlet

class NLRegression_orig():
    # Generative model:
    #         u_t | x_t,z_t = Normal(mu_z_t + W x_t, Sigma_uu)
    #         y_t | u_t,z_t = Normal(A_z_t u_t + B_z_t, Sigma_z_t)
    # with variational posterior on parameters
    #         q(w) = matrix normal (mu_w,lambda_w| Sigma_ww)q(Sigma_uu)
    #         q(mu_z) = normal inverse wishart(mu,lambda|Sigma_ww)q(Sigma_ww)
    #         q(A_z) = matrix normal wishart
    # So that the ciritcal ingredient to make inference easy and fast is that
    # q(Sigma_uu) is shared between mu_z and w

    def __init__(self,n,p,hidden_dim,mixture_dim,batch_shape=()):
        self.hidden_dim = hidden_dim
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.n = n
        self.p = p
        self.mixture_dim = mixture_dim

        self.W = MatrixNormalWishart((hidden_dim,p),batch_shape + (1,))
        self.A = MatrixNormalWishart((n,hidden_dim+1),batch_shape + (mixture_dim,))
        self.U =  NormalInverseWishart((hidden_dim,),batch_shape + (mixture_dim,),fixed_precision=True)
        
        self.U.invU = self.W.invU  # This is dangerous because it means we cant update U in the usual way
        self.pi = Dirichlet((mixture_dim,), batch_shape)        
        self.ELBO_last = -torch.tensor(torch.inf)

    def raw_update(self,X,Y,iters=1.0,lr=1.0,verbose=False):
        for i in range(self.batch_dim+1):
            X = X.unsqueeze(-2)
            Y = Y.unsqueeze(-2)
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)

        SExx = (X@X.transpose(-1,-2)).sum(0)

        for i in range(int(iters)):
            #compute p(u|x,u,z)

#            fw = self.W.predict(X)
#            bw = self.A.Elog_like_X(Y)

            invSigma_u_u = self.W.EinvSigma() + self.A.EXTinvUX()[...,:-1,:-1]
            invSigmamu_u = self.U.EinvSigmamu().unsqueeze(-1) + self.W.EinvUX()@X + self.A.EXTinvU()[...,:-1,:]@Y - self.A.EXTinvUX()[...,:-1,-1:] 
            Sigma_u_u = invSigma_u_u.inverse()  # no dependence on sample :)
            mu_u = Sigma_u_u@invSigmamu_u

            Res = -0.5*Y.transpose(-1,-2)@self.A.EinvSigma()@Y - 0.5*self.A.EXTinvUX()[...,-1:,-1:] + self.A.EXTinvU()[...,-1:,:]@Y
            Res = Res - 0.5*X.transpose(-1,-2)@self.W.EXTinvUX()@X - self.U.mean().unsqueeze(-1).transpose(-2,-1)@self.W.EinvUX()@X + 0.5*mu_u.transpose(-1,-2)@invSigmamu_u            
            Res = Res.squeeze(-1).squeeze(-1) + 0.5*self.A.ElogdetinvSigma() + 0.5*self.W.ElogdetinvSigma() - 0.5*invSigma_u_u.logdet() - 0.5*self.U.EXTinvUX()
            Res = Res - 0.5*self.n*torch.log(2*torch.tensor(torch.pi,requires_grad=False))
            log_p = Res + self.pi.loggeomean()

            shift = log_p.max(-1,keepdim=True)[0]
            self.logZ = (log_p-shift).logsumexp(-1,keepdim=True) + shift
            log_p = log_p - self.logZ
            self.p = log_p.exp()
            self.logZ = self.logZ.squeeze(-1).sum(0)
            self.NA = self.p.sum(0)

            if verbose:
                ELBO = self.ELBO()
                print('Percent Change in ELBO = ',((ELBO-self.ELBO_last)/self.ELBO_last.abs())*100)
                self.ELBO_last = ELBO

            self.pi.ss_update(self.NA,lr)
# Compute SS for A updates
            p = self.p.view(self.p.shape + (1,1))
            NA = self.NA.view(self.NA.shape + (1,1))

            SEuu = ((Sigma_u_u + mu_u@mu_u.transpose(-1,-2))*p).sum(0)
            SEu = (mu_u*p).sum(0)  # batch x mixture_dim x hidden_dim x 1
            SEu1u1 = torch.cat((SEuu,SEu),-1)
            SEu1 = torch.cat((SEu,NA),-2)            
            SEu1u1 = torch.cat((SEu1u1,SEu1.transpose(-2,-1)),-2)
            SEyy = ((Y@Y.transpose(-1,-2))*p).sum(0)
            SEyu1 = torch.cat((((Y@mu_u.transpose(-1,-2))*p).sum(0),(Y*p).sum(0)),-1)

            self.A.ss_update(SEu1u1,SEyu1,SEyy,self.NA,lr)

# For U update we need only compute the mean since the covariance is shared with W
# and updated correctly when we update W.  

            SEx = (X*p).sum(0)
            ubar = self.U.mean().unsqueeze(-1)
            SEdux = ((mu_u-ubar)@X.transpose(-1,-2)*p).sum(0).sum(-3,True)
            SEdudu = SEuu - SEu*ubar.transpose(-2,-1) - ubar@SEu.transpose(-2,-1) + ubar@ubar.transpose(-2,-1)*NA
            SEdudu = SEdudu.sum(-3,True)
            mu = (SEu.squeeze(-1) - (self.W.mean()@SEx).squeeze(-1) + self.U.mu_0*self.U.lambda_mu_0.unsqueeze(-1))/(self.U.lambda_mu_0.unsqueeze(-1) + self.NA.unsqueeze(-1))
            self.W.ss_update(SExx,SEdux,SEdudu,self.NA.sum(-1,True),lr)
            self.U.lambda_mu = self.U.lambda_mu + lr*(self.NA+self.U.lambda_mu_0 - self.U.lambda_mu)
            self.U.mu = self.U.mu + lr*(mu - self.U.mu)

    def forward(self,X):
        print('NL Regression full rank: forward not implemented:  using predict instead')
        return self.predict(X.mean())

    def predict(self,X):
        X = X.unsqueeze(-2).unsqueeze(-1)
        invSigma_u_u = self.W.EinvSigma() 
        invSigmamu_u = self.W.EinvSigma()@self.U.mean().unsqueeze(-1) + self.W.EinvUX()@X 
        Sigma_u_u = invSigma_u_u.inverse()  # no dependence on t :)
        mu_u = Sigma_u_u@invSigmamu_u

        Res = - 0.5*X.transpose(-1,-2)@self.W.EXTinvUX()@X - self.U.mean().unsqueeze(-1).transpose(-2,-1)@self.W.EinvUX()@X + 0.5*mu_u.transpose(-1,-2)@invSigmamu_u
        Res = Res.squeeze(-1).squeeze(-1) + 0.5*self.W.ElogdetinvSigma() - 0.5*invSigma_u_u.logdet() - 0.5*self.U.EXTinvUX()
        log_p = Res + self.pi.loggeomean()

        log_p = log_p - log_p.max(-1,True)[0]
        p= log_p.exp()
        p = p/p.sum(-1,True)

        mu_u1 = torch.cat((mu_u,torch.ones(mu_u.shape[:-2] + (1,1),requires_grad=False)),-2)

        # A better approximation would be to marginalize over u|z instead of averaging in the log domain
        invSigma_y = (self.A.EinvSigma()*p.unsqueeze(-1).unsqueeze(-1)).sum(-3) 
        invSigmamu_y = ((self.A.EinvUX()@mu_u1)*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        Sigma_y = invSigma_y.inverse()

        mu_y  = self.A.mean()@mu_u1
        Sigma_y = self.A.ESigma() + self.A.mean()[...,:-1]@Sigma_u_u@self.A.mean()[...,:-1].transpose(-1,-2)
        Sigma_y = Sigma_y + mu_y@mu_y.transpose(-1,-2) - mu_y@mu_y.transpose(-1,-2) 

        mu_y = (mu_y*p.view(p.shape + (1,1))).sum(-3)
        Sigma_y = (Sigma_y*p.view(p.shape + (1,1))).sum(-3)
        Sigma_y = Sigma_y - mu_y@mu_y.transpose(-1,-2)

        return MultivariateNormal_vector_format(mu = mu_y, Sigma = Sigma_y), p

    def KLqprior(self):
        KL = self.A.KLqprior().sum(-1) + self.W.KLqprior().sum(-1) + self.U.KLqprior().sum(-1)
        KL = KL + self.pi.KLqprior() - self.U.invU.KLqprior().sum(-1)  # because invU is shared with W
        return KL

    def ELBO(self):
        return self.logZ.sum() - self.KLqprior()

