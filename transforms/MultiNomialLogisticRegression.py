import torch
import dists.MVN_ard as MVN_ard
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format

class MultiNomialLogisticRegression():
    # VB updates for multnomial logistic regression using the polyagamma version of Jaakkola and Jordan's
    # lower bouding method which were show to be more or less equivalent by Durante and Rigon
    # The polyagamma trick takes advantage of two equivalences:
    #
    #       pg(w|b,c) = cosh(c/2)^b exp(-w*c^2/2) pg(w|b,0)
    #
    #       exp(phi)^a/(1+exp(phi))^b = 2^(-b)*exp((a-b/2)*phi)/cosh^b(phi/2)
    #
    # It is assumed that X is a matrix of size (sample x batch x p)  and Y is either probabilities or 
    # a one hot tensor or a number of counts but must have tensor size = (sample x batch x n) 
    def __init__(self, n, p, batch_shape = (),pad_X=True):
        if pad_X == True:
            p = p+1
        n=n-1
        self.n=n
        self.p=p

        self.beta = MVN_ard(event_shape = (n,p,1),batch_shape=batch_shape)

        self.beta.mu = torch.randn_like(self.beta.mu)/torch.sqrt(torch.tensor(self.p,requires_grad=False))
        self.pad_X = pad_X               
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_shape = (n,p)
        self.event_dim = 2
        self.ELBO_last = torch.tensor(-torch.inf)
    
    def to_event(self,n):
        if n < 1:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.beta.to_event(n)        
        return self

    def raw_update(self,X,Y,iters = 2, p=None,lr=1.0,beta=None,verbose=False):
        # Assumes X is sample x batch x p and Y is sample x batch x n
        sample_shape = X.shape[:-self.event_dim-self.batch_dim+1]
        sample_dims = tuple(range(len(sample_shape)))
        ELBO = self.ELBO_last
        # if self.pad_X is True:
        #     X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        pgb = Y.sum(-1,True)-(Y.cumsum(-1)-Y)[...,:-1]
        YmN = Y[...,:-1]-pgb/2.0   
        YmN = YmN.view(YmN.shape+ (1,1))

        if self.pad_X is True:
            EX = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        else:
            EX = X
        EX = EX.view(EX.shape[:-1] + (1,) + EX.shape[-1:] + (1,))
        EXXT = EX*EX.transpose(-2,-1)
        # NEED to clean up the extra dimension on SEyx and SEy
        # This would require changes to MVN_ARD as well
        if p is None:
            SEyx = (YmN*EX).sum(sample_dims)  
        else:
            SEyx = ((YmN*EX)*p.view(p.shape+(1,1,1))).sum(sample_dims)
 
        for i in range(iters):
            pgc = (self.beta.EXXT()*EXXT).sum(-1).sum(-1).sqrt()  # shape = (sample x batch x n)
            Ew = (pgb/2.0/pgc*(pgc/2.0).tanh()).unsqueeze(-1).unsqueeze(-1) # expands to sample x batch x n x 1 x 1
            if p is None:
                SExx =  (Ew*EXXT).sum(sample_dims)  # batch x n x p x p 
            else:
                SExx = (Ew*EXXT*p.view(p.shape+(1,1,1))).sum(sample_dims)  # batch x n x p x p
            
            if verbose is True: 
                ELBO = (SEyx*self.beta.mean()).sum((-3,-2,-1)) - (pgb*(0.5*pgc).cosh().log()).sum(sample_dims).sum(-1) - pgb.sum(sample_dims).sum(-1)*torch.log(torch.tensor(2,requires_grad=False)) - self.KLqprior()
                print("MNLR Percent Change in ELBO: ",((ELBO-self.ELBO_last)/self.ELBO_last.abs()*100))
                self.ELBO_last = ELBO

            self.beta.ss_update(SExx,SEyx,lr=lr,beta=beta)

    def update(self,pX,pY,iters=2,p=None,lr=1,beta=None,verbose=False):  
        sample_shape = pX.shape[:-self.event_dim-self.batch_dim]
        sample_dims = tuple(range(len(sample_shape)))
        # Here pX is assumed to be a probability distribution that supports EXXT()
        ELBO = self.ELBO_last

        pgb = pY.sum(-1,True)-(pY.cumsum(-1)-pY)[...,:-1]
        YmN = pY[...,:-1]-pgb/2.0   
        YmN = YmN.view(YmN.shape + (1,1))   # expands to sample x batch x n x 1 x 1

        EXXT = pX.EXXT().unsqueeze(-3)  # sample x batch x 1 x p x p 
        EX = pX.mean().unsqueeze(-3)    # sample x batch x 1 x p x 1

        # NEED TO clean up to avoid using torch.cat until after averaging is done

        if self.pad_X is True:
            EXXT = torch.cat((EXXT,EX),dim=-1) 
            EX = torch.cat((EX,torch.ones(EX.shape[:-2]+(1,1))),dim=-2)
            EXXT = torch.cat((EXXT,EX.transpose(-2,-1)),dim=-2)        
        if p is None:
            SEyx = (YmN*EX).sum(0)  
        else:
            SEyx = ((YmN*EX)*p.view(p.shape+(1,1,1))).sum(sample_dims)

        for i in range(iters):
            pgc = (self.beta.EXXT()*EXXT).sum(-1).sum(-1).sqrt()  # shape = (sample x batch x n)
            Ew = (pgb/2.0/pgc*(pgc/2.0).tanh()).unsqueeze(-1).unsqueeze(-1)  # expands to sample x batch x n x 1 x 1
            if p is None:
                SExx =  (Ew*EXXT).sum(sample_dims)  # sample x batch x n x p x p 
            else:
                SExx = (Ew*EXXT*p.view(p.shape+(1,1,1))).sum(sample_dims)  # sample x batch x n x p x p
            
            if verbose is True: 
                ELBO = (SEyx*self.beta.mean()).sum((-3,-2,-1)) - (pgb*(0.5*pgc).cosh().log()).sum(sample_dims).sum(-1) - pgb.sum(sample_dims).sum(-1)*torch.log(torch.tensor(2,requires_grad=False)) - self.KLqprior()
                print("MNLR Percent Change in ELBO: ",((ELBO-self.ELBO_last)/self.ELBO_last.abs()*100))
                self.ELBO_last = ELBO
            self.beta.ss_update(SExx,SEyx,lr=lr,beta=beta)


    def forward(self, pX):
        sample_shape = pX.mean().shape[:-2]
        Yt = torch.eye(self.n+1)
        for i in range(len(sample_shape)):
            Yt = Yt.unsqueeze(-2)

        N = Yt.sum(-1,True)-(Yt.cumsum(-1)-Yt)
        YmN = Yt-N/2.0            
        pgb = N[...,:-1]
        YmN = YmN[...,:-1]

        EXXT = pX.EXXT().unsqueeze(-3)  # sample x batch x 1 x p x p 
        EX = pX.mean().unsqueeze(-3)    # sample x batch x 1 x p x 1
        if self.pad_X is True:
            EXXT = torch.cat((EXXT,EX),dim=-1) 
            EX = torch.cat((EX,torch.ones(EX.shape[:-2]+(1,1))),dim=-2)
            EXXT = torch.cat((EXXT,EX.transpose(-2,-1)),dim=-2)
        
        SEyxb = (YmN.view(YmN.shape+(1,1))*EX*self.beta.mean()).sum(-1).sum(-1)  

        pgc = (self.beta.EXXT()*EXXT).sum(-1).sum(-1).sqrt()  # shape = (sample x batch x n)        
        out = SEyxb.sum(-1) - (pgb*(0.5*pgc).cosh().log()).sum(-1) - pgb.sum(-1)*torch.log(torch.tenspr(2.0,reuqires_grad=False))
        return  out.movedim(0,-1)  # returns log probability sample x batch x n


    def ELBO(self,X=None,Y=None):
        if X is not None:
            return self.Elog_like(X,Y).sum() - self.KLqprior()
        else:
            return self.ELBO_last
    
    def KLqprior(self):
        KL = self.beta.KLqprior()
        for i in range(self.event_dim-2):
            KL = KL.sum(-1)
        return KL

    def weights(self):
        if self.pad_X is True:
            mu = self.beta.mean()[...,:-1,0]
        return 2*mu - mu.cumsum(-2)

    def Elog_like_given_pX_pY(self,pX,Y):  # assumes pX is in vector format
        if self.pad_X is True:
            EX = pX.mean()
            EXXT = torch.cat((pX.EXXT(),EX),dim=-1)
            EX = torch.cat((EX,torch.ones(EX.shape[:-2]+(1,1))),dim=-2)
            EXXT = torch.cat((EXXT,EX.transpose(-2,-1)),dim=-2)
        else:
            EX = pX.mean()
            EXXT = pX.EXXT()
        N = Y.sum(-1,True)-(Y.cumsum(-1)-Y)
        YmN = Y-N/2.0   # should have shape (sample x batch x n) 
                        # X has sample x batch x n         
        # Remove Superfluous final dimension of Y and N
        pgb = N[...,:-1]
        YmN = YmN[...,:-1]
        EX = EX.unsqueeze(-3)  
        EXXT = EXXT.unsqueeze(-3)
        SEyxb = (YmN.unsqueeze(-1)*EX.squeeze(-1)*self.beta.mean().squeeze(-1)).sum(-1)
        pgc = (EXXT*self.beta.EXXT()).sum(-1).sum(-1).sqrt()  # should have shape (sample x batch x n)
        return SEyxb.sum(-1) - (pgb*(0.5*pgc).cosh().log()).sum(-1) - pgb.sum(-1)*torch.log(torch.tensor(2.0,requires_grad=False))


    def Elog_like(self,X,Y):
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        N = Y.sum(-1,True)-(Y.cumsum(-1)-Y)
        YmN = Y-N/2.0   # should have shape (sample x batch x n) 
                        # X has sample x batch x p         
        # Remove Superfluous final dimension of Y and N
        pgb = N[...,:-1]
        YmN = YmN[...,:-1]
        X = X.unsqueeze(-2)  # expands to sample x batch x  p x 1

        SEyxb = (YmN.unsqueeze(-1)*X*self.beta.mean().squeeze(-1)).sum(-1)
        X = X.unsqueeze(-1)
        pgc = (X*(self.beta.EXXT()@X)).sum(-2).squeeze(-1).sqrt()  # should have shape (sample x batch x n)
        return SEyxb.sum(-1) - (pgb*(0.5*pgc).cosh().log()).sum(-1) - pgb.sum(-1)*torch.log(torch.tensor(2.0,requires_grad=False))

    def backward(self,pY,like_X=None):
        if like_X is None:
            p = self.p - self.pad_X
            like_X = MultivariateNormal_vector_format(invSigmamu = torch.zeros((pY.ndim-1)*(1,)+(p,1)),invSigma = torch.eye(p).expand((pY.ndim-1)*(1,)+(p,p)))
        invSigma, invSigmamu, Sigma, mu, Res = self.Elog_like_X(like_X,pY) # returns invSigma, invSigmamu, Res
        return MultivariateNormal_vector_format(invSigma=invSigma, invSigmamu=invSigmamu, Sigma=Sigma, mu=mu), Res

    def Elog_like_X(self,like_X, pY, iters=2):
        N = pY.sum(-1,True)-(pY.cumsum(-1)-pY)
        YmN = pY-N/2.0   
        pgb = N[...,:-1]
        YmN = YmN[...,:-1]  # sample x batch x n

        BBT = self.beta.EXXT()
        pgc = self.beta.EXXT().sum(-1).sum(-1).sqrt()  # should have shape (sample x batch x n)
        Ew = pgb/2.0/pgc*(pgc/2.0).tanh()

        for i in range(iters):
            if self.pad_X is True:
                invSigmamu = (YmN.view(YmN.shape+(1,1))*self.beta.mean()[...,:-1,-1:] - Ew.view(Ew.shape + (1,1))*BBT[...,:-1,-1:]).sum(-3)
                invSigmamu = like_X.EinvSigmamu()+invSigmamu
                invSigma = (Ew.view(Ew.shape+(1,1))*BBT[...,:-1,:-1]).sum(-3)
                invSigma = like_X.EinvSigma()+invSigma
                Sigma = invSigma.inverse()
                mu = Sigma@invSigmamu
                pgc = ((BBT[...,:-1,:-1]*(Sigma + mu@mu.transpose(-1,-2)).unsqueeze(-3)).sum(-1).sum(-1)+2*(BBT[...,-1:,:-1]@mu.unsqueeze(-3)).squeeze(-1).squeeze(-1)+BBT[...,-1,-1]).sqrt()  # should have shape (sample x batch x n)
            else:
                invSigmamu = (YmN.view(YmN.shape+(1,1))*self.beta.mean()).sum(-3)
                invSigmamu = like_X.EinvSigmamu()+invSigmamu
                invSigma = (Ew.view(Ew.shape+(1,1))*BBT).sum(-3)
                invSigma = like_X.EinvSigma()+invSigma
                Sigma = invSigma.inverse()
                mu = Sigma@invSigmamu
                pgc = ((BBT*(Sigma + mu@mu.transpose(-1,-2)).unsqueeze(-3)).sum(-1).sum(-1)).sqrt()  # should have shape (sample x batch x n)
            Ew = pgb/2.0/pgc*(pgc/2.0).tanh()

        if self.pad_X is True:
            Res = - pgb.sum(-1)*torch.log(torch.tensor(2.0,requires_grad=False)) + (YmN*((self.beta.mean()[...,-1:,:-1]*mu.unsqueeze(-3)).sum(-1).sum(-1) + self.beta.mean()[...,-1,-1])).sum(-1)
        else:
            Res = - pgb.sum(-1)*torch.log(torch.tensor(2.0,requires_grad=False)) + (YmN*((self.beta.mean()*mu.unsqueeze(-3)).sum(-1).sum(-1) )).sum(-1)  
        Res = Res - (pgb*(0.5*pgc).cosh().log()).sum(-1) + like_X.Res()# + 0.5*(mu*invSigmamu).sum(-2).squeeze(-1) - 0.5*invSigma.logdet() + like_X.dim/2.0*np.log(2*np.pi)        
        return invSigma, invSigmamu, Sigma, mu, Res

    def log_predict(self,X):  # slower than log_predict_1 because some calculations are repeated
        sample_shape = X.shape[:-1]
        Yt = torch.eye(self.n+1)
        for i in range(len(sample_shape)):
            Yt = Yt.unsqueeze(-2)        
        return self.Elog_like(X,Yt).movedim(0,-1)

    def log_forward(self,pX):  # slower than log_predict_1 because some calculations are repeated
        sample_shape = pX.shape[:-2]
        Yt = torch.eye(self.n+1)
        for i in range(len(sample_shape)):
            Yt = Yt.unsqueeze(-2)        
        return self.Elog_like_given_pX_pY(pX,Yt).movedim(0,-1)

    def loggeomean(self,X):
        return self.log_predict(X)

    def log_predict_2(self,X):
        # This version of the forward method exactly marginalizes out the
        # betas while using the expectation of w for the pg integration
        # This approach seems to perform ever so slightly worse than 
        # lowerbounding the log probabilities by integrating out q(w|<psi^2>)
        # It is also slower so....
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        X=X.unsqueeze(-2)
        psi_bar = (X*self.beta.mean().squeeze(-1)).sum(-1)
        X=X.unsqueeze(-1)
        pgc = (X*(self.beta.EXXT()@X)).sum(-2).squeeze(-1).sqrt() # sample x batch x n-1
        Ew = 0.5/pgc*(0.5*pgc).tanh() # sample x batch x n-1
        psi_var = (X*(self.beta.ESigma()@X)).sum(-1).sum(-1) 

        nat1_plus = 0.5 + psi_bar/psi_var
        nat1_minus = nat1_plus - 1.0 
        nat2 = Ew + 1.0/psi_var

#        Res = -0.5*pgc.pow(2)*Ew + (0.5*pgc).cosh().log()
        Res =  (0.5*pgc).cosh().log()
        lnpsb = 0.5*nat1_plus.pow(2)/nat2 - 0.5*nat2.log() - 0.5*psi_bar.pow(2)/psi_var - 0.5*psi_var.log() - torch.log(torch.tensor(2,requires_grad=False)) 
        lnpsb = lnpsb + Res
        lnpsb_minus = lnpsb + 0.5*(nat1_minus.pow(2)-nat1_plus.pow(2))/nat2
        
        lnp = torch.zeros(lnpsb.shape[:-1] + (lnpsb.shape[-1]+1,))
        lnp[...,1:] = lnpsb_minus.cumsum(-1)
        lnp[...,:-1] = lnp[...,:-1] + lnpsb

        return lnp

    def log_predict_1(self,X):  # the forward method
        # lower bounds the probability of each class by approximately integrating out
        # the pg augmentation variable using q(w) = pg(w|b,<psi^2>.sqrt())
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        lnpsb = X@self.beta.mean().squeeze(-1).transpose(-2,-1) # contribs from K term
        X=X.unsqueeze(-1).unsqueeze(-3)
        pgc = (X*(self.beta.EXXT()@X)).sum(-2).squeeze(-1).sqrt()  # should have shape (sample x batch x n)
        lnpsb_N = - (0.5*pgc).cosh().log() - torch.log(torch.tensor(2.0,requires_grad=False)) # contribs from N term
        lnpsb_0 = -0.5*lnpsb.sum(-1,True) + lnpsb_N.sum(-1,True)

        lnpsb = lnpsb - 0.5*lnpsb.cumsum(-1) + lnpsb_N.cumsum(-1)  
        return torch.cat((lnpsb,lnpsb_0),dim=-1)

    def predict(self,X):
        # lower bounds the probability of each class by approximately integrating out
        # the pg augmentation variable using q(w) = pg(w|b,<psi^2>.sqrt())
        lnpsb = self.log_predict(X)
        psb = (lnpsb-lnpsb.max(-1,True)[0]).exp()
        psb = psb/psb.sum(-1,True)
        return psb

    def forward(self,pX):
        # lower bounds the probability of each class by approximately integrating out
        # the pg augmentation variable using q(w) = pg(w|b,<psi^2>.sqrt())
        lnpsb = self.log_forward(pX)
        psb = (lnpsb-lnpsb.max(-1,True)[0]).exp()
        psb = psb/psb.sum(-1,True)
        return psb

    def predict_2(self,X):
        lnpsb = self.log_predict_2(X)
        psb = (lnpsb-lnpsb.max(-1,True)[0]).exp()
        psb = psb/psb.sum(-1,True)
        return psb
        

    # def polyagamma_sample(self,sample_shape,numgammas = 20):
    #     denom = torch.arange(0.5, numgammas).pow(2.0)
    #     gk = -torch.rand(sample_shape + (numgammas,)).log()
    #     return (gk/denom).sum(-1)/(2.0*np.pi**2)

    # def psb_gibbs(self,X,num_samples = 100):
    #     if self.pad_X is True:
    #         X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)

    #     psi_bar = X@self.beta.mean().squeeze(-1).transpose(-2,-1)
    #     X=X.unsqueeze(-1)
    #     psi_var = (self.beta.ESigma()*(X@X.transpose(-2,-1)).unsqueeze(-3)).sum(-1).sum(-1) 
    #     beta_Res = self.beta.Res()

    #     w=self.polyagamma_sample((num_samples,) + psi_bar.shape)

    #     nat1 = 0.5 + psi_bar/psi_var
    #     nat2 = w + 1.0/psi_var

    #     lnpsb = 0.5*nat1.pow(2)/nat2 - 0.5*nat2.log() - 0.5*psi_bar.pow(2)/psi_var - 0.5*psi_var.log() - np.log(2) 
    #     psb = lnpsb.exp().mean(0)

    #     for k in range(1,self.n):
    #         psb[...,k] = psb[...,k]*(1-psb[...,:k].sum(-1))
            
    #     res = 1-psb.sum(-1,True)
    #     psb = torch.cat((psb,res),dim=-1)
    #     return psb

    # def forward(self,pX):
    #     X = pX.mean()
    #     EXXT = pX.EXXT()
    #     if self.pad_X is True:
    #         EXXT = torch.cat((EXXT,X),dim=-1)
    #         X = torch.cat((X,torch.ones(X.shape[:-2]+(1,1))),dim=-2)
    #         EXXT = torch.cat((EXXT,X.transpose(-2,-1)),dim=-2)

    #     psi_bar = self.beta.mean().transpose(-2,-1)@X.unsqueeze(-3)
    #     psi_var = (self.beta.EXXT()*EXXT.unsqueeze(-3)).sum(-1).sum(-1) - psi_bar.pow(2) 

    #     pgb = 1.0
    #     pgc = (psi_var + psi_bar.pow(2)).sqrt()  # should have shape (sample x batch x n)
    #     Ew = 1/2.0/pgc*(pgc/2.0).tanh()

    #     nat1 = 0.5 + psi_bar/psi_var
    #     nat2 = Ew + 1.0/psi_var

    #     lnpsb = 0.5*nat1.pow(2)/nat2 - 0.5*nat2.log() - 0.5*psi_bar.pow(2)/psi_var - 0.5*psi_var.log() - pgb*np.log(2)

    #     psb = torch.sigmoid(lnpsb)
    #     for k in range(1,self.n):
    #         psb[...,k] = psb[...,k]*(1-psb[...,:k].sum(-1))
            
    #     res = 1-psb.sum(-1,True)
    #     psb = torch.cat((psb,res),dim=-1)
    #     return psb

    # def backward(self,psb,iters=1):
    #     N = psb.sum(-1,True)-(psb.cumsum(-1)-psb)
    #     YmN = psb-N/2.0   # should have shape (sample x batch x n) 
    #     N = N[...,:-1]
    #     YmN = YmN[...,:-1]
    #     Ew = 0.25*torch.ones(YmN.shape,requires_grad=False)
    #     pgb = N

    #     beta = self.beta.mean()
    #     bbt = self.beta.EXXT()
    #     invSigmamu = (YmN.view(YmN.shape + (1,1))*self.beta.mean()).sum(-3)
    #     for i in range(iters):
    #         invSigma = (self.beta.EXXT()*Ew.view(Ew.shape+ (1,1))).sum(-3)
    #         Sigma = invSigma.inverse()
    #         mu = Sigma@invSigmamu
    #         psi_bar = (mu.unsqueeze(-3)*self.beta.mean()).sum(-1).sum(-1)
    #         psi_var = ((Sigma+mu@mu.transpose(-2,-1)).unsqueeze(-3)*self.beta.EXXT()).sum(-1).sum(-1) - psi_bar.pow(2)
    #         Ew = pgb/2.0/psi_var*(psi_var/2.0).tanh()

    #     pX = MultivariateNormal_vector_format(mu = mu, Sigma = Sigma, invSigma = invSigma, invSigmamu = invSigmamu)
    #     return pX            

