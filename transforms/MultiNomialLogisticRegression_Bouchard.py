import torch
import dists.MVN_ard as MVN_ard
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format

def lmbda(xi):
    return 0.25/xi*(0.5*xi).tanh()

def log_sigmoid(xi):
    return -(1.0+(-xi).exp()).log()

class MultiNomialLogisticRegression_Bouchard():
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

    def raw_update(self,X,Y,iters = 4, p=None,lr=1.0,beta=None,verbose=False):
        # Assumes X is sample x batch x p 
        #     and Y is sample x batch x n
        sample_shape = X.shape[:-self.event_dim-self.batch_dim+1]
        sample_dims = tuple(range(len(sample_shape)))
        ELBO = self.ELBO_last
        # if self.pad_X is True:
        #     X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)

        if self.pad_X is True:
            EX = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        else:
            EX = X
        EX = EX.view(EX.shape[:-1] + (1,) + EX.shape[-1:] + (1,))
        EXXT = EX*EX.transpose(-2,-1)
        N = Y.sum(-1,True).view(Y.shape[:-1] + (1,1,1))
        Y = Y.view(Y.shape+(1,1))

        if p is None:
            SEyx = ((Y-0.5*N)*EX).sum(sample_dims)  
        else:
            SEyx = (((Y-0.5*N)*EX)*p.view(p.shape+(1,1,1))).sum(sample_dims)

        alpha = torch.tensor((self.n-2)/4.0).expand((1,1,1))
        xi = (self.beta.EXXT()*EXXT).sum((-2,-1),True) - 2.0*alpha*(self.beta.EX()*EX).sum((-2,-1),True) + alpha.pow(2)
        xi = xi.sqrt()  # shape = sample x batch x (n,1,1)

        for i in range(iters):
            alpha = ((self.n-2)/4.0 + (lmbda(xi)*(self.beta.EX()*EX).sum((-2,-1),True)).sum(-3,True))/lmbda(xi).sum(-3,True)
            xi = (self.beta.EXXT()*EXXT).sum((-2,-1),True) - 2.0*alpha*(self.beta.EX()*EX).sum((-2,-1),True) + alpha.pow(2)
            xi = xi.sqrt()  # shape = sample x batch x (n,1,1)

            if p is None:
                SExx = 2*(N*lmbda(xi)*EXXT).sum(sample_dims)  # batch x n x p x p
                SEyx_star = 2*(alpha*N*lmbda(xi)*EX).sum(sample_dims)
            else:
                SExx = 2*(lmbda(xi)*EXXT*p.view(p.shape+(1,1,1))).sum(sample_dims)
                SEyx_star = 2*(alpha*N*lmbda(xi)*EX*p.view(p.shape+(1,1,1))).sum(sample_dims)

            if verbose is True: # Valid only after xi update
                ELBO = (SEyx*self.beta.mean()).sum((-3,-2,-1)) - (alpha*N).sum((-3,-2,-1)).sum(sample_dims) - self.KLqprior()
                ELBO = ELBO + 0.5*(N*(xi+alpha)).sum((-3,-2,-1)).sum(sample_dims)
                ELBO = ELBO + (N*log_sigmoid(-xi)).sum((-3,-2,-1)).sum(sample_dims)
                print("MNLR Percent Change in ELBO: ",((ELBO-self.ELBO_last)/self.ELBO_last.abs()*100))
                self.ELBO_last = ELBO

            self.beta.ss_update(SExx,SEyx+SEyx_star,lr=lr,beta=beta)

    def update(self,pX,Y,iters=1,p=None,lr=1,beta=None,verbose=False):  
        sample_shape = pX.shape[:-self.event_dim-self.batch_dim+1]
        sample_dims = tuple(range(len(sample_shape)))
        # Here pX is assumed to be a probability distribution that has supports EXXT()
        ELBO = self.ELBO_last

        EXXT = pX.EXXT().unsqueeze(-3)  # sample x batch x 1 x p x p 
        EX = pX.mean().unsqueeze(-3)    # sample x batch x 1 x p x 1
        N = Y.sum(-1,True).view(Y.shape[:-1] + (1,1,1))
        Y = Y.view(Y.shape+(1,1))

        if self.pad_X is True:
            EXXT = torch.cat((EXXT,EX),dim=-1) 
            EX = torch.cat((EX,torch.ones(EX.shape[:-2]+(1,1))),dim=-2)
            EXXT = torch.cat((EXXT,EX.transpose(-2,-1)),dim=-2)        
        if p is None:
            SEyx = ((Y-0.5*N)*EX).sum(sample_dims)  
        else:
            SEyx = (((Y-0.5*N)*EX)*p.view(p.shape+(1,1,1))).sum(sample_dims)

        alpha = torch.tensor((self.n-2)/4.0).expand((1,1,1))
        xi = (self.beta.EXXT()*EXXT).sum((-2,-1),True) - 2.0*alpha*(self.beta.EX()*EX).sum((-2,-1),True) + alpha.pow(2)
        xi = xi.sqrt()  # shape = sample x batch x (n,1,1)

        for i in range(iters):
            alpha = ((self.n-2)/4.0 + (lmbda(xi)*(self.beta.EX()*EX).sum((-2,-1),True)).sum(-3,True))/lmbda(xi).sum(-3,True)
            xi = (self.beta.EXXT()*EXXT).sum((-2,-1),True) - 2.0*alpha*(self.beta.EX()*EX).sum((-2,-1),True) + alpha.pow(2)
            xi = xi.sqrt()  # shape = sample x batch x (n,1,1)

            if p is None:
                SExx = 2*(N*lmbda(xi)*EXXT).sum(sample_dims)  # batch x n x p x p
                SEyx_star = 2*(alpha*N*lmbda(xi)*EX).sum(sample_dims)
            else:
                SExx = 2*(lmbda(xi)*EXXT*p.view(p.shape+(1,1,1))).sum(sample_dims)
                SEyx_star = 2*(alpha*N*lmbda(xi)*EX*p.view(p.shape+(1,1,1))).sum(sample_dims)

            if verbose is True: # Valid only after xi update
                ELBO = (SEyx*self.beta.mean()).sum((-3,-2,-1)) - (alpha*N).sum((-3,-2,-1)).sum(sample_dims) - self.KLqprior()
                ELBO = ELBO + 0.5*(N*(xi+alpha)).sum((-3,-2,-1)).sum(sample_dims)
                ELBO = ELBO + (N*log_sigmoid(-xi)).sum((-3,-2,-1)).sum(sample_dims)
                print("MNLR Percent Change in ELBO: ",((ELBO-self.ELBO_last)/self.ELBO_last.abs()*100))
                self.ELBO_last = ELBO

            self.beta.ss_update(SExx,SEyx+SEyx_star,lr=lr,beta=0)


    def forward(self, pX):
        sample_shape = pX.mean().shape[:-2]
        Yt = torch.eye(self.n+1)
        for i in range(len(sample_shape)):
            Yt = Yt.unsqueeze(-2)
        log_p = self.Elog_like_given_pX_pY(pX,Yt).movedim(0,-1)
        Res = log_p.logsumexp(-1,True)
        return log_p - Res, Res.squeeze(-1)

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
            return self.beta.mean()[...,:-1,0]
        else:
            return self.beta.mean()[...,0]
    
    def bias(self):
        if self.pad_X is True:
            return self.beta.mean()[...,-1:,0]
        else:
            return torch.tensor(0.0).unsqueeze(-1)

    def Elog_like_given_pX_pY(self,pX,Y,iters=2):  # assumes pX is in vector format

        if self.pad_X is False:
            Ephiphi = (self.beta.EXXT()*pX.EXXT().unsqueeze(-3)).sum((-2,-1))
            Ephi = (self.beta.EX()*pX.mean().unsqueeze(-3)).sum((-2,-1))
        else: 
            Ephiphi = (self.beta.EXXT()[...,:-1,:-1]*pX.EXXT().unsqueeze(-3)).sum((-2,-1))
            Ephiphi = Ephiphi + 2*(self.beta.EX()[...,:-1,:]*pX.mean().unsqueeze(-3)).sum((-2,-1)) 
            Ephiphi = Ephiphi + self.beta.EX()[...,-1,-1]
            Ephi = (self.beta.EX()[...,:-1,:]*pX.mean().unsqueeze(-3)).sum((-2,-1)) + self.beta.EX()[...,-1,-1]

        N = Y.sum(-1,True)
        Y = Y.view(Y.shape)
        # N, Y, Ephiphi, and Ephi are now sample x batch x n            

        alpha = torch.tensor((self.n-2)/4.0).unsqueeze(-1)
        xi = Ephiphi - 2.0*alpha*Ephi + alpha.pow(2)
        xi = xi.sqrt()  # shape = sample x batch x (n)

        for i in range(iters-1):
            alpha = ((self.n-2)/4.0 + (lmbda(xi)*Ephi).sum(-1,True))/lmbda(xi).sum(-1,True)
            xi = Ephiphi - 2.0*alpha*Ephi + alpha.pow(2)
            xi = xi.sqrt()  # shape = sample x batch x (n)

        ELL = ((Y-0.5*N)*Ephi).sum(-1) - (alpha*N).squeeze(-1)
        ELL = ELL + 0.5*(N*(xi+alpha)).sum(-1)
        ELL = ELL + (N*log_sigmoid(-xi)).sum(-1)
        return ELL

    def Elog_like(self,X,Y,iters=2):
        X = X.unsqueeze(-1).unsqueeze(-3)
        if self.pad_X is False:
            Ephiphi = (X.mT@self.beta.EXXT()@X).squeeze(-1).squeeze(-1)
            Ephi = (self.beta.EX()*X).sum((-2,-1))
        else: 
            Ephiphi = (X.mT@self.beta.EXXT()[...,:-1,:-1]@X).squeeze(-1).squeeze(-1)
            Ephiphi = Ephiphi + 2*(self.beta.EX()[...,:-1,:]*X).sum((-2,-1)) 
            Ephiphi = Ephiphi + self.beta.EX()[...,-1,-1]
            Ephi = (self.beta.EX()[...,:-1,:]*X).sum((-2,-1)) + self.beta.EX()[...,-1,-1]

        N = Y.sum(-1,True)

        # N, Y, Ephiphi, and Ephi are now sample x batch x n            

        alpha = torch.tensor((self.n-2)/4.0).unsqueeze(-1)
        xi = Ephiphi - 2.0*alpha*Ephi + alpha.pow(2)
        xi = xi.sqrt()  # shape = sample x batch x (n)

        for i in range(iters-1):
            alpha = ((self.n-2)/4.0 + (lmbda(xi)*Ephi).sum(-1,True))/lmbda(xi).sum(-1,True)
            xi = Ephiphi - 2.0*alpha*Ephi + alpha.pow(2)
            xi = xi.sqrt()  # shape = sample x batch x (n)

        ELL = ((Y-0.5*N)*Ephi).sum(-1) - (alpha*N).squeeze(-1)
        ELL = ELL + 0.5*(N*(xi+alpha)).sum(-1)
        ELL = ELL + (N*log_sigmoid(-xi)).sum(-1)

        return ELL

    def backward(self,pY,like_X=None):
        if like_X is None:
            p = self.p - self.pad_X
            like_X = MultivariateNormal_vector_format(invSigmamu = torch.zeros((pY.ndim-1)*(1,)+(p,1)),invSigma = torch.eye(p).expand((pY.ndim-1)*(1,)+(p,p)))
        invSigma, invSigmamu, Sigma, mu, Res = self.Elog_like_X(like_X,pY) # returns invSigma, invSigmamu, Res
        return MultivariateNormal_vector_format(invSigma=invSigma, invSigmamu=invSigmamu, Sigma=Sigma, mu=mu), Res

    def Elog_like_X(self, Y, like_X=None, iters=2):
        # Here pX is a prior or likelihood for X.  This is necessary because
        # the likelihood function for X has a singular covariance matrix inherited from the 
        # fact that the softmax function is shift invariant.  This situation is only made worse
        # when the dimension of X is greater than the dimension of Y.  The simple solution to this 
        # problem is to use a prior or likelihood function for X that is not singular.  
        sample_shape = Y.shape[:-self.event_dim-self.batch_dim+1]
        sample_dims = tuple(range(len(sample_shape)))

        N = Y.sum(-1,True).view(Y.shape[:-1] + (1,1,1))
        Y = Y.view(Y.shape+(1,1))

        if self.pad_X is True:
            if like_X is None:
                like_X = MultivariateNormal_vector_format(invSigmamu = torch.zeros(self.p-1,1),invSigma = torch.eye(self.p-1))
            invSigmamu = like_X.invSigmamu + ((Y-0.5*N)*self.beta.mean()[...,:-1,-1:]).sum(-3,True)
            invSigma = like_X.invSigma
        else:
            if like_X is None:
                like_X = MultivariateNormal_vector_format(invSigmamu = torch.zeros(self.p,1),invSigma = torch.eye(self.p))
            invSigmamu = like_X.invSigmamu + ((Y-0.5*N)*self.beta.mean()).sum(-3,True)
            invSigma = like_X.invSigma 

        def get_expectations(invSigmamu,invSigma):
            EXXT = torch.linalg.inv(invSigma)
            EX = EXXT@invSigmamu
            EXXT = EXXT + EX@EX.mT

            if self.pad_X is True:
                EXXT = torch.cat((EXXT,EX),dim=-1)
                EX = torch.cat((EX,torch.ones(EX.shape[:-2]+(1,1))),dim=-2)
                EXXT = torch.cat((EXXT,EX.mT),dim=-2)
            return EX, EXXT

        EX, EXXT = get_expectations(invSigmamu,invSigma)

        alpha = torch.tensor((self.n-2)/4.0).expand((1,1,1))
        xi = (self.beta.EXXT()*EXXT).sum((-2,-1),True) - 2.0*alpha*(self.beta.EX()*EX).sum((-2,-1),True) + alpha.pow(2)
        xi = xi.sqrt()  # shape = sample x batch x (n,1,1)
        if self.pad_X is True:
            invSigmamu = like_X.invSigmamu + ((Y-0.5*N+alpha*N*lmbda(xi))*self.beta.mean()[...,:-1,-1:]).sum(-3,True)
            invSigmamu = invSigmamu - (lmbda(xi)*self.beta.EXXT()[...,:-1,-1:]).sum(-3,True)
            invSigma = like_X.invSigma + 2*(lmbda(xi)*self.beta.EXXT()[...,:-1,:-1]).sum(-3,True)
        else: 
            invSigmamu = like_X.invSigmamu + ((Y-0.5*N+alpha*N*lmbda(xi))*self.beta.mean()).sum(-3,True) 
            invSigma = like_X.invSigma + 2*(lmbda(xi)*self.beta.EXXT()).sum(-3,True)
        
        for i in range(iters-1):

            EX, EXXT = get_expectations(invSigmamu,invSigma)
            alpha = ((self.n-2)/4.0 + (lmbda(xi)*(self.beta.EX()*EX).sum((-2,-1),True)).sum(-3,True))/lmbda(xi).sum(-3,True)
            xi = (self.beta.EXXT()*EXXT).sum((-2,-1),True) - 2.0*alpha*(self.beta.EX()*EX).sum((-2,-1),True) + alpha.pow(2)
            xi = xi.sqrt()  # shape = sample x batch x (n,1,1)
            if self.pad_X is True:
                invSigmamu = like_X.invSigmamu + ((Y-0.5*N+alpha*N*lmbda(xi))*self.beta.mean()[...,:-1,-1:]).sum(-3,True)
                invSigmamu = invSigmamu - (lmbda(xi)*self.beta.EXXT()[...,:-1,-1:]).sum(-3,True)
                invSigma = like_X.invSigma + 2*(lmbda(xi)*self.beta.EXXT()[...,:-1,:-1]).sum(-3,True)
            else: 
                invSigmamu = like_X.invSigmamu + ((Y-0.5*N+alpha*N*lmbda(xi))*self.beta.mean()).sum(-3,True) 
                invSigma = like_X.invSigma + 2*(lmbda(xi)*self.beta.EXXT()).sum(-3,True)

        return invSigmamu, invSigma

    def log_predict(self,X):  # slower than log_predict_1 because some calculations are repeated
        sample_shape = X.shape[:-1]
        Yt = torch.eye(self.n)
        for i in range(len(sample_shape)):
            Yt = Yt.unsqueeze(-2)        
        return self.Elog_like(X,Yt).movedim(0,-1)

    def log_forward(self,pX):  # slower than log_predict_1 because some calculations are repeated
        sample_shape = pX.shape[:-2]
        Yt = torch.eye(self.n)
        for i in range(len(sample_shape)):
            Yt = Yt.unsqueeze(-2)        
        return self.Elog_like_given_pX_pY(pX,Yt).movedim(0,-1)

    def loggeomean(self,X):
        return self.log_predict(X)

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

