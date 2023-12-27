# Variational Bayesian Expectation Maximization for a varitaion on recurrent switching 
# linear dynamical system model.  Here there are two sets of latent variables a discrete
# variable s and a continuous variable x.  
#
# Transition dynamics are bipartite, i.e. p(s',x'|s,x) = p(s'|s,x)p(x'|x,s)
# We model p(s'|s,x) as a batch of logistic regressors, i.e. p(s'|s,x) = softmax(W_s x + b_s)
#      and p(x'|x,s) as a batch of linear dynamical systems, i.e. p(x'|x,s) = N(A_s x + B_s, Sigma_s)
#  
# This hidden structure leaves open the possibility for a variety of observation models:
#    Mix Linear Model:  p(y|x,s) = N(C_s x + D_s, R_s)       ::  
#    Role Based Model:  p(y|r,s)p(r|x) = N(mu_rs, Sigma_rs)  ::  
#    Linear Role Model:  p(y|r,s)
import torch
import transforms.dMixtureofLinearTransforms as dMixtureofLinearTransforms
import transforms.MatrixNormalWishart as MatrixNormalWishart
import dists.NormalInverseWishart as NormalInverseWishart
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format

class NonLinearDynamicalSystems():
    def __init__(self, obs_shape, hidden_dim, mixture_dim, batch_shape = (), pad_X = False):

        self.obs_shape = obs_shape
        self.obs_dim = obs_shape[-1]
        self.hidden_dim = hidden_dim
        self.mixture_dim = mixture_dim
        self.pad_X = pad_X
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.logZ = torch.tensor(0.0,requires_grad=False)

        self.x0 = NormalInverseWishart(event_shape = (hidden_dim,), batch_shape = batch_shape) 
        self.A = dMixtureofLinearTransforms(hidden_dim, hidden_dim, mixture_dim, batch_shape=(), pad_X=False, type = 'Gamma')
        self.B = MatrixNormalWishart(obs_shape + (hidden_dim,), batch_shape, pad_X = pad_X)

#        self.set_expectation_parms()
        self.px = None
        self.log2pi = torch.log(2*torch.tensor(torch.pi,requires_grad=False))

        self.ELBO_last = -torch.tensor(torch.inf,requires_grad=False)

    def update(self,y,p=None,iters=1,lr=1.0,verbose=False):

        for i in range(iters):
            self.update_latents(y,u,r)
            ELBO = self.ELBO().sum()
            self.ss_update(p=p,lr=lr)
            self.update_obs_model(lr=lr)
            if verbose:
                print("SLDS Percent Change in ELBO %f" % ((ELBO-self.ELBO_last)/ELBO.abs()*100))
            self.ELBO_last = ELBO

    def ss_update(self,p=None,lr=1.0):
#        self.set_expectation_parms()
        raise NotImplementedError

    def update_latents(self,y,p=None,lr=1.0):

        if self.pX is None:
            self.pX = MultivariateNormal_vector_format(mu = torch.zeros(y.shape[:-2]+(self.hidden_dim,1),requires_grad=False))

        self.pX = self.forward_backward_loop(y)  # updates and stores marginal distribution over hidden states 
                                                 # at each point in time.  shape = T x sample_shape x (hidden_dim,1)

    def KLqprior(self):  # returns batch_size
        KL = self.x0.KLqprior() + self.A.KLqprior()
        for i in range(len(self.offset)):
            KL = KL.squeeze(-1)
        return KL + self.obs_model.KLqprior()

    def ELBO(self):  # returns batch_size
        logZ = self.logZ
        while logZ.ndim > self.batch_dim:
            logZ = logZ.sum(0)
        return logZ - self.KLqprior()

    def log_likelihood_function(self,Y,t=None):
        if t is None:
            return self.B.backward(Y)
        else:
            return self.B.backward(Y[t])

    def forward_step(self,invSigma, invSigmamu, Residual, invSigma_like, invSigmamu_like, Residual_like, U): 
        # On output Residual returns log p(y_t|y_{t-1},y_{t-2},...,y_0) 
        Sigma_tm1_tm1 = (invSigma + self.ATQA_x_x).inverse() # sometimes called SigmaStar_t

        invSigmamu_t = invSigmamu_like + self.QA_xp_u @ U  
        invSigmamu_tm1 = invSigmamu - self.ATQA_x_u @ U    

        invSigma = invSigma_like + self.invQ  -  self.QA_xp_x @ Sigma_tm1_tm1 @ self.QA_xp_x.transpose(-1,-2)  
        invSigmamu = invSigmamu_t + self.QA_xp_x @ Sigma_tm1_tm1 @ invSigmamu_tm1

        Residual = Residual + Residual_like - 0.5*(U.transpose(-2,-1)@self.ATQA_u_u@U).squeeze(-1).squeeze(-1) 
        Residual = Residual + 0.5*self.A.ElogdetinvSigma() # cancels with below - 0.5*(self.hidden_dim)*np.log(2*np.pi)

        Residual = Residual + 0.5*(invSigmamu_tm1.transpose(-2,-1) @ Sigma_tm1_tm1 @ invSigmamu_tm1).squeeze(-1).squeeze(-1) 
        Residual = Residual + 0.5*Sigma_tm1_tm1.logdet() # cancels with above + 0.5*(self.hidden_dim)*np.log(2*np.pi)

        mu = invSigma.inverse()@invSigmamu
        post_Residual = -0.5*(mu*invSigmamu).squeeze(-1).sum(-1) + 0.5*invSigma.logdet() - 0.5*self.hidden_dim*torch.log(2*torch.tensor(torch.pi,requires_grad=False))
        Residual = Residual - post_Residual # so that Residual is log p(y_t|y_{t-1},y_{t-2},...,y_0)

        return invSigma, invSigmamu, post_Residual, Residual, Sigma_tm1_tm1

    # def backward_recursion(self, invGamma, invGammamu, invSigma, invSigmamu, u):
    #     # here invSigma and invSigmamu summarize p(x_t| y_0:t) and are from the forward loop
    #     # invGamma and invGammamu summarize p(x_t+1|y_0:T)
    #     # u is the control input at time t+1
    #     return invGamma, invGammamu, Sigma_t_tp1

    def backward_step(self, invGamma, invGammamu,  invSigma_like, invSigmamu_like, U):

        Sigma_tp1_tp1 = (self.invQ + invSigma_like + invGamma).inverse() # sample x batch x offset x hidden_dim x hidden_dim
        invGamma = (self.ATQA_x_x - self.QA_xp_x.transpose(-1,-2) @ Sigma_tp1_tp1 @ self.QA_xp_x)  # t value
        invGammamu = -self.ATQA_x_u @ U + self.QA_xp_x.transpose(-2,-1) @ Sigma_tp1_tp1 @(self.QA_xp_u @ U + invSigmamu_like + invGammamu)

        return invGamma, invGammamu

    def backward_step_with_Residual(self, invGamma, invGammamu, Residual, invSigma_like, invSigmamu_like, Residual_like, U):

        Sigma_tp1_tp1 = (self.invQ + invSigma_like + invGamma).inverse()  # A.inverse()
        invSigmamu_tp1 = invSigmamu_like + invGammamu + self.QA_xp_u @ U
        invSigmamu_t =  -self.ATQA_x_u @ U

        invGamma = self.ATQA_x_x - self.QA_xp_x.transpose(-2,-1) @ Sigma_tp1_tp1 @ self.QA_xp_x
        invGammamu = invSigmamu_t  + self.QA_xp_x.transpose(-2,-1) @ Sigma_tp1_tp1 @ invSigmamu_tp1

        Residual = Residual + Residual_like - 0.5*(U.transpose(-2,-1)@self.ATQA_u_u@U).squeeze(-1).squeeze(-1)
        Residual = Residual + 0.5*self.A.ElogdetinvSigma() - 0.5*(self.hidden_dim)*torch.log(2*torch.tensor(torch.pi))

        Residual = Residual + (0.5*invSigmamu_tp1.transpose(-2,-1) @ Sigma_tp1_tp1 @ invSigmamu_tp1).squeeze(-1).squeeze(-1)
        Residual = Residual + 0.5*Sigma_tp1_tp1.logdet() + 0.5*self.hidden_dim*torch.log(2*torch.tensor(torch.pi,requires_grad=False))

        mu = invGamma.inverse()@invGammamu
        post_Residual = - 0.5*(mu * invGammamu).squeeze(-1).sum(-1) + 0.5*invGamma.logdet() - 0.5*torch.log(2*torch.tensor(torch.pi,requires_grad=False))*self.hidden_dim
        Residual = Residual - post_Residual

        return invGamma, invGammamu, post_Residual, Residual

    def forward_backward_combiner(self, invSigma, invSigmamu, invGamma, invGammamu):
        invSigma = invSigma + invGamma
        invSigmamu = invSigmamu + invGammamu
        Sigma = invSigma.inverse()
        mu = Sigma @ invSigmamu
        return Sigma, mu, invSigma, invSigmamu

    def forward_backward_loop(self,y,u,r):

        # To make generic we need to use event_shape and batch_shape consistently
        # define y,u,r = T x sample_shape x obs_shape 
        # p is T x sample x batch 
        #  
        # LDS is assumed to have batch_shape and event shape

        sample_shape = y.shape[:-self.event_dim-self.batch_dim-1]
        T_max = y.shape[0]

        logZ = torch.zeros(sample_shape + self.batch_shape + self.offset, requires_grad=False)
        logZ_b = None

        self.px.invSigmamu = torch.zeros(sample_shape + self.batch_shape + self.offset + (self.hidden_dim,1),requires_grad=False)
        self.px.invSigma=torch.zeros(sample_shape + self.batch_shape + self.offset +(self.hidden_dim,self.hidden_dim),requires_grad=False)
        self.px.Sigma = torch.zeros(sample_shape + self.batch_shape + self.offset +(self.hidden_dim,self.hidden_dim),requires_grad=False)
        self.px.mu = torch.zeros(sample_shape + self.batch_shape + self.offset +(self.hidden_dim,1),requires_grad=False)

        self.px.invSigma[-1] = self.x0.EinvSigma() # sample x batch x by hidden_dim by hidden_dim
        self.px.invSigmamu[-1] = self.x0.EinvSigmamu().unsqueeze(-1) # sample by batch x by hidden_dim by 1
        Residual = - 0.5*self.x0.EXTinvUX() + 0.5*self.x0.ElogdetinvSigma() - 0.5*torch.log(2*torch.tensor(torch.pi,requires_grad=False))*self.hidden_dim
        Sigma_t_tp1 = torch.zeros(sample_shape + self.batch_shape + self.offset +(self.hidden_dim,self.hidden_dim),requires_grad=False)
            # Note that initially Sigma_t_tp1 is a holding place for SigmaStar_t which is called Sigma_tm1_tm1 in the forward step

        invSigma_like, invSigmamu_like, Residual_like = self.log_likelihood_function(y,r)
        for t in range(T_max):
            self.px.invSigma[t], self.px.invSigmamu[t], Residual, logZ[t], Sigma_t_tp1[t-1] = self.forward_step(self.px.invSigma[t-1], self.px.invSigmamu[t-1], Residual, invSigma_like[t], invSigmamu_like[t], Residual_like[t], u[t])

        # now go backwards

        self.px.Sigma[-1] = self.px.invSigma[-1].inverse()
        self.px.mu[-1] = self.px.Sigma[-1] @ self.px.invSigmamu[-1]

        invGamma = torch.zeros(self.px.invSigma.shape[1:],requires_grad=False)
        invGammamu = torch.zeros(self.px.invSigmamu.shape[1:],requires_grad=False)
        # Residual = torch.zeros(Residual.shape,requires_grad=False)
        # logZ_b = torch.zeros(logZ.shape,requires_grad=False)

        for t in range(T_max-2,-1,-1):
            Sigma_t_tp1[t] = Sigma_t_tp1[t] @ self.QA_xp_x.transpose(-2,-1) @ (invGamma + invSigma_like[t+1] + self.invQ - self.QA_xp_x@Sigma_t_tp1[t]*self.QA_xp_x.transpose(-2,-1)).inverse()
            invGamma, invGammamu = self.backward_step(invGamma, invGammamu, invSigma_like[t+1], invSigmamu_like[t+1],u[t+1])
#            invGamma, invGammamu, Residual, logZ_b[t] = self.backward_step_with_Residual(invGamma, invGammamu, Residual, invSigma_like[t+1], invSigmamu_like[t+1],Residual_like[t+1],u[t+1])
            self.px.Sigma[t], self.px.mu[t], self.px.invSigma[t], self.px.invSigmamu[t] = self.forward_backward_combiner(self.px.invSigma[t], self.px.invSigmamu[t], invGamma, invGammamu )

        Sigma_t_tp1[-1] = Sigma_t_tp1[-1] @ self.QA_xp_x.transpose(-2,-1) @ (invGamma + invSigma_like[0] + self.invQ - self.QA_xp_x@Sigma_t_tp1[-1]*self.QA_xp_x.transpose(-2,-1)).inverse()#uses invSigma from tp1 which we probably should have stored 
        invGamma, invGammamu = self.backward_step(invGamma, invGammamu, invSigma_like[0], invSigmamu_like[0],u[0])
#        invGamma, invGammamu, Residual, logZ_b[-1] = self.backward_step_with_Residual(invGamma, invGammamu, Residual, invSigma_like[0], invSigmamu_like[0],Residual_like[0],u[0])
        Sigma_x0_x0 = (invGamma+self.x0.EinvSigma()).inverse()   # posterior parameters for t
        mu_x0 = Sigma_x0_x0 @ (invGammamu + self.x0.EinvSigmamu().unsqueeze(-1))

        return Sigma_t_tp1, Sigma_x0_x0, mu_x0, logZ, logZ_b

