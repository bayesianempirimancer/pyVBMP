
import torch
import numpy as np
from .dists.Hierarchical_Dirichlet import Hierarchical_Dirichlet
from .dists.NormalInverseWishart import NormalInverseWishart

class HHMM():
    # 
    def __init__(self, obs_dist, level_dims = (),ptemp=1):        
        self.obs_dist = obs_dist
        self.event_shape = level_dims + obs_dist.batch_shape[-1:] 
        self.event_dim = len(self.event_shape)
        self.batch_shape = tuple(obs_dist.batch_shape[:-1])
        self.batch_dim = len(self.batch_shape)

        self.transition = Hierarchical_Dirichlet(dims = self.event_shape, batch_shape = self.batch_shape + self.event_shape)
        self.initial = Hierarchical_Dirichlet(dims = self.event_shape, batch_shape = self.batch_shape)

        self.left_sum_list = list(range(-2*self.event_dim,-self.event_dim))
        self.right_sum_list = list(range(-self.event_dim,0))
        self.ptemp = ptemp
        self.logZ = torch.tensor(-torch.inf)
        self.ELBO_last = torch.tensor(-torch.inf)

    def unsqueeze_left(self,X):
        return X.view(X.shape[:-self.event_dim] + self.event_dim*(1,) + X.shape[-self.event_dim:])
        
    def unsqueeze_right(self,X):
        return X.view(X.shape + self.event_dim*(1,))

    def stable_logsumexp(self,x,dim=None,keepdim=False):
        if isinstance(dim,int):
            xmax = x.max(dim=dim,keepdim=True)[0]
            if(keepdim):
                return xmax + (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
            else:
                return xmax.squeeze(dim) + (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
        else:
            xmax = x
            for d in dim:
                xmax = xmax.max(dim=d,keepdim=True)[0]
            if(keepdim):
                return xmax + (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
            else:
                x = (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
                for d in dim:
                    xmax = xmax.squeeze(d)
                return xmax + x
            
    def forward_step(self,logits,observation_logits):
        return self.stable_logsumexp(self.unsqueeze_right(logits) + self.unsqueeze_left(observation_logits) + self.transition.loggeomean(),self.left_sum_list)
    
    def backward_step(self,logits,observation_logits):
        return self.stable_logsumexp(self.unsqueeze_left(logits) + self.unsqueeze_left(observation_logits) + self.transition.loggeomean(),self.right_sum_list)


    def forward_backward_steps(self,X,T): 
        sample_shape = X.shape[:-1]
        fw_logits = torch.zeros((T,) + sample_shape[1:] + self.event_shape,requires_grad=False)

        fw_logits[0] = self.forward_step(self.initial.loggeomean(),self.obs_logits(X,0))
        self.stable_logsumexp(self.unsqueeze_right(self.initial.loggeomean()) + self.transition.loggeomean() + self.unsqueeze_left(self.obs_logits(X,0)),self.left_sum_list)
        for t in range(1,T):
            fw_logits[t] = self.forward_step(fw_logits[t-1],self.obs_logits(X,t))

        logZ = self.stable_logsumexp(fw_logits[-1],self.right_sum_list)
        SEzz = torch.zeros(fw_logits.shape[1:]+self.event_shape,requires_grad=False)

        for t in range(T-2,-1,-1):
            ### Backward Smoothing
            temp = self.unsqueeze_right(fw_logits[t]) + self.transition.loggeomean() 
            xi_logits = (temp - self.stable_logsumexp(temp,self.left_sum_list,keepdim=True)) + self.unsqueeze_left(fw_logits[t+1])
            fw_logits[t] = self.stable_logsumexp(xi_logits,self.right_sum_list)
            SEzz = SEzz + (xi_logits - self.stable_logsumexp(xi_logits,self.left_sum_list + self.right_sum_list, keepdim=True)).exp()
                        
        # Now do the initial step
        temp = self.unsqueeze_right(self.initial.loggeomean()) + self.transition.loggeomean() 
        xi_logits = (temp - self.stable_logsumexp(temp,self.left_sum_list,keepdim=True)) + self.unsqueeze_left(fw_logits[0])
        SEz0 = self.stable_logsumexp(xi_logits,self.right_sum_list)
        SEz0 = (SEz0-self.stable_logsumexp(SEz0,self.right_sum_list,True)).exp()
        SEzz = SEzz + (xi_logits - self.stable_logsumexp(xi_logits,self.left_sum_list + self.right_sum_list, keepdim=True)).exp()

        fw_logits =  ((fw_logits - fw_logits.max(-1,keepdim=True)[0])/self.ptemp).exp()
        fw_logits = fw_logits/fw_logits.sum(self.right_sum_list,keepdim=True)
        if fw_logits.isnan().any():
            print('HMM:  NaN in p')
        return fw_logits, SEzz, SEz0, logZ  # Note that only Time has been integrated out of sufficient statistics and fw_logits is now p(z_t|x_{0:T-1})
    
    def forward_backward_logits(self,obs_logits):
        # Assumes that time is in the first dimension of the observation
        # On input fw_logits = observation_logits. 
#        T = observation_logits.shape[0]
        T = obs_logits.shape[0]
        fw_logits = torch.zeros(obs_logits.shape[:-self.event_dim] + self.event_shape,requires_grad=False)

#        logits = self.transition.loggeomean() + observation_logits.unsqueeze(-2)
#        fw_logits = torch.zeros(observation_logits.shape,requires_grad=False)
#        fw_logits[0] = (logits[0] + self.initial.loggeomean().unsqueeze(-1)).logsumexp(-2)

        fw_logits[0] = self.stable_logsumexp(self.unsqueeze_left(self.initial.loggeomean()) + self.transition.loggeomean() + self.unsqueeze_right(obs_logits[0]),self.left_sum_list)
        for t in range(1,T):
            fw_logits[t] = self.stable_logsumexp(self.unsqueeze_right(fw_logits[t-1]) + self.transition.loggeomean() + self.unsqueeze_left(obs_logits[t]),self.left_sum_list)
        logZ = self.stable_logsumexp(fw_logits[-1],self.right_sum_list)
        SEzz = torch.zeros(fw_logits.shape[1:]+self.event_shape,requires_grad=False)
        for t in range(T-2,-1,-1):
            ### Backward Smoothing
            temp = self.unsqueeze_right(fw_logits[t]) + self.transition.loggeomean() 
            xi_logits = (temp - self.stable_logsumexp(temp,self.left_sum_list,keepdim=True)) + self.unsqueeze_left(fw_logits[t+1])
            fw_logits[t] = self.stable_logsumexp(xi_logits,self.right_sum_list)
            SEzz = SEzz + (xi_logits - self.stable_logsumexp(xi_logits,self.left_sum_list + self.right_sum_list, keepdim=True)).exp()
                        
        # Now do the initial step
        # Backward Smoothing
        temp = self.unsqueeze_right(self.initial.loggeomean()) + self.transition.loggeomean() 
        xi_logits = (temp - self.stable_logsumexp(temp,self.left_sum_list,keepdim=True)) + self.unsqueeze_left(fw_logits[0])
        SEz0 = self.stable_logsumexp(xi_logits,self.right_sum_list)
        SEz0 = (SEz0-self.stable_logsumexp(SEz0,self.right_sum_list,True)).exp()
        SEzz = SEzz + (xi_logits - self.stable_logsumexp(xi_logits,self.left_sum_list + self.right_sum_list, keepdim=True)).exp()
        # Backward inference
        # bw_logits = bw_logits.unsqueeze(-2) + logits[0]  
        # xi_logits = self.initial.loggeomean().unsqueeze(-1) + bw_logits
        # xi_logits = (xi_logits - xi_logits.logsumexp([-1,-2], keepdim=True))
        # SEzz = SEzz + xi_logits.exp()
        # bw_logits = self.initial.loggeomean() + bw_logits.logsumexp(-1)
        # SEz0 = (bw_logits - bw_logits.max(-1,keepdim=True)[0]).exp()
        # SEz0 = SEz0/SEz0.sum(-1,True)      

        fw_logits =  ((fw_logits - fw_logits.max(-1,keepdim=True)[0])/self.ptemp).exp()
        fw_logits = fw_logits/fw_logits.sum(self.right_sum_list,keepdim=True)
        if fw_logits.isnan().any():
            print('HMM:  NaN in p')

        return fw_logits, SEzz, SEz0, logZ  # Note that only Time has been integrated out of sufficient statistics
                                            # and the despite the name fw_logits is posterior probability of states, i.e. self.p
    def assignment_pr(self):
        return self.p
    
    def assignment(self):
        return self.p.argmax(-1)

    def obs_logits(self,X,t=None):
        if t is not None:
            ELL =  self.obs_dist.Elog_like(X[t].unsqueeze(-1-self.obs_dist.event_dim))  # has shape sample x event[-1]
        else:
            ELL = self.obs_dist.Elog_like(X.unsqueeze(-1-self.obs_dist.event_dim))
        ELL = ELL.view(ELL.shape[:-1] + (1,)*(self.event_dim-1) + self.event_shape[-1:])  # has shape sample x event])
        return ELL

    def update_states(self,X,T=None):
        # updates states and stores in self.p
        # also updates sufficient statistics of Markov process (self.SEzz, self.SEz0) and self.logZ and self.sumlogZ
        if T is None:
            self.p, SEzz, SEz0, logZ = self.forward_backward_logits(self.obs_logits(X))  # recall that time has been integrated out except for p.
        else:
            self.p, SEzz, SEz0, logZ = self.forward_backward_steps(X,T)  # recall that time has been integrated out except for p.

        NA = self.p.sum(0) # also integrate out time for NA
        sample_dims = tuple(range(len(NA.shape[:-self.event_dim-self.batch_dim])))
        NA = NA.sum(sample_dims)
        SEzz = SEzz.sum(sample_dims)
        SEz0 = SEz0.sum(sample_dims)
        logZ = logZ.sum(sample_dims)

        return SEzz, SEz0, NA, logZ
        
    def update_markov_parms(self,SEzz, SEz0, lr=1.0,beta=None):
        self.transition.ss_update(SEzz,lr=lr,beta=beta)
        self.initial.ss_update(SEz0,lr=lr,beta=beta)

    def update_obs_parms(self,X,lr=1.0,beta=None):
        self.obs_dist.raw_update(X.unsqueeze(-1-self.obs_dist.event_dim),p=self.p.sum(self.right_sum_list[:-1]),lr=lr,beta=beta)

    # def update_parms(self,X,lr=1.0):
    #     self.transition.ss_update(self.SEzz,lr)
    #     self.initial.ss_update(self.SEz0,lr)
    #     self.update_obs_parms(X,self.p,lr)

    def update(self,X,iters=1,lr=1.0,beta=None,verbose=False):   

        for i in range(iters):
            SEzz, SEz0, self.NA, self.logZ = self.update_states(X,T=X.shape[0])
            self.KLqprior_last = self.KLqprior()
            self.update_markov_parms(SEzz, SEz0, lr=lr,beta=beta)
            self.update_obs_parms(X,lr=lr,beta=beta)
            
            if verbose:
                ELBO = self.ELBO().sum()
                print('Percent Change in ELBO = %f' % ((ELBO-self.ELBO_last)/np.abs(self.ELBO_last)*100))
                self.ELBO_last = ELBO
    def Elog_like(self,X):  # assumes that p is up to date
        ELL = (self.obs_dist.Elog_like(X.unsqueeze(-1-self.obs_dist.event_dim))*self.p).sum(-1)
        for i in range(self.event_dim - 1):
            ELL = ELL.sum(-1)
        return ELL        

    def KLqprior(self):
        return self.obs_dist.KLqprior().sum(-1) + self.transition.KLqprior().sum(self.right_sum_list) + self.initial.KLqprior()  # assumes default event_dim = 1

    def ELBO(self):
        return self.logZ - self.KLqprior() 

    def event_average_f(self,function_string,keepdim=False):
        return self.event_average(eval('self.obs_dist.'+function_string)(),keepdim)

    def average_f(self,function_string,keepdim=False):
        return self.average(eval('self.obs_dist.'+function_string)(),keepdim)

    def average(self,A,keepdim=False):  # returns sample_shape 
        # A is mix_batch_shape + mix_event_shape 
        return (A*self.p).sum(-1,keepdim)

    ### Compute special expectations used for VB inference
    def event_average(self,A,keepdim=False):  # returns sample_shape + W.event_shape
        # A is mix_batch_shape + mix_event_shape + event_shape

        out = (A*self.p.view(self.p.shape + (1,)*self.obs_dist.event_dim)).sum(-self.obs_dist.event_dim-1,keepdim)
        for i in range(self.event_dim-1):
            out = out.sum(-self.obs_dist.event_dim-1,keepdim)
        return out


# from matplotlib import pyplot as plt
# from dists import NormalInverseWishart
# obs_dist = NormalInverseWishart(mu_0 = torch.zeros(10,2))
# model = HHMM(obs_dist,level_dims=(3,5,))
# self = model
# X = torch.randn(99,11,2)
# model.update(X,iters=20,lr=1,verbose=True)
