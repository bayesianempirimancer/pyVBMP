import torch
import dists.Dirichlet as Dirichlet
from utils.torch_functions import stable_logsumexp, stable_softmax

class HMM():
    def __init__(self, obs_dist, transition_mask=None,ptemp =1.0):

        self.obs_dist = obs_dist

        self.event_dim = 1
        self.dim = obs_dist.batch_shape[-1]
        self.event_shape = obs_dist.batch_shape[-1:]
        self.batch_shape = obs_dist.batch_shape[:-1]        
        self.batch_dim = len(self.batch_shape)

        self.transition_mask = transition_mask

        alpha = torch.eye(self.dim,requires_grad=False)
        alpha = alpha + 0.5
        if transition_mask is not None:
            alpha = alpha * transition_mask
        prior_parms = {'alpha': alpha}

        self.transition = Dirichlet(self.event_shape, self.batch_shape + self.event_shape, prior_parms=prior_parms)
        self.initial = Dirichlet(self.event_shape, self.batch_shape)

        self.sumlogZ = -torch.inf
        self.p = None
        self.ptemp = ptemp
        self.logZ = torch.tensor(-torch.inf)
        self.ELBO_last = torch.tensor(-torch.inf)

    def forward_step(self,logits,observation_logits):
        return stable_logsumexp(logits.unsqueeze(-1) + observation_logits.unsqueeze(-2) + self.transition.loggeomean(),-2)
    
    def backward_step(self,logits,observation_logits):
        return stable_logsumexp(logits.unsqueeze(-2) + observation_logits.unsqueeze(-2) + self.transition.loggeomean(),-1)

    def forward_backward_steps(self,X,T): 
        temp = self.forward_step(self.initial.loggeomean(),self.obs_logits(X,0))
        fw_logits = torch.zeros((T,)+temp.shape,requires_grad=False)
        trans_logits = self.transition.loggeomean()
        fw_logits[0] = temp
        for t in range(1,T):
            fw_logits[t] = stable_logsumexp(fw_logits[t-1].unsqueeze(-1) + self.obs_logits(X,t).unsqueeze(-2) + trans_logits,-2)

        logZ = stable_logsumexp(fw_logits[-1],-1,True)
        fw_logits = fw_logits - logZ
        logZ = logZ.squeeze(-1)

        SEzz = torch.zeros(fw_logits.shape[1:]+self.event_shape,requires_grad=False)
        for t in range(T-2,-1,-1):
            ### Backward Smoothing
            xi_logits = fw_logits[t].unsqueeze(-1) + trans_logits 
            xi_logits = (xi_logits - stable_logsumexp(xi_logits,-2,keepdim=True)) + fw_logits[t+1].unsqueeze(-2)
            fw_logits[t] = stable_logsumexp(xi_logits,-1)
            SEzz = SEzz + (xi_logits - stable_logsumexp(xi_logits,(-1,-2), keepdim=True)).exp()
                        
        # Now do the initial step
        xi_logits = self.initial.loggeomean().unsqueeze(-1) + trans_logits
        xi_logits = (xi_logits - self.stable_logsumexp(xi_logits,-2,keepdim=True)) + fw_logits[0].unsqueeze(-2)
        SEz0 = stable_logsumexp(xi_logits,-1)
        SEz0 = (SEz0-stable_logsumexp(SEz0,-1,True)).exp()
        SEzz = SEzz + (xi_logits - stable_logsumexp(xi_logits,(-1,-2), keepdim=True)).exp()  

        fw_logits =  ((fw_logits - fw_logits.max(-1,keepdim=True)[0])/self.ptemp).exp()
        fw_logits = fw_logits/fw_logits.sum(-1,keepdim=True)
        if fw_logits.isnan().any():
            print('HMM:  NaN in p')
        return fw_logits, SEzz, SEz0, logZ  # Note that only Time has been integrated out of sufficient statistics and fw_logits is now p(z_t|x_{0:T-1})
    
    def forward_backward_logits(self,fw_logits):
        # Assumes that time is in the first dimension of the observation
        # On input fw_logits = observation_logits. 
#        T = observation_logits.shape[0]
        trans_logits = self.transition.loggeomean()
        T = fw_logits.shape[0]
        fw_logits[0] = stable_logsumexp(self.initial.loggeomean().unsqueeze(-1) + trans_logits + fw_logits[0].unsqueeze(-2),-2)
        for t in range(1,T):
            fw_logits[t] = stable_logsumexp(fw_logits[t-1].unsqueeze(-1) + trans_logits + fw_logits[t].unsqueeze(-2),-2)
        logZ = stable_logsumexp(fw_logits[-1],-1,True)
        fw_logits = fw_logits - logZ
        logZ = logZ.squeeze(-1)
        SEzz = torch.zeros(fw_logits.shape[1:]+self.event_shape,requires_grad=False)
        for t in range(T-2,-1,-1):
            ### Backward Smoothing
            temp = fw_logits[t].unsqueeze(-1) + trans_logits
            xi_logits = (temp - stable_logsumexp(temp,-2,keepdim=True)) + fw_logits[t+1].unsqueeze(-2)
            fw_logits[t] = stable_logsumexp(xi_logits,-1)
            SEzz = SEzz + (xi_logits - stable_logsumexp(xi_logits,(-1,-2), keepdim=True)).exp()
                        
        # Now do the initial step
        # Backward Smoothing
        temp = self.initial.loggeomean().unsqueeze(-1) + trans_logits
        xi_logits = (temp - stable_logsumexp(temp,-2,keepdim=True)) + fw_logits[0].unsqueeze(-2)
        SEz0 = stable_logsumexp(xi_logits,-1)
        SEz0 = (SEz0-stable_logsumexp(SEz0,-1,True)).exp()
        SEzz = SEzz + (xi_logits - stable_logsumexp(xi_logits,(-1,-2), keepdim=True)).exp()     

        fw_logits =  ((fw_logits - fw_logits.max(-1,keepdim=True)[0])/self.ptemp).exp()
        fw_logits = fw_logits/fw_logits.sum(-1,keepdim=True)
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
            return self.obs_dist.Elog_like(X[t].unsqueeze(-1-self.obs_dist.event_dim))
        else:
            return self.obs_dist.Elog_like(X.unsqueeze(-1-self.obs_dist.event_dim))

    def update_states(self,X,T=None):
        # updates states and stores in self.p
        # also updates sufficient statistics of Markov process (self.SEzz, self.SEz0) and self.logZ and self.sumlogZ
        if T is None:
            self.p, SEzz, SEz0, logZ = self.forward_backward_logits(self.obs_logits(X))  # recall that time has been integrated out except for p.
        else:
            self.p, SEzz, SEz0, logZ = self.forward_backward_steps(X,T)  # recall that time has been integrated out except for p.
        NA = self.p.sum(0) # also integrate out time for NA
        sample_dims = list(range(NA.ndim - self.batch_dim - self.event_dim))
        NA = NA.sum(sample_dims)
        SEzz = SEzz.sum(sample_dims)
        SEz0 = SEz0.sum(sample_dims)
        logZ = logZ.sum(sample_dims)
        return SEzz, SEz0, NA, logZ
        
    def update_markov_parms(self,SEzz, SEz0, lr=1.0,beta=None):
        self.transition.ss_update(SEzz,lr=lr,beta=beta)
        self.initial.ss_update(SEz0,lr=lr,beta=beta)

    def update_obs_parms(self,X,lr=1.0,beta=None):
        self.obs_dist.raw_update(X.unsqueeze(-1-self.obs_dist.event_dim),p=self.p,lr=lr,beta=beta)

    def update(self,X,iters=1,T=None,lr=1.0,beta=None,verbose=False):   

        for i in range(iters):
            SEzz, SEz0, self.NA, self.logZ = self.update_states(X,T)
            self.KLqprior_last = self.KLqprior()
            self.update_markov_parms(SEzz, SEz0, lr=lr,beta=beta)
            self.update_obs_parms(X,lr=lr,beta=beta)
            
            ELBO = self.ELBO()
            if verbose:
                print('Percent Change in ELBO = ', ((ELBO-self.ELBO_last)/torch.abs(self.ELBO_last)*100))
            self.ELBO_last = ELBO

    def KLqprior(self):
        KL = self.obs_dist.KLqprior().sum(-1) + self.transition.KLqprior().sum(-1) + self.initial.KLqprior()  # assumes default event_dim = 1
        return KL

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
