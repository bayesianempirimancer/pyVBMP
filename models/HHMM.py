
import torch
import dists.Dirichlet as Dirichlet
import transforms.Hierarchical_Transition as Hierarchical_Transition
import dists.NormalInverseWishart as NormalInverseWishart
from utils.torch_functions import stable_logsumexp, stable_softmax

class HHMM():
    def __init__(self, obs_dist, event_dim=2, event_shape=(), ptemp=1):
        assert event_dim > 1, 'HHMM: event_dim must be > 1 if event_dim==1 use HMM instead'
        self.obs_dist = obs_dist
        if event_shape == ():
            self.event_shape = obs_dist.batch_shape[-event_dim:]
        else:
            assert len(event_shape) == event_dim, 'HHMM: event_shape must have length event_dim'
            obs_event_shape = obs_dist.batch_shape[-event_dim:]
            for k, es in enumerate(event_shape):
                assert(es == obs_event_shape[k] or obs_event_shape[k] == 1), 'HHMM: event_shape must be compatible with obs_dist.event_shape'
            self.event_shape = event_shape
        self.event_dim = event_dim
        self.batch_shape = obs_dist.batch_shape[:-event_dim] 
        self.batch_dim = len(self.batch_shape)

        self.transition = Hierarchical_Transition(self.event_shape, self.batch_shape)
        self.initial = Dirichlet(self.event_shape, self.batch_shape)

        self.left_sum_list = list(range(-2*self.event_dim,-self.event_dim))
        self.right_sum_list = list(range(-self.event_dim,0))
        self.sumlogZ = -torch.inf
        self.p = None
        self.ptemp = ptemp
        self.logZ = torch.tensor(-torch.inf)
        self.ELBO_last = torch.tensor(-torch.inf)

    def unsqueeze_left(self,X):
        return X.view(X.shape[:-self.event_dim] + self.event_dim*(1,) + X.shape[-self.event_dim:])
        
    def unsqueeze_right(self,X):
        return X.view(X.shape + self.event_dim*(1,))

    def forward_step(self,logits,observation_logits):
        return stable_logsumexp(self.unsqueeze_right(logits) + self.unsqueeze_left(observation_logits) + self.transition.loggeomean(),self.left_sum_list)
    
    def backward_step(self,logits,observation_logits):
        return stable_logsumexp(self.unsqueeze_left(logits) + self.unsqueeze_left(observation_logits) + self.transition.loggeomean(),self.right_sum_list)

    def forward_backward_steps(self,X,T): 
        temp = self.forward_step(self.initial.loggeomean(),self.obs_logits(X,0))
        fw_logits = torch.zeros((T,) + temp.shape,requires_grad=False)
        fw_logits[0] = temp
        for t in range(1,T):
            fw_logits[t] = self.forward_step(fw_logits[t-1],self.obs_logits(X,t))

        logZ = stable_logsumexp(fw_logits[-1],self.right_sum_list,True)
#        fw_logits = fw_logits - logZ
        logZ = logZ.view(logZ.shape[:-self.event_dim])

        SEzz = torch.zeros(fw_logits.shape[1:]+self.event_shape,requires_grad=False)
        for t in range(T-2,-1,-1):
            ### Backward Smoothing
            temp = self.unsqueeze_right(fw_logits[t]) + self.transition.loggeomean() 
            xi_logits = (temp - stable_logsumexp(temp,self.left_sum_list,keepdim=True)) + self.unsqueeze_left(fw_logits[t+1])
            fw_logits[t] = stable_logsumexp(xi_logits,self.right_sum_list)
            SEzz = SEzz + (xi_logits - stable_logsumexp(xi_logits,self.left_sum_list + self.right_sum_list, keepdim=True)).exp()
                        
        # Now do the initial step
        temp = self.unsqueeze_right(self.initial.loggeomean()) + self.transition.loggeomean() 
        xi_logits = (temp - stable_logsumexp(temp,self.left_sum_list,keepdim=True)) + self.unsqueeze_left(fw_logits[0])
        SEz0 = stable_logsumexp(xi_logits,self.right_sum_list)
        SEz0 = (SEz0-stable_logsumexp(SEz0,self.right_sum_list,True)).exp()
        SEzz = SEzz + (xi_logits - stable_logsumexp(xi_logits,self.left_sum_list + self.right_sum_list, keepdim=True)).exp()

        fw_logits =  ((fw_logits - stable_logsumexp(fw_logits,self.right_sum_list,keepdim=True))/self.ptemp).exp()
        fw_logits = fw_logits/fw_logits.sum(self.right_sum_list,keepdim=True)
        if fw_logits.isnan().any():
            print('HMM:  NaN in p')
        return fw_logits, SEzz, SEz0, logZ  # Note that only Time has been integrated out of sufficient statistics and fw_logits is now p(z_t|x_{0:T-1})
    
    def forward_backward_logits(self,fw_logits):
        # Assumes that time is in the first dimension of the observation
        # On input fw_logits = observation_logits. 
#        T = observation_logits.shape[0]
        T = fw_logits.shape[0]
        fw_logits = fw_logits.expand(fw_logits.shape[:-self.event_dim] + self.event_shape).clone()
        fw_logits[0] = stable_logsumexp(self.unsqueeze_left(self.initial.loggeomean()) + self.transition.loggeomean() + self.unsqueeze_left(fw_logits[0]),self.left_sum_list)
        for t in range(1,T):
            fw_logits[t] = stable_logsumexp(self.unsqueeze_right(fw_logits[t-1]) + self.transition.loggeomean() + self.unsqueeze_left(fw_logits[t]),self.left_sum_list)
        logZ = stable_logsumexp(fw_logits[-1],self.right_sum_list,True)
#        fw_logits = fw_logits - logZ
        logZ = logZ.view(logZ.shape[:-self.event_dim])
        SEzz = torch.zeros(fw_logits.shape[1:]+self.event_shape,requires_grad=False)
        for t in range(T-2,-1,-1):
            ### Backward Smoothing
            temp = self.unsqueeze_right(fw_logits[t]) + self.transition.loggeomean() 
            xi_logits = (temp - stable_logsumexp(temp,self.left_sum_list,keepdim=True)) + self.unsqueeze_left(fw_logits[t+1])
            fw_logits[t] = stable_logsumexp(xi_logits,self.right_sum_list)
            SEzz = SEzz + (xi_logits - stable_logsumexp(xi_logits,self.left_sum_list + self.right_sum_list, keepdim=True)).exp()
                        
        # Now do the initial step
        # Backward Smoothing
        temp = self.unsqueeze_right(self.initial.loggeomean()) + self.transition.loggeomean() 
        xi_logits = (temp - stable_logsumexp(temp,self.left_sum_list,keepdim=True)) + self.unsqueeze_left(fw_logits[0])
        SEz0 = stable_logsumexp(xi_logits,self.right_sum_list)
        SEz0 = (SEz0-stable_logsumexp(SEz0,self.right_sum_list,True)).exp()
        SEzz = SEzz + (xi_logits - stable_logsumexp(xi_logits,self.left_sum_list + self.right_sum_list, keepdim=True)).exp()
   
        fw_logits =  ((fw_logits - stable_logsumexp(fw_logits,self.right_sum_list,keepdim=True))/self.ptemp).exp()
        fw_logits = fw_logits/fw_logits.sum(self.right_sum_list,keepdim=True)
        if fw_logits.isnan().any():
            print('HHMM:  NaN in p')

        return fw_logits, SEzz, SEz0, logZ  # Note that only Time has been integrated out of sufficient statistics
                                            # and the despite the name fw_logits is posterior probability of states, i.e. self.p
    def assignment_pr(self):
        return self.p
    
    def assignment(self):
        raise NotImplementedError

    def obs_logits(self,X,t=None):
        sample_shape = X.shape[:-self.batch_dim-self.obs_dist.event_dim]  ### CHECK THIS
        Xv = X.view(sample_shape + self.batch_dim*(1,) + self.event_dim*(1,)+ self.obs_dist.event_shape )
        if t is not None:
            ELL =  self.obs_dist.Elog_like(Xv[t])  # has shape sample x event[-1]
        else:
            ELL = self.obs_dist.Elog_like(Xv)
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
        sample_shape = X.shape[:-self.obs_dist.event_dim]  ###CHECK THIS
        Xv = X.view(sample_shape + self.event_dim*(1,)+ self.obs_dist.event_shape )
        self.obs_dist.raw_update(Xv,p=self.p,lr=lr,beta=beta)

    def update(self,X,iters=1,T=None,lr=1.0,beta=None,verbose=False):   
        if T is None:
            T=X.shape[0]
        for i in range(iters):
            SEzz, SEz0, self.NA, self.logZ = self.update_states(X,T)
            ELBO = self.ELBO()
            if verbose:
                print('Percent Change in ELBO = ', ((ELBO-self.ELBO_last)/torch.abs(self.ELBO_last)*100))
            self.ELBO_last = ELBO
            self.update_markov_parms(SEzz, SEz0, lr=lr,beta=beta)
            self.update_obs_parms(X,lr=lr,beta=beta)

    def KLqprior(self):
        KL = self.obs_dist.KLqprior()
        for i in range(self.event_dim):
            KL = KL.sum(-1)
        return KL + self.transition.KLqprior() + self.initial.KLqprior() 

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
