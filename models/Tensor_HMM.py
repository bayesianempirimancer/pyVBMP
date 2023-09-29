import torch
import dists.Dirichlet as Dirichlet
import transforms.Transition as Transition
from utils.torch_functions import stable_logsumexp, stable_softmax

class Tensor_HMM():
    def __init__(self, obs_dist, event_shape, ptemp =1.0, prior_parms = None):

        assert len(obs_dist.batch_shape) >= len(event_shape), 'HMM:  len(obs_dist.event_shape) < len(event_shape)'
        for i in range(len(event_shape)): 
            assert obs_dist.batch_shape[-(i+1)] in (event_shape[-(i+1)],1), 'HMM:  obs_dist.event_shape[-(i+1)] != event_shape[-(i+1)] or 1'

        self.obs_dist = obs_dist

        self.dim = 1
        for i in event_shape: self.dim = self.dim*i

        self.event_dim = len(event_shape)
        self.event_shape = event_shape
        self.batch_shape = obs_dist.batch_shape[:-len(event_shape)]        
        self.batch_dim = len(self.batch_shape)
        if prior_parms is None:
            alpha = torch.eye(self.dim,requires_grad=False).reshape(event_shape + event_shape) + 0.5
            prior_parms = {'alpha': alpha}

        self.transition = Transition(event_shape, self.batch_shape, prior_parms=prior_parms)
        self.initial = Dirichlet(self.event_shape, self.batch_shape)

        self.sumlogZ = -torch.inf
        self.p = None
        self.ptemp = ptemp
        self.logZ = torch.tensor(-torch.inf)
        self.ELBO_last = torch.tensor(-torch.inf)
    
    def forward_step(self,logits,observation_logits):
        return self.transition.forward_filter(logits,observation_logits)
    
    def backward_step(self,logits_t,logits_tp1):
        return self.transition.backward_smoothe(logits_t,logits_tp1)

    def forward_backward_steps(self,X,T): 
        temp = self.forward_step(self.initial.loggeomean(),self.obs_logits(X,0))
        fw_logits = torch.zeros((T,)+temp.shape,requires_grad=False)
        fw_logits[0] = temp
        for t in range(1,T):
            fw_logits[t] = self.transition.forward_filter(fw_logits[t-1],self.obs_logits(X,t))

        logZ = stable_logsumexp(fw_logits[-1],self.transition.right_sum_list)
        SEzz = 0.0
        for t in range(T-2,-1,-1):
            fw_logits[t], xi_logits = self.transition.backward_smoothe(fw_logits[t],fw_logits[t+1])
            SEzz = SEzz + stable_softmax(xi_logits,list(range(-2*self.event_dim,0))).exp()
                        
        # Now do the initial step
        SEz0, xi_logits = self.transition.backward_smoothe(self.initial.loggeomean(),fw_logits[0])
        SEzz = SEzz + stable_softmax(xi_logits,list(range(-2*self.event_dim,0))).exp()
        SEz0 = stable_softmax(SEz0,list(range(-self.event_dim,0))).exp()

        fw_logits = (stable_softmax(fw_logits,list(range(-self.event_dim,0)))/self.ptemp).exp()
        fw_logits = fw_logits/fw_logits.sum(list(range(-self.event_dim,0)),keepdim=True)
        if fw_logits.isnan().any():
            print('HMM:  NaN in p')
        return fw_logits, SEzz, SEz0, logZ  # Note that only Time has been integrated out of sufficient statistics and fw_logits is now p(z_t|x_{0:T-1})
    
    def assignment_pr(self):
        return self.p
    
    def assignment(self):
        return self.p.argmax(-1)

    def obs_logits(self,X,t=None):
        Xv = X.view(X.shape[:-self.obs_dist.event_dim] + self.event_dim*(1,) +X.shape[-self.obs_dist.event_dim:])   
        if t is not None:
            return self.obs_dist.Elog_like(Xv[t])   
        else:
            return self.obs_dist.Elog_like(Xv)
        
    def update_states(self,X,T=None):
        # updates states and stores in self.p
        # also updates sufficient statistics of Markov process (self.SEzz, self.SEz0) and self.logZ and self.sumlogZ
        if T is None:  T=X.shape[0]
        self.p, SEzz, SEz0, logZ = self.forward_backward_steps(X,T)  # recall that time has been integrated out except for p.
        NA = self.p.sum(0) # also integrate out time for NA
        sample_dims = list(range(NA.ndim - self.batch_dim - self.event_dim))
        NA = NA.sum(sample_dims)
        SEzz = SEzz.sum(sample_dims)
        SEz0 = SEz0.sum(sample_dims)
        logZ = logZ.sum(sample_dims)
        return SEzz, SEz0, NA, logZ
        
    def update_markov_parms(self, SEzz, SEz0, lr=1.0,beta=None):
        self.transition.ss_update(SEzz,lr=lr,beta=beta)
        self.initial.ss_update(SEz0,lr=lr,beta=beta)

    def update_obs_parms(self,X,lr=1.0,beta=None):
        Xv = X.view(X.shape[:-self.obs_dist.event_dim] + self.event_dim*(1,) +X.shape[-self.obs_dist.event_dim:])   
        self.obs_dist.raw_update(Xv,p=self.p,lr=lr,beta=beta)

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
        return self.obs_dist.KLqprior().sum(list(range(-self.event_dim,0))) + self.transition.KLqprior() + self.initial.KLqprior()  

    def ELBO(self):
        return self.logZ - self.KLqprior() 

    def event_average_f(self,function_string,keepdim=False):
        return self.event_average(eval('self.obs_dist.'+function_string)(),keepdim)

    def average_f(self,function_string,keepdim=False):
        return self.average(eval('self.obs_dist.'+function_string)(),keepdim)

    def average(self,A,keepdim=False):  # returns sample_shape 
        # A is mix_batch_shape + mix_event_shape 
        return (A*self.p).sum(list(range(-self.event_dim,0)),keepdim)

    ### Compute special expectations used for VB inference
    def event_average(self,A,keepdim=False):  # returns sample_shape + W.event_shape
        # A is mix_batch_shape + mix_event_shape + event_shape
        print('Tensor_HMM:  event_average has not been verified')
        out = (A*self.p.view(self.p.shape + (1,)*self.obs_dist.event_dim)).sum(-self.obs_dist.event_dim-1,keepdim)
        for i in range(self.event_dim-1):
            out = out.sum(-self.obs_dist.event_dim-1,keepdim)
        return out
