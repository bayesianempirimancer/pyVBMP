import torch
from dists.Dirichlet import Dirichlet
from utils.torch_functions import stable_logsumexp, stable_softmax

class Transition(Dirichlet):
    def __init__(self, event_shape, batch_shape = (), 
                prior_parms={'alpha':torch.tensor(0.5)}):
        
        super().__init__(event_shape=event_shape,batch_shape = batch_shape+event_shape, prior_parms = prior_parms)
        self.left_sum_list = list(range(-2*self.event_dim,-self.event_dim))
        self.right_sum_list = list(range(-self.event_dim,0))

    def unsqueeze_left(self,X):
        return X.view(X.shape[:-self.event_dim] + self.event_dim*(1,) + X.shape[-self.event_dim:])
        
    def unsqueeze_right(self,X):
        return X.view(X.shape + self.event_dim*(1,))

    def forward_filter(self,logits,obs_logits):
        return stable_logsumexp(self.unsqueeze_right(logits) + self.unsqueeze_left(obs_logits) + self.loggeomean(),self.left_sum_list)

    def backward_smoothe(self,logits_t,logits_tplus1):
        xi_logits = stable_softmax(self.unsqueeze_right(logits_t) + self.loggeomean(),self.left_sum_list)
        xi_logits = xi_logits + self.unsqueeze_left(logits_tplus1)
        return stable_logsumexp(xi_logits,self.right_sum_list), xi_logits

    def log_forward(self,logits):
        return stable_logsumexp(self.unsqueeze_right(logits)+self.loggeomean(),self.left_sum_list)

    def log_backward(self,logits):
        return stable_logsumexp(self.unsqueeze_left(logits)+self.loggeomean(),self.right_sum_list)

    def KLqprior(self):
        return super().KLqprior().sum(list(range(-self.event_dim,0)))
    
    def Elog_like(self,X,Y):
        return (self.unsqueeze_right(X)*self.unsqueeze_left(Y)*self.loggeomean()).sum(list(range(-2*self.event_dim,0)))
    
