
import torch
from .Tensor_HMM import Tensor_HMM
import dists.NormalInverseWishart as NormalInverseWishart

class Factorial_HMM(Tensor_HMM):
    def __init__(self, num_factors, factor_shape, event_shape=(), batch_shape = ()):
        obs_dist = NormalInverseWishart(event_shape = event_shape, batch_shape = batch_shape + num_factors*factor_shape)
        self.num_factors = num_factors
        self.factor_shape = factor_shape
        alpha = 0.0
        self.marg_sum_list = []
        for i in range(num_factors):
            event_shape = i*len(factor_shape)*(1,) + factor_shape + (num_factors-i-1)*len(factor_shape)*(1,)
            lil_alpha = torch.eye(torch.tensor(event_shape,requires_grad=False).prod()).reshape(event_shape + event_shape)+0.5
            alpha = alpha + lil_alpha
            self.marg_sum_list.append([x for x in range(-2*len(event_shape),0) if (2*event_shape)[x]==1])

        alpha = alpha/alpha.max()*2
        prior_parms = {'alpha': alpha}

        super().__init__(obs_dist, event_shape = num_factors*factor_shape, prior_parms = prior_parms)

    def factorize_transition_probabilities(self):
        alpha = self.transition.alpha
        alpha_marg =[]
        for i in range(self.num_factors):
            alpha_marg.append(alpha.mean(self.marg_sum_list[i],True)/self.num_factors)
        alpha = 0.0
        for i in range(self.num_factors):
            alpha = alpha + alpha_marg[i]
        self.transition.alpha = alpha

    def update_markov_parms(self, SEzz, SEz0, lr=1, beta=None):
        super().update_markov_parms(SEzz, SEz0, lr, beta)
        self.factorize_transition_probabilities()
