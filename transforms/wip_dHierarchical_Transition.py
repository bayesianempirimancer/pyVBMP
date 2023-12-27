import torch
import dists.Dirichlet as Dirichlet
import transforms.Hierarchical_Transition as Hierarchical_Transition

class Hierarchical_Transition_Object(Hierarchical_Transition):
    def __init__(self, A_dim, S_dim, Nz, N_hs, Ns, Nb, batch_shape=(), prior_parms = None):

        self.batch_shape = batch_shape + (A_dim,S_dim, Nz, N_hs, Ns, Nb)
        self.event_shape = (Nz,N_hs,Ns,Nb)
        self.event_dim = len(self.event_shape)
        self.batch_dim = len(self.batch_shape)

        n_dims=4 # (Nz,N_hs,Ns,Nb)
        self.dists = []
        self.sum_list = []

        alpha_0 = torch.tensor(0.5)
        alpha_sticky = torch.tensor(1.0)

        # Object Identity Transitions
        shape1 = (A_dim,S_dim,Nz,1,1,1)
        shape2 = (Nz,1,1,1)
        alpha = alpha_0.expand(shape1 + shape2) + alpha_sticky*torch.eye(Nz,requires_grad=False).view(2*shape2)
        self.dists.append(Dirichlet(event_shape = shape2 , batch_shape = self.batch_shape + shape1, prior_parms = {'alpha':alpha}))
        sum_list1 = list(range(-2*n_dims+i+1,-n_dims))
        sum_list2 = list(range(-3,0))
        self.sum_list = sum_list1 + sum_list2

        # Object 'Inertia/Internal State' Transitions
        shape1 = (A_dim,S_dim,Nz,N_hs,1,1)
        shape2 = (1,N_hs,1,1)
        alpha = alpha_0.expand(shape1 + shape2) + alpha_sticky*torch.eye(N_hs,requires_grad=False).view(2*shape2)
        self.dists.append(Dirichlet(event_shape = shape2 , batch_shape = self.batch_shape + shape1, prior_parms = {'alpha':alpha}))
        sum_list2 = list(range(-2,0))
        self.sum_list = sum_list1 + sum_list2

        # Object 'State + Location' Transitions
        shape1 = (1,1,Nz,N_hs,Ns,Nb)
        shape2 = (1,1,Ns,Nb)
        alpha = alpha_0.expand(shape1 + shape2) + alpha_sticky*torch.eye(Ns*Nb,requires_grad=False).reshape(Ns,Nb,Ns,Nb).view(2*shape2)
        self.dists.append(Dirichlet(event_shape = shape2 , batch_shape = self.batch_shape + shape1, prior_parms = {'alpha':alpha}))
        sum_list2 = list(range(-4,-2))
        self.sum_list = sum_list1 + sum_list2
        self.NA = 0.0

