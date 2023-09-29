import torch
import transforms.wip_GenerativeBayesianTransformer as GBT
import transforms.wip_FocusedBayesianTransformer as FBT
#import transforms.wip_ChainedBayesianTransformer as CBT
import matplotlib.pyplot as plt


mixture_dim = 8
role_dim = 4
obs_dim = 2
hidden_dim = 2


num_samples = 1000
num_obs = 10
Y = 4*torch.randn(num_samples, num_obs, obs_dim)*torch.rand(num_samples,num_obs,1)
pY = torch.randint(role_dim,(num_samples,obs_dim))
pY = torch.eye(role_dim,requires_grad=False)[pY]


X = torch.randn(num_samples, mixture_dim, hidden_dim)
z = torch.rand(num_samples, mixture_dim).argmax(-1)
A = torch.randn(mixture_dim, obs_dim, hidden_dim)/hidden_dim**(0.5)
Y = (A[z].unsqueeze(1)@X.unsqueeze(-1)).squeeze(-1)

m5 = FBT.FocusedBayesianTransformer(mixture_dim, role_dim, obs_dim, hidden_dim, batch_shape = (), pad_X=True)
m6 = GBT.GenerativeBayesianTransformer(mixture_dim, role_dim, obs_dim, hidden_dim, batch_shape = (), pad_X=True)

Y = torch.randn(num_samples, num_obs, obs_dim)

m5.raw_update(Y,iters=20,lr=1,verbose=True)
m6.raw_update(Y,iters=20,lr=1,verbose=True)


#m0.update(model.pX,Delta(Y.unsqueeze(-1).unsqueeze(-3)),iters=5,lr=1,verbose=True)
#m0.Elog_like_given_pX_pY(model.pX,Delta(Y.unsqueeze(-1).unsqueeze(-3))).sum(0)

