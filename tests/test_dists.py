# Tests Basic Functionality of models
#
import torch
# torch.set_default_device('cuda')  # This breaks all plotting routines (matplotlib doesn't work with device = cuda)
import numpy as np
import time
from matplotlib import pyplot as plt
import dists, transforms

# print ('TEST Multilinear Normal Wishart')
# num_samples = 1000
# n=12
# p_list = (3,4,5)
# X_list = []
# W_list = []
# Y = torch.randn(n,1)
# for i in range(len(p_list)):
#     X_list.append(torch.randn(num_samples, p_list[i], 1))
#     W_list.append(torch.randn(n, p_list[i]))
#     Y = Y + W_list[i]@X_list[i]

# Y = Y + torch.randn_like(Y)*0.1
# model = transforms.MultiLinearNormalWishart(n,p_list)
# model.raw_update(X_list,Y,iters = 3, lr=1)

# Yhat = model.predict(X_list)[0].mean()
# plt.scatter(Y,Yhat)
# plt.show()
# for i in range(len(p_list)):
#     plt.scatter(W_list[i],model.A[i].mean())
# plt.show()

# pX_list= model.postdict(Y,iters=10)
# for i in range(3):
#     plt.scatter(X_list[i],pX_list[i].mean())
# plt.plot([-3,3],[-3,3])
# plt.show()

# pX_list = [None]*len(X_list)
# for i in range(len(X_list)):
#     pX_list[i] = dists.MultivariateNormal_vector_format(invSigma = 100*torch.eye(X_list[i].shape[-2]),invSigmamu = 100*X_list[i])

# pY,pY_list = model.forward(pX_list)
# plt.scatter(Y,pY.mean())

print('TEST VANILLA Matrix Normal Wishart with X_mask, and mask')
import torch
import numpy as np
import time
from matplotlib import pyplot as plt

n=50
p=50
n_samples = 1000
batch_num = 1
w_true = torch.randn(n,p)/np.sqrt(p)
X_mask = w_true.abs().sum(-2)<w_true.abs().sum(-2).mean()
X_mask = X_mask.unsqueeze(-2)
w_true = w_true*X_mask

b_true = torch.randn(n,1)
pad_X = True

X=torch.randn(n_samples,p)
Y=torch.zeros(n_samples,n)
for i in range(n_samples):
    Y[i,:] = X[i:i+1,:]@w_true.transpose(-1,-2) + b_true.transpose(-2,-1)*pad_X + torch.randn(1)/100.0
from matplotlib import pyplot as plt

W0 = transforms.MatrixNormalWishart(event_shape = (n,p), pad_X=pad_X)
t = time.time()
W0.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
print(time.time()-t)


W1 = transforms.MatrixNormalWishart(event_shape = (n,p), X_mask=X_mask,pad_X=pad_X)
W2 = transforms.MatrixNormalWishart(event_shape = (n,p), mask=X_mask.expand(n,p),pad_X=pad_X)


W1.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
W2.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))


pY, Res = W0.predict(X.unsqueeze(-1))
Yhat = pY.mean().squeeze(-1)
pX = dists.MultivariateNormal_vector_format(invSigma = 10*torch.eye(p),invSigmamu = 10*X.unsqueeze(-1))
pYf, Resf = W0.forward(pX)
Yhat = pYf.mean().squeeze(-1)
plt.scatter(Y.numpy(),Yhat.numpy())
plt.title('W0 Predition')
plt.show()
plt.scatter(w_true.numpy(),W0.weights().numpy())
plt.title('Weights')
plt.show()

Yhat = W1.predict(X.unsqueeze(-1))[0].mean().squeeze(-1)
plt.scatter(Y.numpy(),Yhat.numpy())
plt.title('W1 Predition')
plt.show()
plt.scatter(w_true.numpy(),W1.weights().numpy())
plt.title('Weights')
plt.show()

Yhat = W2.predict(X.unsqueeze(-1))[0].mean().squeeze(-1)
plt.scatter(Y.numpy(),Yhat.numpy())
plt.title('W2 Prediction')
plt.show()
plt.scatter(w_true.numpy(),W2.weights().numpy())
plt.title('Weights')
plt.show()

invSigma_xx, invSigmamu_x, Res = W0.Elog_like_X(Y.unsqueeze(-1))
mu_x0 = torch.linalg.solve(invSigma_xx+1e-6*torch.eye(p),invSigmamu_x)
plt.scatter(X.numpy(),mu_x0.squeeze().numpy(),alpha=0.2)
plt.title('W0 backward Prediction')
plt.show()
invSigma_xx, invSigmamu_x, Res = W1.Elog_like_X(Y.unsqueeze(-1))
mu_x1 = invSigma_xx.pinverse()@invSigmamu_x
plt.scatter(mu_x0.squeeze().numpy(),mu_x1.squeeze().numpy(),alpha=0.2)
plt.title('W1 backward Prediction')
plt.show()
invSigma_xx, invSigmamu_x, Res = W2.Elog_like_X(Y.unsqueeze(-1))
mu_x2 = invSigma_xx.pinverse()@invSigmamu_x
plt.scatter(mu_x0.squeeze().numpy(),mu_x2.squeeze().numpy(),alpha=0.2)
plt.title('W2 backward Prediction')
plt.show()

print('TEST VANILLA Matrix Normal Gamma with X_mask, and mask')
import transforms.MatrixNormalGamma
n=10
p=20
n_samples = 4000
batch_num = 4
w_true = torch.randn(n,p)/np.sqrt(p)
X_mask = w_true.abs().sum(-2)<w_true.abs().sum(-2).mean()
X_mask = X_mask.unsqueeze(-2)
w_true = w_true*X_mask

b_true = torch.randn(n,1)
pad_X = True

W0 = transforms.MatrixNormalGamma(event_shape = (n,p), pad_X=pad_X)
W1 = transforms.MatrixNormalGamma(event_shape = (n,p), X_mask=X_mask,pad_X=pad_X)
W2 = transforms.MatrixNormalGamma(event_shape = (n,p), mask=X_mask.expand(n,p),pad_X=pad_X)
X=torch.randn(n_samples,p)
Y=torch.zeros(n_samples,n)
for i in range(n_samples):
    Y[i,:] = X[i:i+1,:]@w_true.transpose(-1,-2) + b_true.transpose(-2,-1)*pad_X + torch.randn(1)/100.0
from matplotlib import pyplot as plt
W0.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
W1.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
W2.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))

Yhat = W0.predict(X.unsqueeze(-1))[0].mean().squeeze(-1)
plt.scatter(Y.numpy(),Yhat.numpy())
plt.title('W0 Predition')
plt.show()
plt.scatter(w_true.numpy(),W0.weights().numpy())
plt.title('Weights')
plt.show()

Yhat = W1.predict(X.unsqueeze(-1))[0].mean().squeeze(-1)
plt.scatter(Y.numpy(),Yhat.numpy())
plt.title('W1 Predition')
plt.show()
plt.scatter(w_true.numpy(),W1.weights().numpy())
plt.title('Weights')
plt.show()

Yhat = W2.predict(X.unsqueeze(-1))[0].mean().squeeze(-1)
plt.scatter(Y.numpy(),Yhat.numpy())
plt.title('W2 Prediction')
plt.show()
plt.scatter(w_true.numpy(),W2.weights().numpy())
plt.title('Weights')
plt.show()

invSigma_xx, invSigmamu_x, Res = W0.Elog_like_X(Y.unsqueeze(-1))
mu_x0 = torch.linalg.solve(invSigma_xx+1e-6*torch.eye(p),invSigmamu_x)
plt.scatter(X.numpy(),mu_x0.squeeze().numpy(),alpha=0.2)
plt.title('W0 backward Prediction')
plt.show()
invSigma_xx, invSigmamu_x, Res = W1.Elog_like_X(Y.unsqueeze(-1))
mu_x1 = invSigma_xx.pinverse()@invSigmamu_x
plt.scatter(X.numpy(),mu_x1.squeeze().numpy(),alpha=0.2)
plt.title('W1 backward Prediction')
plt.show()
invSigma_xx, invSigmamu_x, Res = W2.Elog_like_X(Y.unsqueeze(-1))
mu_x2 = invSigma_xx.pinverse()@invSigmamu_x
plt.scatter(X.numpy(),mu_x2.squeeze().numpy(),alpha=0.2)
plt.title('W2 backward Prediction')
plt.show()


print('TEST Gaussian Mixture Model')
import torch
from matplotlib import pyplot as plt
import numpy as np
import time
from models.GaussianMixtureModel import GaussianMixtureModel as GMM

dim = 2
nc = 6
mu = torch.randn(nc,dim)*4  
A = torch.randn(nc,dim,dim)/np.sqrt(2)

num_samples = 400
data = torch.zeros(num_samples,dim)

for i in range(num_samples):
    data[i,:] = mu[i%nc,:] + A[i%nc,:,:]@torch.randn(dim) + torch.randn(dim)/8.0

data = data/data.std()

#data = data-data.mean(0,True)
#data = data/data.std(0,True)
t = time.time()
gmm = GMM(nc,dim)
gmm.initialize(data)
gmm.update(data,20,1,verbose=False)
print(time.time()-t)
plt.scatter(data[:,0].numpy(),data[:,1].numpy(),c=gmm.assignment().numpy())
plt.show()


print('GMM TEST COMPLETE')

print('TEST Isotropic Gaussian Mixture Model')
import torch
import numpy as np
from matplotlib import pyplot as plt
from models.GaussianMixtureModel import GaussianMixtureModel as GMM
dim = 2
nc = 4
nb = 10
mu = torch.randn(4,2)*4  
A = torch.randn(4,2,2)/np.sqrt(2)

num_samples = 200
data = torch.zeros(num_samples,2)

for i in range(num_samples):
    data[i,:] = mu[i%4,:] + A[i%4,:,:]@torch.randn(2) + torch.randn(2)/8.0


#data = data-data.mean(0,True)
#data = data/data.std(0,True)
nc = 6

gmm = GMM(nc,dim,isotropic=True)
gmm.update(data,20,1,verbose=True)
plt.scatter(data[:,0].numpy(),data[:,1].numpy(),c=gmm.assignment().numpy())
plt.show()


print('TEST MIXTURE with non-trivial batch shape')
import torch
import numpy as np
from matplotlib import pyplot as plt
import dists
batch_dim = 3
dim=2
nc=6
mu = torch.randn(nc,batch_dim,dim)*4  
A = torch.randn(nc,batch_dim,dim,dim)/np.sqrt(dim)
data = torch.zeros(200,batch_dim,2)
for i in range(200):
    data[i,:,:] = mu[i%nc,:,:] + (A[i%nc,:,:,:]@torch.randn(batch_dim,dim,1)).squeeze(-1) + torch.randn(batch_dim,dim)/8.0

dist = dists.NormalInverseWishart(event_shape = (dim,), batch_shape = (batch_dim,nc))
model = dists.Mixture(dist, event_shape=(nc,))

idx = model.ELBO().argmax()
model.update(data,iters=10,lr=1,verbose=True)
plt.scatter(data[:,0,0].numpy(),data[:,0,1].numpy(),c=model.assignment()[:,idx].numpy())
plt.show()

print('TEST MIXTURE WITH NON-TRIVIAL EVENT SHAPE')
ndim = 3
dist = dists.NormalInverseWishart(event_shape = (ndim,dim),batch_shape=(nc,))
model = dists.Mixture(dist, event_shape = (nc,))
model.update(data,iters=10,lr=1,verbose=True)

print('TEST MIXTURE WITH NON-TRIVIAL EVENT FOR BOTH DIST AND MIX')
ndim = 3
dist = dists.NormalInverseWishart(event_shape = (ndim,dim), batch_shape = (batch_dim,nc))
model = dists.Mixture(dist, event_shape = (batch_dim,nc))
model.update(data,iters=10,lr=1,verbose=True)

print('Test Tensor Normal Wishart')
import torch
import numpy as np
import dists

batch_shape = (2,)
model = dists.TensorNormalWishart((4,3,2),batch_shape=batch_shape)
X = torch.randn((400,)+batch_shape + (4,3,2))
A = torch.randn(batch_shape+(4,4))
B = torch.randn(batch_shape + (3,3))
C = torch.randn(batch_shape + (2,2))

ABC = A.view(batch_shape + (4,1,1,4,1,1))*B.view(batch_shape + (1,3,1,1,3,1))*C.view(batch_shape + (1,1,2,1,1,2))
AAT = A@A.transpose(-2,-1)
BBT = B@B.transpose(-2,-1)
CCT = C@C.transpose(-2,-1)
ABCABCT = AAT.view(batch_shape + (4,1,1,4,1,1))*BBT.view(batch_shape + (1,3,1,1,3,1))*CCT.view(batch_shape +(1,1,2,1,1,2))

X = X - X.mean(0,keepdim=True)
X = (X.view((400,)+batch_shape+(1,1,1,4,3,2))*ABC).sum((-3,-2,-1))

alpha = AAT.det()**(1/4)*BBT.det()**(1/3)*CCT.det()**(1/2)
AAT = AAT/AAT.det().unsqueeze(-1).unsqueeze(-1)**(1/4)
BBT = BBT/BBT.det().unsqueeze(-1).unsqueeze(-1)**(1/3)
CCT = CCT/CCT.det().unsqueeze(-1).unsqueeze(-1)**(1/2)

model.raw_update(X,lr=1)
from matplotlib import pyplot as plt

plt.scatter(AAT.numpy(),model.invU[0].ESigma().squeeze().numpy())
plt.scatter(BBT.numpy(),model.invU[1].ESigma().squeeze().numpy())
plt.scatter(CCT.numpy(),model.invU[2].ESigma().squeeze().numpy())
m1 = torch.tensor([AAT.min(),BBT.min(),CCT.min()]).min()
m2 = torch.tensor([AAT.max(),BBT.max(),CCT.max()]).max()
plt.plot([m1,m2],[m1,m2])
plt.show()

plt.scatter(ABCABCT.reshape(ABCABCT.numel()).numpy(),model.ESigma().reshape(model.ESigma().numel()).numpy())
m1 = ABCABCT.min()
m2 = ABCABCT.max()
plt.plot([m1,m2],[m1,m2])
plt.show()

print('Test Reduced Rank Regression')
num_samps = 2000
n=10
p=20
dim=2

from matplotlib import pyplot as plt
import torch
import numpy as np
import transforms

model = transforms.ReducedRankRegression(n,p,2*dim,batch_shape = (6,),pad_X=True,independent=False)
U=torch.randn(num_samps,dim)
B=torch.randn(p,dim)/np.sqrt(dim)
A=torch.randn(n,dim)/np.sqrt(dim)

W = A@B.pinverse()

X=U@B.transpose(-2,-1) + torch.randn(num_samps,p)/p
Y=U@A.transpose(-2,-1) + torch.randn(num_samps,n)/n

model.raw_update(X.unsqueeze(-2),Y.unsqueeze(-2),iters=10,lr=1,verbose=True)
What = model.A.mean()@model.B.mean().pinverse()
idx = model.logZ.argmax()
plt.scatter(W.numpy(),What[idx].numpy())
plt.title('Weights')
minW = W.min()
maxW = W.max()
plt.plot([minW,maxW],[minW,maxW],'k')
plt.show()

pY = model.predict(X.unsqueeze(-2).unsqueeze(-1))[0]
#pY = model.predict(X.unsqueeze(-1))
plt.scatter(Y.numpy(),pY.mean().squeeze(-1)[...,idx,:].numpy())
minY = Y.min()
maxY = Y.max()
plt.plot([minY,maxY],[minY,maxY],'k')
plt.title('Predictions')
plt.show()

