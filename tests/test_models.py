# Tests Basic Functionality of models
#
import torch
# torch.set_default_device('cuda')  # This breaks all plotting routines (matplotlib doesn't work with device = cuda)
import numpy as np
import time
from matplotlib import pyplot as plt

print('Test Autoregressive Hidden Markov Model Variants...')
print('TEST Vanilla ARHMM')
from models.ARHMM import *

dim =6
batch_dim = 7
hidden_dim = 5
T = 100
num_samples = 200
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+4*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = torch.randn(hidden_dim,dim)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
y = torch.randn(T,num_samples,dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log() + torch.randn(1,num_samples,hidden_dim)).argmax(-1)
    y[t]= B[z[t]] + torch.randn(num_samples,dim)/5.0

Y=y[:,:,0:2]
X=y[:,:,2:5]

X=X.unsqueeze(-2).unsqueeze(-1)
Y=Y.unsqueeze(-2).unsqueeze(-1)

XY = (X,Y)
model = ARHMM(5,2,3)
model.update(XY,iters=20,lr=1,verbose=True)
loc= model.ELBO().argmax()

plt.plot(model.p[:,0,:].numpy())
plt.plot(z[:,0].numpy()-hidden_dim/2.0)
plt.show()


print('Test ARHMM prXY')
import dists
model = ARHMM_prXY(5,2,3)

pX = dists.Delta(X)
pY = dists.Delta(Y)
pXY = (pX,pY)
model.update(pXY,iters=20,lr=1,verbose=True)

plt.plot(model.p[:,0,:].numpy())
plt.plot(z[:,0].numpy()-hidden_dim/2.0)
plt.show()

Y = Y.unsqueeze(-4)
X = X.unsqueeze(-4)
pX = Delta(X)
pY = Delta(Y)
pXY = (pX,pY)

print('Test batch of ARHMM prXY')

model = ARHMM_prXY(5,2,3,batch_shape=(batch_dim,))
model.update(pXY,iters=20,lr=1,verbose=True)
loc= model.ELBO().argmax()
plt.plot(model.p[:,0,loc,:].numpy())
plt.plot(z[:,0].numpy()-hidden_dim/2.0)
plt.show()


print('Test ARHMM prXRY')
import dists
dim =6
rdim=2
xdim=3
batch_dim = 0
hidden_dim = 5
T = 100
num_samples = 200
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+4*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = torch.randn(hidden_dim,xdim,dim)
C = torch.randn(hidden_dim,rdim,dim)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
r = torch.randn(T,num_samples,rdim)
x = torch.randn(T,num_samples,xdim)
y = torch.randn(T,num_samples,dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log() + torch.randn(1,num_samples,hidden_dim)).argmax(-1)
    y[t]= (x[t].unsqueeze(-2)@B[z[t]]).squeeze(-2) + (r[t].unsqueeze(-2)@C[z[t]]).squeeze(-2) + torch.randn(num_samples,dim)/5.0

x=x.unsqueeze(-1).unsqueeze(-3)
pX = MultivariateNormal_vector_format(mu=x, Sigma = torch.zeros(x.shape[:-1] + (xdim,))+torch.eye(xdim)/10)
model = ARHMM_prXRY(5,dim,xdim,rdim,batch_shape=())
pXRY = (pX,r.unsqueeze(-1).unsqueeze(-3),y.unsqueeze(-1).unsqueeze(-3))
model.update(pXRY,iters=20,lr=1,verbose=True)
print('ARHMM TEST COMPLETE')


print('Test Bayesian Factor Analysis')
import torch
import models
from matplotlib import pyplot as plt

obs_dim=8
latent_dim=4
num_samps=200
model = models.BayesianFactorAnalysis(obs_dim, latent_dim,pad_X=False)

A=torch.randn(latent_dim,obs_dim)
Z=torch.randn(num_samps,latent_dim)
Y = Z@A + torch.randn(num_samps,obs_dim)/10.0

Y=Y-Y.mean(0,True)
A=A.transpose(-2,-1)
model.raw_update(Y,iters=10,lr=1,verbose=True)

Yhat = model.A.mean()@model.postdict(Y).mean()
from matplotlib import pyplot as plt
plt.scatter(Y.numpy(),Yhat.numpy())
plt.show()

plt.scatter((A@A.transpose(-2,-1)).numpy(),model.A.EXXT().numpy())
plt.show()


print('Test dHMM')
import torch
import numpy as np
from matplotlib import pyplot as plt
import dists
from models.dHMM import dHMM
obs_dim = 2
hidden_dim = 4
p=10 


T = 100
num_samples = 199
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+5*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = 2*torch.randn(hidden_dim,obs_dim)
C = torch.randn(hidden_dim,p,hidden_dim)/np.sqrt(p)

X = torch.randn(T,num_samples,p,1)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
Y = torch.randn(T,num_samples,obs_dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log()+(X[t]*C[z[t-1]]).sum(-2)+torch.randn(num_samples,hidden_dim)).argmax(-1)
    Y[t]= B[z[t]] + torch.randn(num_samples,obs_dim)/10.0


X = X.squeeze(-1)

obs_dist = dists.NormalInverseWishart(event_shape = (obs_dim,), batch_shape = (hidden_dim,))
model = dHMM(obs_dist=obs_dist,p=p)

model.raw_update(X,Y,iters=20,lr=0.5,verbose=True)


from matplotlib import pyplot as plt
plt.plot(-z[:,0].numpy())
plt.plot(model.p[:,0].numpy())
plt.show() 
B1 = B/(B*B).sum(-1,keepdim=True).sqrt()
B2 = model.obs_dist.mean()
B2 = B2/(B2*B2).sum(-1,keepdim=True).sqrt()

m,idx= (B1@B2.transpose(-2,-1)).max(-1)

print(B1@B2[idx].transpose(-2,-1))

print('Test dMixtureofLinearTransforms')
n=4
p=4
nc=3
num_samps=1000
import torch
import numpy as np
from torch.distributions import OneHotCategorical
import dists, transforms
from matplotlib import pyplot as plt

A = 4*torch.randn(nc,n,p)/np.sqrt(p)
B = 4*torch.randn(nc,n)/np.sqrt(p)
X = torch.randn(num_samps,p)
BX = torch.randn(nc,p)

W = 4*torch.randn(nc,p)/np.sqrt(p)
logits = X@W.transpose(-2,-1) + torch.randn(nc)
pr = (logits-logits.max(-1,True)[0]).exp()
pr = pr/pr.sum(-1,True)
label = OneHotCategorical(logits = logits).sample().argmax(-1)

X = X + BX[label]
Y = ((A[label]@X.unsqueeze(-1)).squeeze(-1)+B[label])
# Y = Y + torch.randn_like(Y)/100/

model = transforms.dMixtureofLinearTransforms(n,p,nc+1,batch_shape=(),pad_X=True)
for i in range(model.batch_dim):
    X = X.unsqueeze(-2)
    Y = Y.unsqueeze(-2)

model.raw_update(X,Y,iters=20,lr=1.0,verbose=True)

# pX = MultivariateNormal_vector_format(invSigmamu=100*X.unsqueeze(-1), invSigma = torch.eye(n)*100)
# pY = MultivariateNormal_vector_format(invSigmamu=100*Y.unsqueeze(-1), invSigma = torch.eye(n)*100)
#model.update(pX,pY,iters=20,lr=1,verbose=True)

pY, p = model.predict(X)
px,logz,pb = model.postdict(Y)

ELL = model.ELBO_last
if ELL.ndim>0:
    m,idx = ELL.max(0)
    mu = pY.mean()[:,idx]
    Sigma = pY.EinvSigma()[:,idx]
    invSigmamu = pY.EinvSigmamu()[:,idx]
    invSigma = pY.EinvSigma()[:,idx]
    p=p[:,idx]
    muX = px.mean()[:,idx]
    Abar = model.A.weights()[idx]
else:
    Abar = model.A.weights()
    muX = px.mean()
    mu = pY.mean()

idx = (p.unsqueeze(-1)*pr.unsqueeze(-2)).mean(0).argmax(-2)
plt.scatter(pr.log().numpy(),p[:,idx].log().numpy())
plt.plot([p.log().min().numpy(),p.log().max().numpy()],[p.log().min().numpy(),p.log().max().numpy()])
plt.title('log Assignment Probabilities')
plt.xlabel('True')
plt.ylabel('Estimated')
plt.show()

plt.scatter(A.numpy(),Abar[idx].numpy(),c=idx.unsqueeze(-1).unsqueeze(-1).expand(A.shape).numpy())
plt.plot([A.min().numpy(),A.max().numpy()],[A.min().numpy(),A.max().numpy()])
plt.title('Regression Weights')
plt.xlabel('True')
plt.ylabel('Estimated')
plt.show()

plt.scatter(X.numpy(),muX.squeeze(-1).numpy())
plt.plot([X.min(),X.max()],[X.min(),X.max()])
plt.title('Regressors: X from Backward routine')
plt.xlabel('True')
plt.ylabel('Estimated')
plt.show()

plt.scatter(Y.numpy(),mu.squeeze(-1).numpy(),c=p.argmax(-1,True).expand(-1,n).numpy())
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()])
plt.title('Predictions')
plt.xlabel('True')
plt.ylabel('Estimated')
plt.show()

mu = mu.squeeze(-1)
plt.scatter(Y[:,0].numpy(),Y[:,1].numpy(),c=logits.argmax(-1).numpy())
plt.title('True Labels')
plt.show()

plt.scatter(mu[:,0].numpy(),mu[:,1].numpy(),c=p.argmax(-1).numpy())
plt.title('Predited Labels')
plt.show()

MSE = ((Y-mu)**2).mean()
PVE = 1 - MSE/Y.var()
print('Percent Variance Explained :  ',PVE*100)

hit = (model.pi.predict(X)[:,idx].argmax(-1)==label).float().mean()
print('hit probability = ',hit)


print('TEST HMM')
import time
import torch
import numpy as np
from models.HMM import HMM
from models.HHMM import HHMM
import dists
from matplotlib import pyplot as plt

print("TEST VANILLA HMM")
obs_dim = 2
hidden_dim = 12
T = 100
num_samples = 99
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+4*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = torch.randn(hidden_dim,obs_dim)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
y = torch.randn(T,num_samples,obs_dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log() + torch.randn(1,num_samples,hidden_dim)).argmax(-1)
    y[t]= B[z[t]] + torch.randn(num_samples,obs_dim)/5.0

hidden_dim = 24
obs_dist = dists.NormalInverseWishart(event_shape = (obs_dim,), batch_shape=(hidden_dim,))
model = HMM(obs_dist)  
t = time.time()
model.update(y,T=None,iters=20,lr=1,verbose=True)
print('HMM time = ',time.time()-t)
plt.scatter(y[...,0].numpy(),y[...,1].numpy(),c=model.assignment().numpy())
plt.show()

print('TEST Vanilla HHMM')
dims = (2,3,4)
obs_dist = dists.NormalInverseWishart(event_shape = (obs_dim,), batch_shape = dims)
model = HHMM(obs_dist,3)
t = time.time()
model.update(y,iters=20,lr=1,verbose=True)
print('HHMM time = ',time.time()-t)
plt.scatter(y[...,0].numpy(),y[...,1].numpy(),c=model.p.reshape(T,num_samples,24).argmax(-1).numpy())
plt.show()

print('Test Discrete Generalized Coorinate model')
obs_dist = dists.NormalInverseWishart(event_shape = (obs_dim,), batch_shape = (len(dims)-1)*(1,) + dims[-1:])
model = HHMM(obs_dist,event_dim = 3, event_shape=dims)
t=time.time()
model.update(y,iters=20,lr=1,verbose=True)
print('HHMM time = ',time.time()-t)
plt.scatter(y[...,0].numpy(),y[...,1].numpy(),c=model.p.reshape(T,num_samples,24).argmax(-1).numpy())
plt.show()

print('TEST TENSOR_HMM')
from models.Tensor_HMM import Tensor_HMM
import dists
dims = (2,3,4)
obs_dist = dists.NormalInverseWishart(event_shape = (obs_dim,), batch_shape = dims)
model = Tensor_HMM(obs_dist,event_shape=dims)
t=time.time()
model.update(y,iters=20,lr=1,verbose=True)
print('Tensor_HMM time = ',time.time()-t)
plt.scatter(y[...,0].numpy(),y[...,1].numpy(),c=model.p.reshape(T,num_samples,24).argmax(-1).numpy())
plt.show()


print('TEST BATCH OF HMMS')
batch_size = 3
obs_dist = dists.NormalInverseWishart(event_shape = (obs_dim,), batch_shape = (3,24))

model = HMM(obs_dist)  
model.update(y.unsqueeze(-2),20,verbose=True)

ELBO = model.ELBO()
loc = ELBO.argmax()
print(ELBO - ELBO[loc])
plt.scatter(y[:,0,0].numpy(),y[:,0,1].numpy(),c=model.assignment()[:,0,loc].numpy())
plt.show()
plt.plot(model.p[:,0,loc,:].numpy())
plt.plot(z[:,0].numpy()-hidden_dim/2.0)
plt.show()


print('TEST BATCH OF HHMMs')

batch_shape =(3,)
dims = (2,3,4)
obs_dist = dists.NormalInverseWishart(event_shape = (obs_dim,), batch_shape = batch_shape+dims)
model = HHMM(obs_dist,3)
t = time.time()
model.update(y.unsqueeze(-2),iters=20,lr=1,verbose=True)
print('HHMM time = ',time.time()-t)


print('TEST HMM non-trivial event_dim')

dim =6
hidden_dim = 5
T = 100
num_samples = 199
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+4*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = torch.randn(hidden_dim,dim)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
y = torch.randn(T,num_samples,dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log() + torch.randn(1,num_samples,hidden_dim)).argmax(-1)
    y[t]= B[z[t]] + torch.randn(num_samples,dim)/5.0


y = y.reshape(T,num_samples,3,2)

batch_size = 3
dim = 2
lambda_mu_0 = torch.ones(hidden_dim,batch_size)
mu_0 = torch.zeros(hidden_dim,batch_size,dim)
nu_0 = torch.ones(hidden_dim,batch_size)*(dim+2)
invSigma_0 = torch.zeros(hidden_dim,batch_size,dim,dim)+torch.eye(dim)
obs_dist = dists.NormalInverseWishart(event_shape = (batch_size,dim), batch_shape = (hidden_dim,))

model = HMM(obs_dist)  
model.update(y,10,verbose=True)


y = y.reshape(T,num_samples,6)
plt.scatter(y[:,0,0],y[:,0,1],c=model.assignment()[:,0])
plt.show()
plt.plot(model.p[:,0,:].numpy())
plt.plot(z[:,0].numpy()-hidden_dim/2.0)
plt.show()

print('HMM TEST COMPLETE')


print('TEST LinearDynamicalSystem')
import torch
# torch.set_default_device('cuda')  # This breaks all plotting routines (matplotlib doesn't work with device = cuda)

import numpy as np
from models.LinearDynamicalSystems import LinearDynamicalSystems
from matplotlib import pyplot as plt
import time
dt = 0.2
num_systems = 6
obs_dim = 4
hidden_dim = 2
control_dim = 2
regression_dim = 3


#A_true = torch.randn(hidden_dim,hidden_dim)/(hidden_dim) 
#A_true = -A_true @ A_true.transpose(-1,-2) * dt + torch.eye(hidden_dim)
C_true = 0.05*torch.randn(hidden_dim,control_dim)/control_dim
A_true = torch.eye(2) + dt*torch.tensor([[-0.01,-1.0],[1.0,-0.01]])
#A_true = torch.eye(4) + dt*torch.randn(4,4)/10.0
B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
D_true = 0.05*torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)

Tmax = 200
batch_num = 2
sample_shape = (Tmax,batch_num)
num_iters = 20
y = torch.zeros(Tmax,batch_num,obs_dim)
x = torch.zeros(Tmax,batch_num,hidden_dim)
x[0] = torch.randn(batch_num,hidden_dim)
y[0] = x[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)*0.02
u = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
r = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

for t in range(1,Tmax):
    x[t] = x[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u[t] @ C_true.transpose(-1,-2)*dt 
    y[t] = x[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim)/20.0 + r[t] @ D_true.transpose(-1,-2) 

y2 = y.reshape(y.shape[:-1]+(2,2))
r2 = r.unsqueeze(-2).repeat(1,1,3,1)


print('TEST LDS VANILLA NO REGRESSORS OR CONTROLS or BIAS TERMS')
obs_shape = (obs_dim,)
sample_shape = (Tmax,batch_num)
t = time.time()
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim=-1,regression_dim=-1,latent_noise='indepedent')
lds.update(y,iters=15,lr=1,verbose=True)
print('LDS time = ',time.time()-t)
fbw_mu = lds.px.mean().squeeze()
fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

xp=fbw_mu[:,0,0].numpy()
yp=fbw_mu[:,0,1].numpy()
xerr=fbw_Sigma[:,0,0].numpy()
yerr=fbw_Sigma[:,1,1].numpy()

plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
plt.plot(xp[:-1],yp[:-1])
plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
plt.show()

print('TEST LDS VANILLA with BIAS TERMS')
obs_shape = (obs_dim,)
sample_shape = (Tmax,batch_num)
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim=-1,regression_dim=0,latent_noise='indepedent')
lds.update(y,iters=20,lr=1,verbose=True)
fbw_mu = lds.px.mean().squeeze()
fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

xp=fbw_mu[:,0,0].numpy()
yp=fbw_mu[:,0,1].numpy()
xerr=fbw_Sigma[:,0,0].numpy()
yerr=fbw_Sigma[:,1,1].numpy()

plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
plt.plot(xp[:-1],yp[:-1])
plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
plt.show()

print('TEST LDS VANILLA with control BIAS TERM only')
obs_shape = (obs_dim,)
sample_shape = (Tmax,batch_num)
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim=0,regression_dim=-1,latent_noise='indepedent')
lds.update(y,iters=20,lr=1,verbose=True)
fbw_mu = lds.px.mean().squeeze()
fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

xp=fbw_mu[:,0,0].numpy()
yp=fbw_mu[:,0,1].numpy()
xerr=fbw_Sigma[:,0,0].numpy()
yerr=fbw_Sigma[:,1,1].numpy()

plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
plt.plot(xp[:-1],yp[:-1])
plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
plt.show()

print('TEST LDS WITH REGRESSORS AND CONTROLS and full noise model')
obs_shape = (obs_dim,)
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='shared')
lds.update(y,u,r,iters=20,lr=1,verbose=True)
fbw_mu = lds.px.mean().squeeze()
fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

xp=fbw_mu[:,0,0].numpy()
yp=fbw_mu[:,0,1].numpy()
xerr=fbw_Sigma[:,0,0].numpy()
yerr=fbw_Sigma[:,1,1].numpy()

plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
plt.plot(xp,yp)
plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
plt.show()


print('TEST LDS WITH REGRESSORS AND CONTROLS and non-trivial event_shape and independent noise and non-trivial batch_shape')

import torch
import numpy as np
from models.LinearDynamicalSystems import LinearDynamicalSystems
from matplotlib import pyplot as plt

Tmax = 100
dt=0.2
batch_num = 99
sample_shape = (Tmax,batch_num)
obs_dim = 6
hidden_dim = 2
num_iters = 20
control_dim = 2
regression_dim = 2
C_true = torch.randn(hidden_dim,control_dim)/control_dim
A_true = torch.eye(2) + dt*torch.tensor([[-0.01,-1.0],[1.0,-0.01]])
B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)
y = torch.zeros(Tmax,batch_num,obs_dim)
x = torch.zeros(Tmax,batch_num,hidden_dim)
x[0] = torch.randn(batch_num,hidden_dim)
y[0] = x[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
u = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
r = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

for t in range(1,Tmax):
    x[t] = x[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u[t] @ C_true.transpose(-1,-2)*dt 
    y[t] = x[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r[t] @ D_true.transpose(-1,-2) 
T=(torch.ones(batch_num)*Tmax).long()

y2 = y.reshape(y.shape[:-1]+(3,2))
r2 = r.unsqueeze(-2).repeat(1,1,3,1)
u2 = u
obs_shape = (3,2)
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='indepedent',batch_shape=(4,))

lds.update(y2.unsqueeze(-3),u2.unsqueeze(-2),r2.unsqueeze(-3),iters=20,lr=1,verbose=True)
fbw_mu = lds.px.mean().squeeze()
fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

m,idx = lds.ELBO().max(-1)

xp=fbw_mu[:,0,idx,0].numpy()
yp=fbw_mu[:,0,idx,1].numpy()
xerr=fbw_Sigma[:,0,idx,0].numpy()
yerr=fbw_Sigma[:,0,idx,1].numpy()

plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
plt.plot(xp,yp)
plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
plt.show()



print('TEST Mixture of Linear Dynamical Systems')
import torch
import numpy as np
from models.MixtureofLinearDynamicalSystems import *
dt = 0.2
num_systems = 6
obs_dim = 6
hidden_dim = 2
control_dim = 2
regression_dim = 3


#A_true = torch.randn(hidden_dim,hidden_dim)/(hidden_dim) 
#A_true = -A_true @ A_true.transpose(-1,-2) * dt + torch.eye(hidden_dim)
C_true = torch.randn(hidden_dim,control_dim)/control_dim
A_true = torch.eye(2) + dt*torch.tensor([[-0.01,-1.0],[1.0,-0.01]])
B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)

Tmax = 100
batch_num = 99
sample_shape = (Tmax,batch_num)
num_iters = 20
y = torch.zeros(Tmax,batch_num,obs_dim)
x = torch.zeros(Tmax,batch_num,hidden_dim)
x[0] = torch.randn(batch_num,hidden_dim)
y[0] = x[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
u = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
r = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

for t in range(1,Tmax):
    x[t] = x[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u[t] @ C_true.transpose(-1,-2)*dt 
    y[t] = x[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r[t] @ D_true.transpose(-1,-2) 

y2 = y.reshape(y.shape[:-1]+(3,2))
r2 = r.unsqueeze(-2).repeat(1,1,3,1)

C_true = torch.randn(hidden_dim,control_dim)/control_dim
A_true = torch.eye(2) + dt*torch.tensor([[-0.01,1.0],[-1.0,-0.01]])
B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)
Tmax = 100
batch_num = 99
sample_shape = (Tmax,batch_num)
num_iters = 20
y2 = torch.zeros(Tmax,batch_num,obs_dim)
x2 = torch.zeros(Tmax,batch_num,hidden_dim)
x2[0] = torch.randn(batch_num,hidden_dim)
y2[0] = x2[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
u2 = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
r2 = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

for t in range(1,Tmax):
    x2[t] = x2[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u2[t] @ C_true.transpose(-1,-2)*dt 
    y2[t] = x2[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r2[t] @ D_true.transpose(-1,-2) 
T=(torch.ones(batch_num)*Tmax).long()

bigy = torch.cat([y,y2],dim=1)
bigu = torch.cat([u,u2],dim=1)
bigr = torch.cat([r,r2],dim=1)
bigT = torch.cat([T,T],dim=0)

model = MixtureofLinearDynamicalSystems(num_systems,(obs_dim,),hidden_dim,control_dim,regression_dim)
import time
t= time.time()
model.update(bigy,bigu,bigr,iters=20,lr=1)
print(time.time()-t)



print('TEST LDS MIXTURE WITH REGRESSORS AND CONTROLS and non-trivial event_shape and independent noise')


y2 = bigy.reshape(bigy.shape[:-1]+(3,2))
r2 = bigr.unsqueeze(-2).repeat(1,1,3,1)
u2 = bigu
obs_shape = (3,2)
model = MixtureofLinearDynamicalSystems(10,obs_shape,hidden_dim,control_dim,regression_dim)
model.update(y2,u2,r2,iters=20,lr=1)
print(model.NA)

print('LDS MIXTURE TEST COMPLETE')




print('TEST dMIXTURE and Mixture of Linear Transforms')
import torch
import numpy as np
import matplotlib.pyplot as plt
import dists, transforms
from torch.distributions import OneHotCategorical 

dim = 2
p=10
n = 2
n_samples = 1000
w_true = torch.randn(dim,n,p)/np.sqrt(p)
b_true = 2.0*torch.randn(dim,n)
beta_true = 4*torch.randn(dim,p)/np.sqrt(p)
beta_x_true = 2.0*torch.randn(dim,p)

Z = OneHotCategorical(logits = torch.zeros(n_samples,dim)).sample().argmax(-1)
X= torch.randn(n_samples,p) + beta_x_true[Z]

# Z = Z.argmax(-1)
#logits = X@beta_true.transpose(-2,-1)

Y = (w_true[Z]@X.unsqueeze(-1)).squeeze(-1) + b_true[Z]

Y = Y + torch.randn_like(Y)/100.0
pY = dists.MultivariateNormal_vector_format(invSigmamu = 100*Y.unsqueeze(-1), invSigma = 100*torch.eye(n))
pX = dists.MultivariateNormal_vector_format(invSigmamu = 100*X.unsqueeze(-1), invSigma = 100*torch.eye(p))

# model = dMixtureofLinearTransforms(n,p,dim+2,pad_X=True,type = 'Wishart')
# model.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1),iters=20,lr=1,verbose=True)
model = transforms.dMixtureofLinearTransforms(n,p,dim,pad_X=False,type = 'Gamma')
model.raw_update(X,Y,iters=20,lr=1,verbose=True)

model2 = transforms.MixtureofLinearTransforms(n,p,dim,pad_X=False,type = 'Gamma')
model2.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1),iters=20,lr=1,verbose=True)

Yhat = model.predict(X)[0].mean().squeeze(-1)
assignments = model.pi.predict(X).argmax(-1)
plt.scatter(Y[:,0],Y[:,1],c=assignments)
plt.show()
plt.scatter(Y[:,0],Y[:,1],c=model2.p.argmax(-1,True))
plt.show()

plt.scatter(Y.numpy(),Yhat.numpy(),c=assignments.unsqueeze(-1).expand(-1,n).numpy())
plt.title('dMixtures of Linear Transforms Predictions')
plt.show()


Yhat2 = model2.predict(X.unsqueeze(-1))[0].mean().squeeze()
plt.scatter(Y.numpy(),Yhat2.numpy(),c=model2.p.argmax(-1,True).expand(-1,n).numpy())
plt.title('Mixtures of Linear Transforms Predictions')
plt.show()
print('dMix Percent Variance Explained = ',100-((Y-Yhat)**2).mean()/Y.var()*100)
print('Mix Percent Variance Explained = ',100-((Y-Yhat2)**2).mean()/Y.var()*100)


print('Test Multinomial Logistic Regression with ARD')
import torch
import numpy as np
from  matplotlib import pyplot as plt
import dists, transforms

n=10
p=20
num_samples =1000
W = 4*torch.randn(n,p)/np.sqrt(p)
X = torch.randn(num_samples,p)@torch.randn(p,p)/np.sqrt(p)
X = X-X.mean(0,True)
X = X/X.std()
B = torch.randn(n)


logpY = X@W.transpose(-2,-1)+B
pY = (logpY - logpY.logsumexp(-1,True)).exp()

Y = torch.distributions.OneHotCategorical(logits = logpY).sample()

model = transforms.MultiNomialLogisticRegression(n,p,pad_X=True)
mini_batch_size = 100
num_mini_batches = X.shape[0]//mini_batch_size

for i in range(num_mini_batches):
    Xtrain = X[i*mini_batch_size:(i+1)*mini_batch_size]
    Ytrain = Y[i*mini_batch_size:(i+1)*mini_batch_size]
    model.raw_update(Xtrain,Ytrain,iters = 1,verbose=True,beta=1)



#model.update(Delta(X.unsqueeze(-1)),Y,iters =4)
W_hat = model.beta.mean().squeeze(-1)
W_true = (W[:-1] - W[-1:])
W_hat = 2*W_hat - W_hat.cumsum(0)
if model.pad_X is True:
    W_hat = W_hat[:,:-1]
plt.scatter(W_true.numpy(),W_hat.numpy())
plt.plot([W_true.min(),W_true.max()],[W_true.min(),W_true.max()])
plt.title('weights')
plt.show()

print('Predictions by lowerbounding with q(w|b,<psi^2>)')
psb = model.predict(X)
for i in range(n):
    plt.scatter(pY.log()[:,i].numpy(),psb.log()[:,i].numpy())    
plt.plot([pY.log().min(),0],[pY.log().min(),0])
plt.show()
# for i in range(n):
#     plt.scatter(pY[:,i],psb[:,i])    
# plt.plot([0,1],[0,1])
# plt.show()

print('Predictions by marginaling out q(beta) with w = <w|b,<psi^2>>')
psb2 = model.predict_2(X)
for i in range(n):
    plt.scatter(pY.log()[:,i].numpy(),psb2.log()[:,i].numpy())    
plt.plot([pY.log().min(),0],[pY.log().min(),0])
plt.show()
psb2 = model.predict(X)
# for i in range(n):
#     plt.scatter(pY[:,i],psb2[:,i])    
# plt.plot([0,1],[0,1])
# plt.show()
print('Percent Correct (best possible)', ((Y.argmax(-1)==pY.argmax(-1)).sum()/Y.shape[0]).numpy()*100)
print('Percent Correct   = ',((Y.argmax(-1)==psb.argmax(-1)).sum()/Y.shape[0]).numpy()*100)
print('Percent Correct_2 = ',((Y.argmax(-1)==psb2.argmax(-1)).sum()/Y.shape[0]).numpy()*100)

xbar = X[Y.argmax(-1)==0].mean(0,True)
for i in range(1,n):
    xbar = torch.cat((xbar,X[Y.argmax(-1)==i].mean(0,True)),dim=-2)

plt.scatter(xbar.numpy(),model.backward(torch.eye(n),None)[0].mean().squeeze().numpy())
plt.plot([xbar.min(),xbar.max()],[xbar.min(),xbar.max()])
plt.title('backward estimate')
plt.show()


print('TEST NL REGRESSION')

import time
import torch
import numpy as np
from matplotlib import pyplot as plt
import transforms

n=1
p=10
hidden_dim = 2
nc =  20
num_samps=800
batch_shape = ()
t=time.time()
X = 4*torch.rand(num_samps,p)-2
Y = torch.randn(num_samps,n)
W_true = 5.0*torch.randn(p,n)/np.sqrt(p)
pad_X = True

Y = (X@W_true).tanh() + torch.randn(num_samps,1)/20*0
#Y = (X@W_true)*(X@W_true) + torch.randn(num_samps,1)/20*0

X=X/X.std()
Y=Y/Y.std()
Y=Y-Y.mean()

model0 = transforms.NLRegression_low_rank(n,p,hidden_dim,nc,batch_shape=batch_shape)
model1 = transforms.NLRegression_full_rank(n,p,nc,batch_shape=batch_shape)
model2 = transforms.dMixtureofLinearTransforms(n,p,nc,batch_shape=batch_shape,pad_X=pad_X)
model3 = transforms.NLRegression_Multinomial(n,p,nc,batch_shape=batch_shape)
model4 = transforms.MixtureofLinearTransforms(n,p,nc,batch_shape=batch_shape,pad_X=pad_X)
models = (model0,model1,model2,model3,model4)
predictions = []
inference_cost=[]
prediction_cost=[]

for k, model in enumerate(models):
    print('Training Model ',k)
    t= time.time()
    if(k==4):
        model.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1),iters = 40,lr=1,verbose=True)
        inference_cost.append(time.time()-t)
        predictions.append(model.predict(X.unsqueeze(-1))[0].mean().squeeze(-1))
        prediction_cost.append(time.time()-t)
    else:
        model.raw_update(X,Y,iters = 40,lr=1,verbose=True)
        inference_cost.append(time.time()-t)
        t= time.time()
        predictions.append(model.predict(X)[0].mean())
        prediction_cost.append(time.time()-t)


print('inference_cost = ',inference_cost)
print('prediction_cost = ',prediction_cost)
label = ['Low Rank','Full Rank','dMix','NL Multinomial','Mix Linear']


U_true = X@W_true
U_true = U_true/U_true.std(0,True)
plt.scatter(U_true.numpy(),Y.numpy(),c='black')
for k, pred in enumerate(predictions):
    plt.scatter(U_true,pred[...,0].numpy(),alpha=0.5)
plt.legend(['True']+label)
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

print('Test Poisson Mixture Model')
from models.PoissonMixtureModel import *
from matplotlib import pyplot as plt
mu = torch.rand(4,10)*20
X = torch.zeros(200,10)

for i in range(200):
    X[i,:] = torch.poisson(mu[i%4,:])

model = PoissonMixtureModel(4,10)
model.update(X,iters=10,verbose=True)
plt.scatter(X[:,0].numpy(),X[:,1].numpy(),c=model.assignment().numpy(),alpha=model.assignment_pr().max(-1)[0].numpy())
plt.show()

print('Test Poisson Mixture Model')
import torch
import models
from matplotlib import pyplot as plt
mu = torch.rand(4,10)*20
X = torch.zeros(200,10)

for i in range(200):
    X[i,:] = torch.poisson(mu[i%4,:])

model = models.PoissonMixtureModel(nc=4,dim=10)
model.update(X,iters=10,verbose=True)
plt.scatter(X[:,0].numpy(),X[:,1].numpy(),c=model.assignment().numpy(),alpha=model.assignment_pr().max(-1)[0].numpy().numpy())
plt.show()