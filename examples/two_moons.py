from matplotlib import pyplot as plt
import torch

x = torch.linspace(-torch.pi/2, torch.pi/2, 200)
y1 = torch.sin(x)
y2 = torch.cos(x)-0.25

y3 = torch.sin(x)+1.0
y4 = -torch.cos(x)+0.25

X1  = torch.stack([y1,y2],axis=1)
X2  = torch.stack([y3,y4],axis=1)

X1 = X1 + 0.05*torch.randn(X1.shape)
X2 = X2 + 0.05*torch.randn(X2.shape)

# plt.scatter(X1[:,0],X1[:,1])
# plt.scatter(X2[:,0],X2[:,1])

X = torch.concatenate([X1,X2],axis=0) 
X = X/X.std()
Z = torch.concatenate([torch.zeros(X1.shape[0]),torch.ones(X2.shape[0])],axis=0)
Z = torch.eye(2)[Z.long()]
import dists, transforms

use_bias = True


dim0 = 2
nc1 = 10
dim1 = 4
#dim2 = 4
fixed_precision=True
scale = 0.1
layer1 = transforms.dMixtureofLinearTransforms(dim1,dim0,nc1,pad_X=use_bias, fixed_precision=fixed_precision)
layer2 = transforms.dMixtureofLinearTransforms(2,dim1,nc1,pad_X=use_bias, fixed_precision=fixed_precision)
# layer3 = transforms.MultiNomialLogisticRegression(2,dim2)
pX = dists.MultivariateNormal_vector_format(invSigmamu = X.unsqueeze(-1)*100, invSigma = 100*torch.eye(dim0))
lr=0.9


pZ = dists.MultivariateNormal_vector_format(invSigmamu = Z.unsqueeze(-1)*1000, invSigma = 1000*torch.eye(2))

for i in range(40):
    p1 = layer1.forward(pX)
#    p2 = layer2.forward(p1)
    bw = layer2.backward(pZ)[0]

    # p2.invSigma = p2.EinvSigma() + bw.EinvSigma()
    # p2.invSigmamu = p2.EinvSigmamu() + bw.EinvSigmamu()
    # p2.mu = None
    # p2.Sigma = None

    # bw = layer2.backward(p2)[0]
    p1.invSigma = p1.EinvSigma() + bw.EinvSigma()
    p1.invSigmamu = p1.EinvSigmamu() + bw.EinvSigmamu()
    p1.mu = None
    p1.Sigma = None

    layer1.update(pX,p1,lr=lr)
    layer2.update(p1,pZ,lr=lr)
#    layer3.update(p1,Z,lr=lr)

# Zhat = layer3.forward(layer2.forward(layer1.forward(pX)))
Zhat = layer2.forward(layer1.forward(pX)).mean().squeeze(-1)
plt.scatter(X[:,0],X[:,1],c=Zhat.argmax(-1))



