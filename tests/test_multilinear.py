import torch
import transforms.MultiLinearNormalWishart as MultiLinearNormalWishart
import transforms
import matplotlib.pyplot as plt
import time

num_samps = 10000

p_list = 20*(20,)
p = sum(p_list)
n=2
X=[]
XX=torch.zeros(num_samps,0,1)
for i in range(len(p_list)):
    X.append(torch.randn(p_list[i],p_list[i])/torch.sqrt(torch.tensor(p_list[i],requires_grad=False))@torch.randn(num_samps,p_list[i],1))
    XX = torch.cat((XX,X[i]),dim=-2)

W_true = torch.randn(n,p)/torch.sqrt(torch.tensor(p,requires_grad=False))
Y = W_true@XX + torch.randn(num_samps,n,1)*0.1

m0 = transforms.MatrixNormalWishart(batch_shape = (), event_shape = (n,p),pad_X=False)
m1 = MultiLinearNormalWishart(n,p_list,batch_shape=(),pad_X=False)

t = time.time()
m0.raw_update(XX,Y,lr=1)
t0 = time.time()-t
Yhat_0 = m0.weights()@XX + m0.bias()

t = time.time()
m1.raw_update(X,Y,iters=1,lr=1)
t1 = time.time()-t

What = torch.zeros(n,0)
bhat = torch.zeros(n,1)
for i in range(len(p_list)):
    What = torch.cat((What,m1.A[i].weights()),dim=-1)
    bhat = bhat + m1.A[i].bias()
plt.scatter(W_true,What)
plt.scatter(W_true,m0.weights())
plt.show()

Yhat_1 = What@XX + bhat
plt.scatter(Y,Yhat_1)
plt.scatter(Y,Yhat_0)
plt.show()
print('m0 MSE = ',((Y-Yhat_0)**2).mean())
print('m1 MSE = ',((Y-Yhat_1)**2).mean())

print('m0 update time', t0)
print('m1 update time', t1)