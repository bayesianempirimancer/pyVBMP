import torch, torchvision
from matplotlib import pyplot as plt
import time
from models.wip_BayesNet import *
from models.dMixtureofLinearTransforms import dMixtureofLinearTransforms
from models.MixtureofLinearTransforms import MixtureofLinearTransforms
from models.NLRegression import NLRegression_full_rank
from models.NLRegression import NLRegression_low_rank
from models.wip_BayesianTransformers import *

batch_size_train=4000
batch_size_test=1000

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((16,16)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((16,16)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

train = enumerate(train_loader)
test = enumerate(test_loader)

n=10
p=16*16#784

lr=0.1


# X = torch.randn(num_samples,p)
# X = X-X.mean(0,True)
# W = 2*torch.randn(p,n)/np.sqrt(10)
# Y_logits = X@W + torch.randn(num_samples,n)/100.0
# Y_logits = Y_logits-Y_logits.logsumexp(-1,True)
# Y = Y_logits.argmax(-1)
# Y = torch.eye(n)[Y]

#hidden_dims = (7,)
#latent_dims = (6,)

hidden_dims = (p//4,)
latent_dims = (32,)
model = BayesNet(n,p,hidden_dims,latent_dims)

iters = 50
#W_hat = MatrixNormalWishart(mu_0 = torch.zeros(n,p),pad_X=True)
t=time.time()
#m0.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
train = enumerate(train_loader)
test = enumerate(test_loader)
batch_idx, (X, Y) = next(train)
X = X.view(-1,p)
Y = torch.eye(10)[Y]
#X = torch.fft.fft2(X).abs().pow(2)
#stdX = X.std()
#muX = X.mean(0,True)
#X = (X-muX)/stdX

#X = X.view(-1,p)
#Y = torch.eye(10)[Y]
#m0.raw_update(X,Y,iters=2,lr=1.0)

# for batch_idx, (X, Y) in train:
# #    fftX = (torch.fft.fft2(X).abs().pow(2)-muX)/stdX
# #    fftX = fftX.view(-1,784)
# #    Yhat = m0.predict(fftX)

# #    X = (X-muX)/stdX
#     X = X.view(-1,p)
#     Y = torch.eye(10)[Y]
#     Yhat = model.predict(X)[0].squeeze(-1)
#     percent_correct = (Y.argmax(-1)==Yhat.argmax(-1)).float().mean()*100
#     print('iteration = ', batch_idx,'   percent correct train = ',percent_correct, ' in ',time.time()-t,' seconds per iteration')
#     t=time.time()
# #    m0.raw_update(fftX,Y,iters=1,lr=lr)
# #    m0.raw_update(X,Y,iters=3,lr=lr)
# #    m1.raw_update(X,Y,iters=1,lr=lr)
#     model.update(X,Y,iters=1,lr=lr,verbose=True,FBI=True)
#     Yhat = model.predict(X)[0].squeeze(-1)
#     percent_correct = (Y.argmax(-1)==Yhat.argmax(-1)).float().mean()*100
#     print('iteration = ', batch_idx,'   percent correct train = ',percent_correct, ' in ',time.time()-t,' seconds per iteration')
# # batch_idx, (X, Y) = next(test)
#     # X = X.view(-1,784)
#     # Y = torch.eye(10)[Y]

# batch_idx, (X, Y) = next(test)
# # fftX = torch.fft.fft2(X).abs().pow(2)
# # fftX = (fftX-muX)/stdX
# # fftX = fftX.view(-1,784)
# # Yhat = m0.predict(fftX)
# Y = torch.eye(10)[Y]
# X = X.view(-1,p)

m0=MultiNomialLogisticRegression(n,p,pad_X=False)
t=time.time()
m0.raw_update(X,Y,iters=iters,lr=0.5,verbose=True)
t_MNRL = time.time()-t

like_X = MultivariateNormal_vector_format(invSigmamu = torch.zeros(X.shape[-1:]+(1,)),invSigma = 1e-6*torch.eye(X.shape[-1]))
pxhat = m0.backward(torch.eye(10),like_X)[0]
xhat = pxhat.mean().reshape(10,16,16)
Yhat = m0.predict(X)
percent_correct_test = (Y.argmax(-1)==Yhat.argmax(-1)).float().mean()*100
print('MNLR percent correct test = ',percent_correct_test, 'in ', t_MNRL,' seconds' )
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(xhat[i].detach())
plt.show()

m1=dMixtureofLinearTransforms(n,p,mixture_dim=32,pad_X=True)
t=time.time()
m1.raw_update(X,Y,iters=iters,lr=0.5,verbose=True)
t_dMix = time.time()-t
pxhat = m1.postdict(torch.eye(10))[0]
xhat = pxhat.mean().reshape(10,16,16)
Yhat = m1.predict(X)[0].mean().squeeze(-1)
percent_correct_test = (Y.argmax(-1)==Yhat.argmax(-1)).float().mean()*100
print('dMix percent correct test = ',percent_correct_test, 'in ', t_dMix,' seconds' )
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(xhat[i].detach())
plt.show()

m2 = NLRegression_full_rank(n,p, mixture_dim=32)
t=time.time()
m2.raw_update(X,Y,iters=iters,lr=0.5,verbose=True)
t_NL = time.time()-t
Yhat = m2.predict(X)[0].mean().squeeze(-1)
percent_correct_test = (Y.argmax(-1)==Yhat.argmax(-1)).float().mean()*100
print('NL full rank percent correct test = ',percent_correct_test, 'in ', t_NL,' seconds' )

m3 = NLRegression_low_rank(n,p,hidden_dim=p//2,mixture_dim=32)
t=time.time()
m3.raw_update(X,Y,iters=iters,lr=0.5,verbose=True)
t_NL_low = time.time()-t
Yhat = m3.predict(X)[0].mean().squeeze(-1)
percent_correct_test = (Y.argmax(-1)==Yhat.argmax(-1)).float().mean()*100
print('NL low rank percent correct test = ',percent_correct_test, 'in ', t_NL_low,' seconds' )

m4 = MixtureofLinearTransforms(n,p,32,pad_X=True)
t=time.time()
m4.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1),iters=iters,lr=0.5,verbose=True)
t_MixLin = time.time()-t
Yhat = m4.predict(X.unsqueeze(-1))[0].mean().squeeze(-1)
percent_correct_test = (Y.argmax(-1)==Yhat.argmax(-1)).float().mean()*100
print('MixLin low rank percent correct test = ',percent_correct_test, 'in ', t_MixLin,' seconds' )

m5 = FocusedGenerativeBayesianTransformer(mixture_dim, role_dim, obs_dim, hidden_dim, batch_shape = (), pad_X=True)
m6 = GenerativeBayesianTransformer(mixture_dim, role_dim, obs_dim, hidden_dim, batch_shape = (), pad_X=True)


mua = m1.A.Elog_like_X(torch.eye(10).unsqueeze(-1).unsqueeze(-3))
mua = (mua[0].inverse()@mua[1]).squeeze(-1)
for i in range(10):
    for j in range(20):
      plt.subplot(20,10,i+10*j+1)
      plt.imshow(mua[i,j].reshape(16,16))
      plt.xticks([])
      plt.yticks([])
plt.show()

mua = m2.A.Elog_like_X(torch.eye(10).unsqueeze(-1).unsqueeze(-3))
mua = (mua[0].inverse()@mua[1]).squeeze(-1)
for i in range(10):
    for j in range(20):
      plt.subplot(20,10,i+10*j+1)
      plt.imshow(mua[i,j].reshape(16,16))
      plt.xticks([])
      plt.yticks([])
plt.show()

mua = m4.W.Elog_like_X(torch.eye(10).unsqueeze(-1).unsqueeze(-3))
mua = (mua[0].inverse()@mua[1]).squeeze(-1)
for i in range(10):
    for j in range(20):
      plt.subplot(20,10,i+10*j+1)
      plt.imshow(mua[i,j].reshape(16,16))
      plt.xticks([])
      plt.yticks([])
plt.show()




# #pY = MultivariateNormal_vector_format(mu = Y.unsqueeze(-1),invSigma=1000*torch.eye(n))
# #pY=Y
# px,Res = m0.backward(None,Y)
# #invSigma_x_x, invSigmamu_x, Residual = m0.Elog_like_X(Y.unsqueeze(-1))
# #mu_x = (invSigma_x_x.inverse()@invSigmamu_x)


# # plt.scatter(mu_x,px.mean().squeeze(-1))
# # plt.show()

# #Y_hat = W_hat.predict(X.unsqueeze(-1))[0]
# Y_hat = m0.predict(X)
# #MSE = ((Y-Y_hat.squeeze(-1))**2).mean()
# #Y_hat2 = W_hat.forward(MultivariateNormal_vector_format(mu = X.unsqueeze(-1),Sigma=torch.eye(p)/1000.0)).mean().squeeze(-1)
# m0_percent_correct =(Y.argmax(-1)==Y_hat.argmax(-1)).float().mean()

# # fig, axs = plt.subplots(2, 1, figsize=(6, 6))
# # # axs[0].scatter(W, m0.weights())
# # # axs[0].plot([W.min(), W.max()], [W.min(), W.max()])
# # # axs[0].set_title('W_hat vs W')
# # axs[0].scatter(X, px.mean().squeeze(-1))
# # #axs[0].scatter(X, mu_x.squeeze(-1))
# # axs[0].plot([X.min(), X.max()], [X.min(), X.max()])
# # axs[0].set_title('Backward Prediction')
# # axs[1].scatter(Y_logits, Y_hat.log())
# # #axs[2].scatter(Y, Y_hat2.squeeze(-1))
# # axs[1].plot([Y_logits.min(), Y_logits.max()], [Y_logits.min(), Y_logits.max()])
# # axs[1].set_title('Forward Prediction')
# # plt.tight_layout()
# # plt.show()
# # #print('MSE: ',MSE, '  Time: ',W_hat_runtime)
# # print('percent correct: ',m0_percent_correct*100, '  Time: ',m0_runtime)


# model = BayesNet(n,p,hidden_dims,latent_dims)


# t=time.time()
# model.update(X,Y,lr=lr,iters=iters,verbose=False,FBI=False)
# model_run_time=time.time()-t

# set_model = BayesNet(n,p,hidden_dims,latent_dims)
# # for k, layer in enumerate(model.layers): 
# #     layer.mu = torch.randn_like(layer.mu)/np.sqrt(p)*0.1
# #     set_model.layers[k].mu = layer.mu.clone().detach()
# t=time.time()
# set_model.update(X,Y,lr=lr,iters=iters,verbose=False,FBI=True)
# set_model_run_time=time.time()-t

# Yhat = model.predict(X)
# percent_correct = (Y.argmax(-1)==Yhat.argmax(-1)).float().mean()
# set_Yhat = set_model.predict(X)
# percent_correct_set = (Y.argmax(-1)==set_model.predict(X).argmax(-1)).float().mean()


# print('M0 percent correct: ',m0_percent_correct*100, '  Time: ',m0_runtime)
# print('Net percent correct: ',percent_correct*100, '  Time: ',model_run_time)
# print('FBI_Net percent correct: ',percent_correct_set*100, '  Time: ',set_model_run_time)


# # W_net = torch.eye(X.shape[-1])
# # for k, layer in enumerate(model.layers):
# #     W_net = layer.weights()@W_net
# # set_W_net = torch.eye(X.shape[-1])
# # for k, layer in enumerate(set_model.layers):
# #     set_W_net = layer.weights()@set_W_net

# # fig, axs = plt.subplots(3, 1, figsize=(6, 6))
# # axs[0].scatter(Y[:,0], Yhat.squeeze(-1)[:,0],c='b')
# # axs[0].scatter(Y[:,1], Yhat.squeeze(-1)[:,1],c='b')
# # axs[0].plot([Y.min(), Y.max()], [Y.min(), Y.max()])
# # axs[0].set_title('Prediction')
# # axs[1].plot(torch.tensor(model.ELBO_save[2:]).diff())
# # axs[1].set_title('Change in ELBO')
# # axs[2].plot(model.MSE[2:])
# # axs[2].set_title('MSE')
# # # axs[3].scatter(W, W_net)
# # # axs[3].plot([W.min(), W.max()], [W.min(), W.max()])
# # # axs[3].set_title('Weights')

# # # plt.tight_layout()
# # # plt.show()
# # print('MSE: ',model.MSE[-1], '  Time: ',model_run_time)

# # fig, axs = plt.subplots(3, 1, figsize=(6, 6))
# # axs[0].scatter(Y[:,0], set_Yhat.squeeze(-1)[:,0],c='orange')
# # axs[0].scatter(Y[:,1], set_Yhat.squeeze(-1)[:,1],c='orange')
# # axs[0].plot([Y.min(), Y.max()], [Y.min(), Y.max()])
# # axs[0].set_title('Prediction')
# # axs[1].plot(torch.tensor(set_model.ELBO_save[2:]).diff())
# # axs[1].set_title('Change in ELBO')
# # axs[2].plot(set_model.MSE[2:])
# # axs[2].set_title('MSE')
# # # axs[3].scatter(W, set_W_net)
# # # axs[3].plot([W.min(), W.max()], [W.min(), W.max()])
# # # axs[3].set_title('Weights')

# # plt.tight_layout()
# # plt.show()
# # print('set_MSE: ',set_model.MSE[-1], '  Time: ',set_model_run_time)

