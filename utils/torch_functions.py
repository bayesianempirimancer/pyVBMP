import torch
def stable_logsumexp(x,dims,keepdim=False):
    xmax = x.amax(dims,keepdim=True)
    return xmax.amax(dims,keepdim) + (x-xmax).exp().sum(dim=dims,keepdim=keepdim).log()

def stable_softmax(x,dims):
    return x - stable_logsumexp(x,dims,keepdim=True)
    
def logmatmulexp(x,y):
    x_shift = x.max(-1, keepdim=True)[0]
    y_shift = y.max(-2, keepdim=True)[0]
    xy = torch.matmul((x - x_shift).exp(), (y - y_shift).exp()).log()
    return xy + x_shift + y_shift