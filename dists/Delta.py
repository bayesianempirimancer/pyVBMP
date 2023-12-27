import torch
# Implements a delta function distribution centered on the values of X
# The principle utility of this distribution is to allow for the treatment
# of observations as distributions which can emit required expectations

class Delta():
    def __init__(self,X):
        self.X = X

    def unsqueeze(self,dim):  # only appliles to batch
        return Delta(self.X.unsqueeze(dim))

    def squeeze(self,dim):  # only appliles to batch
        return Delta(self.X.squeeze(dim))

    def sum(self,dim,keepdim=False):
        return self.X.sum(dim,keepdim=keepdim)

    def cumsum(self,dim):
        return self.X.cumsum(dim)

    @property
    def shape(self):
        return self.X.shape

    def mean(self):
        return self.X

    def EX(self):
        return self.X

    def EXXT(self):
        return self.X@self.X.transpose(-1,-2)

    def EXTX(self):
        return self.X.transpose(-1,-2)@self.X

    def EXTAX(self,A):
        return self.X.transpose(-1,-2)@A@self.X

    def EXX(self):
        return self.X**2

    def ElogX(self):
        return torch.log(self.X)

    def E(self,f):
        return f(self.X)

    def logZ(self):
        return torch.zeros(self.batch_shape)


