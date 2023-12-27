import torch
import transforms.MatrixNormalWishart as MatrixNormalWishart
import transforms.MatrixNormalGamma as MatrixNormalGamma
import dists.Wishart as Wishart
import dists.DiagonalWishart as DiagonalWishart
import dists.MultivariateNormal_vector_format as MultivariateNormal_vector_format
import dists.NormalInverseWishart_vector_format as NormalInverseWishart_vector_format

class MultiLinearNormalWishart():
    # This class performs inference for a multilinear model of the form Y = A_1@X_1 + A_2@X_2 + ... + B 
    # with an approximate posterior distribution that factorizes over the A's and the B.  In order to handle the 
    # case where the different X_i's have different sizes we 
    def __init__(self, n, p_list, batch_shape=(), mask_list=None, X_mask_list=None, pad_X=False, noise_type = 'Wishart'):
        print('MultiLinearNormalWishart: not working')
        self.noise_type = noise_type
        self.pad_X = pad_X
        self.p_list = p_list
        self.n = n
        self.event_dim = 2  
        self.batch_dim = len(batch_shape)
        self.event_shape = (n,0)
        self.batch_shape = batch_shape
        if mask_list is None:
            mask_list = [None]*len(self.p_list)
        if X_mask_list is None:
            X_mask_list = [None]*len(self.p_list)

        self.A = []
        if noise_type == 'Wishart':
            self.invU = Wishart(event_shape = (n,n), batch_shape = batch_shape)
            for i in range(len(self.p_list)):
                self.A.append(MatrixNormalWishart((n,self.p_list[i]), batch_shape = batch_shape,
                                                  mask=mask_list[i],X_mask=X_mask_list[i],
                                                  fixed_precision=True,pad_X=False))
                self.A[i].invU = self.invU
        elif noise_type == 'Gamma':
            self.invU = DiagonalWishart((n,), batch_shape = batch_shape)
            for i in range(len(self.p_list)):
                self.A.append(MatrixNormalGamma((n,self.p_list[i]),batch_shape = batch_shape,
                                                mask=mask_list[i],X_mask=X_mask_list[i],
                                                fixed_precision=True,pad_X=False))
                self.A[i].invU = self.invU        
        self.bias = NormalInverseWishart_vector_format((n,1), batch_shape = batch_shape,
                                                       fixed_precision=True)
        self.bias.invU = self.invU

    def to_event(self,n):
        if n == 0:
            return self
        for i in range(len(self.p_list)):           
            self.A[i].event_dim = self.A[i].event_dim + n
            self.A[i].batch_dim = self.A[i].batch_dim - n 
            self.A[i].event_shape = self.A[i].batch_shape[-n:] + self.A[i].event_shape
            self.A[i].batch_shape = self.A[i].batch_shape[:-n]
        self.invU.to_event(n)
        return self

    def raw_update(self, X_list, Y, p=None, iters = 1, lr=1.0, beta=None):
        sample_shape = Y.shape[:-self.event_dim-self.batch_dim]
        sample_dims = list(range(len(sample_shape)))

        if p is None:
            N = torch.prod(torch.tensor(sample_shape,requires_grad=False))
            N = N.expand(self.batch_shape + self.event_shape[:-2])            
        else:
            N = p.sum(sample_dims)
     
        ###  THe basic idea here is that we are have a SExx that is nearly block sparse
        ###  The violation of the block sparsity is due to  the bias term which puts an
        ###  additional row and column at the end of teh SExx matrix.  We can handle this 
        ###  by using the matrix inversion lemma to compute the effective V and mu without 
        ###  having to invert the full SExx matrix.  This is done in the following code.  
        ###  Taking advantage of the fact that for dim(D)=(1,1), M = [A B;B^T,D] implies
        ###  M^-1  =  [invA + 1/K invvA @ B @ B^T invvA, -1/K invA B ; -1/K B^T invA, 1/K]
        ###      where K = D - B^T invA B, and A is block diagonal
        Y_res = Y - self.bias.mean()
        for i in range(len(self.p_list)):
            Y_res = Y_res - self.A[i].mean()@X_list[i]

        for _ in range(iters):
            idx = torch.randperm(len(self.p_list))
            for i in idx:
                Y_res = Y_res + self.A[i].mean()@X_list[i]
                self.A[i].raw_update(X_list[i],Y_res,lr=lr,beta=beta)
                Y_res = Y_res - self.A[i].weights()@X_list[i]
            
            Y_res = Y_res + self.bias.mean()
            self.bias.raw_update(Y_res,lr=lr,beta=beta)
            Y_res = Y_res - self.bias.mean()
        SEyy = 0.0
        if p is None:
            SEyy = (Y_res*Y_res.transpose(-2,-1)).sum(sample_dims)  
        else:
            SEyy = ((Y_res*Y_res.transpose(-2,-1))*p.view(p.shape+(1,1))).sum(sample_dims)

        for i in range(len(self.p_list)):
            SEyy = SEyy + self.A[i].mu_0@self.A[i].invV_0@self.A[i].mu_0.transpose(-2,-1)
            SEyy = SEyy + self.bias.mu_0@self.bias.mu_0.transpose(-2,-1)*self.bias.lambda_mu_0.unsqueeze(-1).unsqueeze(-1)
            
        if self.noise_type == 'Wishart':
            self.invU.ss_update(SEyy,N,lr,beta)                
        elif self.noise_type == 'Gamma':
            self.invU.ss_update(SEyy.diagonal(dim1=-1,dim2=-2),N.unsqueeze(-1),lr,beta)

    def Elog_like(self,X_list,Y):
        sample_shape = Y.shape[:-self.event_dim-self.batch_dim]
        temp = Y - self.bias.mean()
        ELL = 0.5*self.ElogdetinvSigma() - 0.5*self.n*torch.log(2*torch.tensor(torch.pi,requires_grad=False)) - 0.5*self.bias.EXmMUTinvUXmMU()
        for i in range(len(self.p_list)):
            temp = temp - self.A[i].mean()@X_list[i]
            ELL = ELL - 0.5*(X_list[i].transpose(-2,-1)@self.A[i].EXmMUTinvUXmMU()@X_list[i]).squeeze(-1).squeeze(-1)
        ELL = ELL - 0.5*(temp.transpose(-2,-1)@self.EinvSigma()@temp).squeeze(-1).squeeze(-1)
        return ELL

    def Elog_like_pX_pY(self,pX_list,pY):
        sample_shape = pY.mean.shape[:-self.event_dim-self.batch_dim]
        ELL = 0.5*self.ElogdetinvSigma() - 0.5*self.n*torch.log(2*torch.tensor(torch.pi,requires_grad=False)) - 0.5*self.bias.EXmMUTinvUXmMU()
        temp = pY.mean() - self.bias.mean()
        for i in range(len(self.p_list)):
            temp = temp - self.A[i].mean()@X_list[i]
            ELL = ELL - 0.5*(pX_list[i].mean().transpose(-2,-1)@self.A[i].EXmMUTinvUXmMU()@pX_list[i].mean()).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*pX_list[i].EXXT*self.A[i].EXTinvUX()
        ELL = ELL - 0.5*(temp.transpose(-2,-1)@self.EinvSigma()@temp).squeeze(-1).squeeze(-1)
        return ELL


    def predict(self,X_list):
        mu_y = self.bias.mean()
        Res = -0.5*self.bias.EXmMUTinvUXmMU()
        for i in range(len(self.p_list)):
            mu_y = mu_y + self.A[i].mean()@X_list[i]
            Res = Res - 0.5*(X_list[i].transpose(-2,-1)@self.A[i].EXmMUTinvUXmMU()@X_list[i])
        Res = Res.squeeze(-1).squeeze(-1)
        return MultivariateNormal_vector_format(invSigma = self.EinvSigma(), Sigma = self.ESigma(), invSigmamu = self.EinvSigma()@mu_y, mu = mu_y), Res

    def forward(self,pX_list):
        # This approximate forward routine assumes that y = bias + y_0 + y_1 + ...
        # where y_i = A_i@x_i + noise/sqrt(len(p_list)).  This assumption of convience 
        # means that we can use the predictions of the A_i's to compute the prediction
        # for y.  
        self.invU.nu = self.invU.nu/len(self.p_list)
        pY_list = [None]*len(self.p_list)
        for i in range(len(self.p_list)):
            pY_list[i], Res_i = self.A[i].forward(pX_list[i])
            Res = Res + Res_i

        self.invU.nu = self.invU.nu*len(self.p_list)
        mu_y = self.bias.mean()
        Sigma_y = 0.0
        for i in range(len(self.p_list)):
            mu_y = mu_y + pY_list[i].mean()
            Sigma_y = Sigma_y + pY_list[i].ESigma()

        return MultivariateNormal_vector_format(mu=mu_y,Sigma=Sigma_y), Res

    def postdict(self,Y,iters=10):
        assert iters > 1, 'MultiLinearNormalWishart postdict: iters must be greater than 1'
        Y_res = Y - self.bias.mean()    

        idx = torch.randperm(len(self.p_list))
        pX_list =[None]*len(self.p_list)
        for i in idx:
            pX_list[i] = self.A[i].postdict(Y_res)[0]
            Y_res = Y_res - self.A[i].mean()@pX_list[i].mean()

        for i in range(iters-1):
            Res = 0.0
            for i in idx:
                Y_res = Y_res + self.A[i].mean()@pX_list[i].mean()
                pX_list[i], Res_i = self.A[i].postdict(Y_res)
                Y_res = Y_res - self.A[i].mean()@pX_list[i].mean()
                Res = Res + Res_i

        return pX_list

    def backward(self,pY,Res):
        raise NotImplementedError

    def update(self,pX,pY,p=None,lr=1.0,beta=None):
        raise NotImplementedError

    def KLqprior(self):
        KL = -self.invU.KLqprior()*(len(self.p_list)-1)
        for i in range(len(self.p_list)):
            KL = KL + self.A[i].KLqprior()
        return KL

    def Elog_like_given_pX_pY(self,pX,pY):  
        raise NotImplementedError

    def Elog_like_X(self,Y):
        raise NotImplementedError
    
    def Elog_like_X_given_pY(self,pY):
        raise NotImplementedError

    def Ebackward(self,pY):
        raise NotImplementedError

    def predict_given_pX(self,pX):
        return self.forward(pX)

    def ElogdetinvSigma(self):
        return self.invU.ElogdetinvSigma()
    
    def EinvSigma(self):
        return self.invU.EinvSigma()

    def ESigma(self):
        return self.invU.ESigma()




