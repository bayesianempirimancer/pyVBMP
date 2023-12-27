classdef poissonMVN < handle
    %poissonMVN.m Updates Poisson Likelihood-Latent Gaussian model using
    %Khan's dual variational trick. Potentially confusing part: n (number of
    %observations) in Khan's paper is #neurons x #TimeBins in our case. If
    %number of trials is accounted into, i.e. Khan's "n" is #neurons x
    %#TimeBins x #trials and Khan's "L" is #neurons x #TimeBins, then Poisson layer
    % would be more meaningful.
    %   Uses MVN.m code as a wrapper function.
    
    properties
        dim % dim of poisson and mu
        eta
        lambda
        % lower bound
        L
        
    end
    
    methods
        function self = poissonMVN(mu_0, invSigma_0, mu, invSigma)
            if ~exist('mu','var') || ~exist('i++++++++-nvSigma', 'var')
                self.eta = dists.expfam.MVN(mu_0,invSigma_0);
                self.lambda = exp(mu_0);
            else
                self.eta = dists.expfam.MVN(mu_0,invSigma_0,mu,invSigma);
                self.lambda=exp(mu);
            end
            self.dim = size(mu_0,1);
            self.L = -Inf;
        end
        
        function set_prior(mu_0,invSigma_0)
            self.eta.mu_0 = mu_0;
            self.eta.invSigma_0 = invSigma_0;
            self.eta.invSigmamu_0 = invSigma_0*mu_0;
            self.eta.Sigma_0;
        end
                
        function dualVB(self, y, N, eps)
            if(~exist('eps','var'))
               eps=0.9999; 
            end
            A = self.eta.invSigma_0 + N*diag(self.lambda);
%            F/N = N/2*(lambda- y)'*Sigma*(lambda- y) - mu'*(lambda- y) -...
%                1/N*0.5*self.logdet(A) + sum(lambda.*(log(lambda) - 1));
            invA =  inv(A); 
            dFoverN = N*self.eta.Sigma_0*(self.lambda-y) - self.eta.mu_0 - 0.5*diag(invA) + log(self.lambda);
            d2FoverN = N*self.eta.Sigma_0 + diag(1./self.lambda)+ N*0.5*invA.^2;
            self.lambda = self.lambda - eps * d2FoverN\dFoverN;             
            tol = 0.0001;
            self.lambda(self.lambda<0) = tol;
            
        end

        function res = F(self,y,N)
            A   = self.eta.invSigma_0 + N*diag(self.lambda);
            res = N/2*(self.lambda- y)'*self.eta.Sigma_0*(self.lambda- y) ...
                - self.eta.mu_0'*(self.lambda- y) ...
% possible sign error in the line below.
                - 0.5/N*self.logdet(A) ...
                + sum(self.lambda.*(log(self.lambda) - 1));
        end
        
        function updateSS(self,Ex,N,niter,eps)
            if(N==0)
                self.eta.mu = self.eta.mu_0;
                self.eta.invSigma = self.eta.invSigma_0;
                self.eta.Sigma =  inv(self.eta.invSigma);
                self.eta.invSigmamu = self.eta.invSigma*self.eta.mu;            
                return 
            end
            if(~exist('niter','var'))
                niter = 5;
            end
            if(~exist('eps','var'))
                eps=0.9999;
            end
            for i=1:niter
                self.dualVB(Ex,N,eps);
            end
            self.eta.mu = self.eta.mu_0 - N*self.eta.Sigma_0*(self.lambda - Ex);
            self.eta.invSigma = self.eta.invSigma_0 + N*diag(self.lambda);
            self.eta.Sigma =  inv(self.eta.invSigma);
            self.eta.invSigmamu = self.eta.invSigma*self.eta.mu;            
        end
                
        function L = rawupdate(self,y,p,niter,eps)  % assumes y is a standard TxN data matrix
            if(nargin < 3 || isempty(niter))
                p=ones(1,size(y,1));
            end
            if(nargin < 4 || isempty(eps))
                niter = 5;
            end
            if(nargin < 5)
               eps=0.9999;
            end
            N=sum(p);
            if(N==0)
                self.updateSS([],0);
            else
                self.updateSS((p*y)'/N,N,niter,eps);
            end
            L = p*self.Eloglikelihood(y) - self.KLqprior;
            self.L = L;
        end
        
        function res = mean(self)
            res = exp(self.eta.mu+0.5*diag(self.eta.Sigma));
        end
        
        function res = secondmoment(self)
            res = (self.mean*self.mean').*(exp(self.eta.Sigma)-1);
        end
        
        function res = var(self)
            res = self.secondmement - self.mean.^2 
        end
        
        function res = KLqprior(self)
            res = self.eta.KLqprior;
        end
        
        function res = Eloglikelihood(self,y)  % assumes y is standard data matrix TxN            
            res = y*self.eta.mu - sum(exp(self.eta.mu+0.5*diag(self.eta.Sigma))) - sum(gamma(y+1),2);
        end
                
        function res = logdet(A)  
            %avoid calling unless necessary, this shit ain't cheap.
                res = chol(A);
                res = 2*sum(log(diag(res)));
        end
        
        function [dExcoeff,dExxcoeff] = effloglikelihood(self)
            dExcoeff = self.eta.invSigmamu - self.eta.invSigmamu_0;
            dExxcoeff = self.eta.invSigma - self.eta.invSigma_0;            
            %             mu_0, invSigma_0, mu, invSigma
        end
    end
    
end

