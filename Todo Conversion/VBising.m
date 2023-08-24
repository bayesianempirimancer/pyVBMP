classdef VBising < handle
    
    properties
        
        D % dimensionality of model 
        NR % dimensionality of inputs
        issparse
        
        LR
        L
        iters
    end
    
    methods
        function self = VBising(D,NR,issparse)
            self.D=D;
            self.NR=NR;
            self.issparse=issparse;
            iters=0;
            L=-Inf;
            for j=1:D
                self.LR{j}=VBLR(NR+D,issparse);
            end        
        end
        
        function DL = fit(self,X,U,maxiters,tol)
            if(~exist('maxiters','var'))
                maxiters=4*(self.D+self.NR);
            end
            if(~exist('tol','var'))
                tol=(1e-13)*size(X,1);
            end
            
            for j=1:self.D
                Xtemp=[X,U];
                Xtemp(:,j)=ones(size(X,1),1);
                self.LR{j}.fit(X(:,j),Xtemp,maxiters,tol);
                L(j)=self.LR{j}.L;
            end
            self.L=sum(L);
            DL=self.update(X,U,1);
        end

        function DL = update(self,X,U,iters)
            if(~exist('iters','var'))
                iters=1;
            end
            for j=1:self.D
                Xtemp=[X,U];
                Xtemp(:,j)=ones(size(X,1),1);
                for i=1:iters
%                     idx=true(1,self.D);
%                     idx(j)=false;
                   DL(j)=self.LR{j}.update(X(:,j),Xtemp);
                   L(j)=self.LR{j}.L;
                end
            end
            DL=sum(DL);
            self.L=sum(L);
        end
        
        function [J,h] = getJ(self)
            for j=1:self.D
                J(j,:)=self.LR{j}.beta.mean()';
            end
            J=J(:,1:self.D);
            h=diag(J);
            J=J-diag(h);
            J=(J+J')/4;
        end
        
        function w = getW(self)
            if(self.NR>0)
                for j=1:self.D
                    w(j,:)=self.LR{j}.beta.mean';
                end
                w=w(:,self.D+1:end);
            else
                w=[];
            end
        end
        
    end
    
end

