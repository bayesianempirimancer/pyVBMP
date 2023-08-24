classdef GSM < handle

    properties
        % Parameters
        D % number of observations
        N % number of binary latents
        
        L % lower bound 
        
        z % GIG for scale latents
        
        W % MVN
        zp % gamma scale dist (NOT UPDATED)
        gamma % noise on Y's assumed to be the same....
        
        alpha % implementation of sparsity.  
        beta % 1./mean sparsity.;
        
        alpha_0 % generally fixed, but can be a gamma
        beta_0 % generally fixed, but can be a gamma

        iters      
        
        SEXX
        SEYX
        SEYY
        SERR
        %latent x's
        mu
        EXX
        TRWTWXXT
                
        scale
        sparse % set to one of W is sparse too.
   end
    
    methods
        function self = GSM(D,N,scale,alpha_0,beta_0,sparse)
            self.D = D;
            self.N = N;
            self.L = -Inf;
            self.scale=scale;
            if(~exist('sparse','var'))
                self.sparse=0;
            else
                self.sparse=sparse;
            end
            self.alpha_0=alpha_0;
            self.beta_0=beta_0;
            self.z.mean=1;
            self.z.secondmoment=1;
            self.z.logZ=0;
            self.z.meaninv=1;
            self.z.entropy=0;
            self.z.loggeomean=0;
            self.zp = dists.gammagamma1(2,1);
            
            for i=1:self.D
                if(self.sparse)
                    self.W{i}=dists.normalsparse(N,1,1);
                else
                    self.W{i}=dists.expfam.MVN(zeros(N,1),D*eye(N));
                end
            end
            self.gamma=dists.expfam.gamma(1,1);
            self.iters=0;            
        end
                        
        function [DL,VarExp] = update(self,Y,iters)
            for i=1:iters
                self.iters=self.iters+1;
                L=self.L;
                self.updateLatents(Y,2);                    
                
                VarExp = self.updateParms(Y);          
                DL = self.L - L;
                if(DL<0)
                    fprintf(['Warning Lower Bound Decreasing (DL/L = ',num2str(DL/abs(self.L)),') at iteration ',num2str(self.iters),'\n'])
                end
%                DL=DL/abs(L);
            end
        end
        
        function updateLatents(self,Y,iters)
            Ns=size(Y,2);
            if(size(self.mu,2)~=Ns)
%                self.alpha = dists.expfam.gamma(self.alpha_0*ones(self.N,Ns),self.alpha_0*ones(self.N,Ns));
                self.alpha = dists.expfam.gamma(repmat(self.alpha_0,1,Ns),repmat(self.beta_0,1,Ns));
                self.EXX = NaN(self.N,Ns);
                self.mu = NaN(self.N,Ns);
                if(self.scale)
                    self.TRWTWXXT = ones(1,Ns);
                    a=repmat(2*self.zp.alphamean,1,size(Y,2));
                     self.z=dists.GIG(a,zeros(size(a)), ...
                         repmat((self.zp.alphamean),1,size(Y,2)));
                end
            end
            WTW=self.EWTW*self.gamma.mean;
            Wmat=self.EWmat*self.gamma.mean;
            
            for i=1:iters
                alpha=self.alpha.mean;
                invSigmamu = Wmat'*Y;

                if(self.scale & iters>2)
                    zhat=self.z.mean;
                else
                    zhat=repmat(1,1,Ns);
                end

                self.EXX=NaN(self.N,Ns);
                self.SEXX=zeros(self.N);
                for n=1:Ns
                    invSigma=WTW*zhat(n)+diag(alpha(:,n))*self.gamma.mean;
                    [Xentropy(n),R]=util.logdet(invSigma);
                    Xentropy(n) = self.N/2*log(2*pi*exp(1)) - 1/2*Xentropy(n);
                    R=inv(R);
                    Sigma=R*R';
                    
                    mu=Sigma*invSigmamu(:,n);
                    XXT=Sigma+mu*mu';
                    self.TRWTWXXT(1,n)=sum(WTW(:).*XXT(:));
                    self.EXX(:,n)=diag(XXT);
    %                 self.invSigma(:,:,n) = WTW*zhat(n)+diag(alpha(:,n));
    %                 self.Sigma(:,:,n)=inv(self.invSigma(:,:,n));
    %                 self.mu(:,n)=self.Sigma(:,:,n)*invSigmamu(:,n);
    %                 self.XXT(:,:,n) = self.Sigma(:,:,n)+self.mu(:,n)*self.mu(:,n)';
    %                 TRWTWXXT(1,n) = sum(sum(WTW.*self.XXT(:,:,n)));
    %                 EXX(:,n)=diag(self.XXT(:,:,n));

%                    self.XXT(:,:,n)=XXT;
%                    self.SEXX = self.SEXX + XXT*zhat(n);

                    self.mu(:,n)=mu;                    
                    
                    if(self.scale)
                        za =(self.TRWTWXXT(1,n)) + 2*self.zp.alphamean;
                        zb = self.gamma.mean*(Y(:,n)'*Y(:,n));
                        zp = (self.zp.alphamean-self.D/2);
                        ab=za*zb;
                        binva=zb/za;            
                        ztemp = sqrt(binva)*besselk(zp+1,sqrt(ab),1)/besselk(zp,sqrt(ab),1);
                        self.SEXX = self.SEXX + XXT*ztemp;
                    end
                end
                
                if(self.scale)  
                    self.z.a = (self.TRWTWXXT) ...
                                 + 2*self.zp.alphamean;
                    self.z.b = self.gamma.mean*sum(Y.*Y,1);
                    self.z.p = (self.zp.alphamean-self.D/2)*ones(size(self.z.a));
                end               
%                self.alpha.update(1/2*self.EXX,1/2*ones(size(EXX)),ones(size(EXX)));
                self.alpha.update(1/2*self.EXX*self.gamma.mean,1/2*ones(size(self.EXX)),ones(size(self.EXX)));
            end                                    
            
            

%            self.SEXX = squeeze(sum(bsxfun(@times,self.XXT,zhat),3));
            self.SEYX = Y*self.mu';
            self.SEYY = bsxfun(@times,Y,self.z.meaninv)*Y';
            self.SERR = sum(diag(self.SEYY))-2*sum(sum(self.SEYX.*self.EWmat)) + sum(sum(self.SEXX.*(self.EWTW)));            
            
%             self.L = - 1/2*self.gamma.mean*self.SERR + self.D*Ns/2*(self.gamma.loggeomean-log(2*pi)); 
%             self.L = self.L - 1/2*sum(sum(EXX.*self.alpha.mean)) + 1/2*sum(sum(self.alpha.loggeomean)) - Ns*self.N/2*log(2*pi);
% 
%             %NOW ASSUMES ZP NOT UPDATED
%             self.L = self.L + (self.D/2 + 1 - self.zp.alphamean)*sum(self.z.loggeomean) ...
%                                - self.zp.alphamean*sum(self.z.mean) ...
%                                + Ns*(self.zp.alphamean*log(self.zp.alphamean) - gammaln(self.zp.alphamean));
% 
%             self.L = self.L - self.KLqprior + sum(sum(self.alpha.entropy)) ... 
%                             + sum(Xentropy) + sum(self.z.entropy);

%%%NEEDS FIX BELOW for alpha to gamma*alpha. and needs 1/2log(gamma)
%%%terms...

            self.L = sum(sum(self.SEYX.*Wmat)) - (self.D+self.N)*Ns/2*log(2*pi) + Ns*(self.D+self.N)/2*self.gamma.loggeomean;
            self.L = self.L + Ns*self.D*(self.zp.alphamean*log(self.zp.alphamean) - gammaln(self.zp.alphamean));
            self.L = self.L - self.KLqprior + sum(Xentropy) + sum(self.z.logZ) + sum(sum(self.alpha.logZ));
        end

        function VarExp = updateParms(self,Y)
            if(isempty(self.mu))
                self.updateLatents(Y,1);
            end
            Ns=size(Y,2);
            
            
            VarExp=1-sum(sum((Y-bsxfun(@times,self.EWmat*self.mu,self.z.mean)).^2))/sum(var(Y'))/Ns
%            self.gamma.mean
%             if(self.scale)
%                 self.zp.updateSS(mean(self.z.mean),mean(self.z.loggeomean),Ns);  
%                 self.zp.alphamean
%             end
            
            for i=1:self.D
                self.W{i}.updateSS(self.gamma.mean*self.SEYX(i,:)'/Ns,self.gamma.mean*self.SEXX/Ns,Ns);
            end

            self.SERR = sum(diag(self.SEYY))-2*sum(sum(self.SEYX.*self.EWmat)) + sum(sum(self.SEXX.*(self.EWTW)));
           if(mod(self.iters,4)==0)
%               self.gamma.updateSS(self.SERR/(2*Ns*self.D),1/2,Ns*self.D);            
               self.gamma.updateSS((self.SERR+sum(sum(self.alpha.mean.*self.EXX)))/(2*Ns*(self.N+self.D)),1/2,Ns*(self.N+self.D));
           end
        end
         
        
        function fillunused(self,Y,pc)
            
            W=self.EWmat;
            stdW=std(W(:));
            Yhat=bsxfun(@times,W*self.mu,self.z.mean);
            [m,idx]=sort(mean((Y-Yhat).^2),'descend');
            
            [m,idxW]=sort(diag(W'*W),'ascend');
            
            for i=1:self.D
            for j=1:ceil(self.N*pc)
                self.W{i}.mu(idxW(j))=Y(i,idx(j))*stdW;
            end
            end
            
        end
        
        function res = EWTW(self)
            res = zeros(self.N,self.N);
            for i=1:self.D
                res = res + self.W{i}.secondmoment;
            end
        end
        
        function Wmat = EWmat(self)
            for i=1:self.D
                Wmat(i,:)=self.W{i}.mu';
            end
            Wmat=Wmat;
        end

        function RFplotter(self,nfigs,V,D)
            if(~exist('D','var'))
                EWmat=self.EWmat;
            else
                EWmat=V*diag(sqrt(D))*self.EWmat;                
            end
            [m,idx]=sort(var(self.mu'),'descend');
            temp=var(EWmat);
            idx2=temp(idx)<0.05;
            nfigs=min(sum(idx2),nfigs);
            idx=idx(idx2);
            
%            [m,idx]=sort(kurtosis(self.mu'),'descend');
%            [m,idx]=sort(var(EWmat),'ascend');
%            [m,idx]=sort(var(EWmat).*var(self.mu'),'ascend');
            if(nfigs>=self.N)
                nfigs=self.N;
            end
            figure
            nfigs=floor(sqrt(nfigs));
            for i=1:nfigs^2
                subplot(nfigs,nfigs,i), imagesc(reshape(EWmat(:,idx(i)),sqrt(self.D),sqrt(self.D)))
                axis off;
            end
            figure
            cc=corr(EWmat);
            imagesc(cc-diag(diag(cc))), colorbar
            
            Wf=zeros(sqrt(self.D));
            
            for i=1:self.N
                Wf=Wf+abs(fft2(reshape(EWmat(:,i),sqrt(self.D),sqrt(self.D)))).^2;
            end
            Wf=Wf/self.N;
            Wf=(Wf+Wf')/2;
            figure
            Wfmean=mean(Wf);
            plot(Wfmean);
        end
        
        function KL = KLqprior(self)            

            KL = self.gamma.KLqprior + self.zp.KLqprior;
            for i=1:self.D
                KL = KL + self.W{i}.KLqprior;
            end                    
        end
       
    end
    
end

