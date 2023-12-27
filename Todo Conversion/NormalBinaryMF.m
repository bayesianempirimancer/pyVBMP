classdef NormalBinaryMF < handle

    properties
        % Parameters
        D % number of observations
        N % number of binary latents
        
        L % lower bound 
        
        p % NS by N matrix of latent cause probabilities
        pij % NS x NS x 1
        z
        
        W % MVN
        zp % gamma scale dist
        gamma % noise on Y's assumed to be the same....
        pi % Beta distribution on prior cause probability (N vector)
        iters      
        scale
        baseline
        alpha_0
        pisparse
    end
    
    methods
        function self = NormalBinaryMF(D,N,alpha_0,scale,pisparse,baseline)
            self.D = D;
            self.N = N;
            self.L = -Inf;
            self.alpha_0=alpha_0;
            self.scale=scale;
            if(~exist('baseline','var'))
                self.baseline=1;
            else
                self.baseline=baseline;
            end
%            if(self.scale==0)
                self.z.mean=1;
                self.z.secondmoment=1;
                self.z.logZ=0;
                self.z.meaninv=1;
                self.z.entropy=0;
                self.z.loggeomean=0;
                self.zp.mean=1;
                self.zp.secondmoment=1;
                self.zp.KLqprior=0;
%            end
            self.zp = dists.gammagamma1(2,1);
            self.pisparse=pisparse;
            if(pisparse)
                self.pi = dists.expfam.betadist(ones(N,1)*alpha_0,ones(N,1));
            else
                self.pi = dists.expfam.betadist(alpha_0,1);             
            end
            for i=1:self.D
                self.W{i}=dists.expfam.MVN(zeros(N,1),eye(N)*N);
            end
            self.gamma=dists.expfam.gamma(1,1);
            self.iters=0;
            
        end
                        
        function [DL,VarExp] = update(self,Y,iters)
            ns=size(Y,2);
            for i=1:iters
                self.iters=self.iters+1;
                L=self.L;
                self.updateLatents(Y,3);                    
                
                VarExp = self.updateParms(Y);          
                DL = self.L - L;
                if(DL<0)
                    fprintf(['Warning Lower Bound Decreasing (DL/L = ',num2str(DL/abs(self.L)),') at iteration ',num2str(self.iters),'\n'])
                end
            end
        end
        
        function updateLatents(self,Y,iters)
            if(size(self.p,2)~=size(Y,2))
                self.p=rand(self.N,size(Y,2));
                if(self.scale)
                    a=repmat(2*self.zp.alphamean,1,size(Y,2));
                     self.z=dists.GIG(a,zeros(size(a)), ...
                         repmat((self.zp.alphamean),1,size(Y,2)));
                end
            end
%            if(self.iters==0)
%                 [V,D]=eig(cov(Y'),'vector');
%                 W=2*V*diag(sqrt(D))/self.N;  
% 
%                 for i=1:min(self.N,self.D)
%                     self.W{i}.mu = W(end-i+1,:);
%                 end
%                 fprintf('Initializing...\n')
%            end
            
%            self.p=rand(self.N,size(Y,2));

            
            J=self.EJ()*self.gamma.mean;
            Wmat=self.EWmat*self.gamma.mean;
            h=bsxfun(@plus,Wmat'*Y,self.pi.loglike);

            diagJ = diag(J);
            h=h+diagJ*self.z.mean;
            J=J-diag(diagJ);    
            for i=1:iters
                idx=randperm(self.N);
                for j=1:self.N
                    if(idx(j)==1 & self.baseline)
                        self.p(idx(j),:)=1;
                    else
                        hstar=h(idx(j),:)+(J(:,idx(j))'+J(idx(j),:))*bsxfun(@times,self.p,self.z.mean);
                        self.p(idx(j),:)=1./(1+exp(-hstar));
                    end
                end
            end
            if(self.scale & self.iters>2)  %HERE
                WTW = self.EWTW();
                self.z.a = self.gamma.mean*(sum((WTW*self.p).*self.p,1) + diag(WTW)'*(self.p-self.p.^2)) ...
                             + 2*self.zp.alphamean;
                self.z.b = self.gamma.mean*sum(Y.*Y,1);
                self.z.p = (self.zp.alphamean-self.D/2)*ones(size(self.z.a));
            end
        end

        function VarExp = updateParms(self,Y)
            if(isempty(self.p))
                self.updateLatents(Y,4);
            end
            Ns=size(Y,2);
            
            %Compute Sufficient statistics
            hit=0;
            for i=self.baseline+1:self.N
                if(mean(self.p(i,:))>0.5)
                    for j=1:self.D
                        self.W{j}.mu(i) = -self.W{j}.mu(i);
                    end
                    hit=hit+1;
                end                
            end
            if(hit>0)
                self.updateLatents(Y,4);                
                'flipping polarity'
            end
            
            NA=sum(self.p,2);
            SEXX = bsxfun(@times,self.p,self.z.mean)*self.p';
            SEXX = SEXX + diag(sum(bsxfun(@times,self.p-self.p.^2,self.z.mean),2));
            SEYX = Y*self.p';
            SEYY = sum(Y.*Y,1)*self.z.meaninv';
            
            SERR=sum(SEYY)-2*sum(sum(SEYX.*self.EWmat)) + sum(sum(SEXX.*(self.EWTW)));
            
            idx=self.p>0 & self.p<1; 
            self.L = - sum(self.p(idx).*log(self.p(idx))) - sum((1-self.p(idx)).*log(1-self.p(idx))) ...
                     + sum(sum(self.pi.loggeomean()'*self.p)) + sum(sum(self.pi.loggeomeanmirror()'*(1-self.p)));

            
            if(self.scale)
                self.L = self.L + self.gamma.mean*sum(sum(Y.*(self.EWmat*self.p))) + self.D*Ns/2*(self.gamma.loggeomean-log(2*pi));            
                self.L = self.L + sum(self.z.logZ) - sum(self.z.logZp);
            else
                %STUFF HERE
                self.L = self.L - 1/2*self.gamma.mean*SERR; 
            end
            
            self.L = self.L - self.KLqprior;

            
            VarExp=1-mean(mean((Y-bsxfun(@times,self.EWmat*self.p,self.z.mean)).^2))/var(Y(:))
%            VarExp=1-SERR/Ns/sum(var(Y'));
            
            if(self.scale)
                self.zp.updateSS(mean(self.z.mean),mean(self.z.loggeomean),Ns);                
            end
            if(self.iters>20)
            if(self.pisparse)
                self.pi.updateSS(NA/Ns,Ns);
                if(self.baseline)
                    self.pi.alpha(1)=self.pi.alpha_0(1);
                    self.pi.beta(1)=self.pi.beta_0(1);
                end      
            else
                if(self.baseline)
                    self.pi.updateSS(sum(NA(2:end))/Ns/(self.N-1),Ns*(self.N-1));
                else
                    self.pi.updateSS(sum(NA)/Ns/(self.N),Ns*(self.N));                
                end          
            end
            end
            
            self.gamma.updateSS(SERR/2/(Ns*self.D),1/2,Ns*self.D);            
            for i=1:self.D
                self.W{i}.updateSS(self.gamma.mean*SEYX(i,:)'/Ns,self.gamma.mean*SEXX/Ns,Ns);
            end
            

        end
        
        function fillunused(self,Y,tol)
            if(~exist('tol','var'))
                tol=1;
            end
            ERR=sum((bsxfun(@times,self.EWmat*self.p,self.z.mean)-Y).^2,1);
            idx=find(sum(self.p,2)<tol);
            
            [m,ERRidx]=sort(ERR,'descend');
            
            idxi=randi(1000,1,length(idx));
            
            for j=1:self.D
            for i=1:length(idx)
                self.W{j}.mu(idx(i))=Y(j,ERRidx(idxi(i)));
            end
                self.W{j}.invSigma=eye(self.N);
                self.W{j}.Sigma=eye(self.N);
                self.W{j}.invSigmamu=self.W{j}.mu;
                self.W{j}.setUpdated(false);                
            end
            ['filling ', num2str(length(idx)),' unused features']
            
        end
 
        function J = EJ(self)
            J=-1/2*self.EWTW;
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

        
        function KL = KLqprior(self)            

            KL = self.pi.KLqprior + self.gamma.KLqprior + self.zp.KLqprior;
            for i=1:self.D
                KL = KL + self.W{i}.KLqprior;
            end                    
        end
        
        
        function idx = RFplotter(self,nfigs,V,D)
            if(~exist('D','var'))
                EWmat=self.EWmat;
            else
                EWmat=V*diag(sqrt(D))*self.EWmat;                
            end
            [m,idx]=sort(sum(self.p,2),'descend');
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
        
       
    end
    
end

