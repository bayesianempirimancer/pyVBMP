classdef sLDA < handle

    properties
        % Parameters
        D % number of observations
        N % number of mixture components
        
        L % lower bound 
                
        W % dirichlet
        zp % prior on scale parameter
        
        z % scale parameter (1xNs)
        c % concentration posterior (NxNs) dirichlet

        alpha_0 %sparsity parameter
        iters      
        
        %latent x's
        p % multinomial distribution on latents
        logptilde % unnormalized log probability        
        
        baseline
        scale
        gammac
        
        SENij
   end
    
    methods
        function self = sLDA(D,N,alpha_0,scale,gammac)
            self.D = D;
            self.N = N;
            self.L = -Inf;
            self.scale=scale;
            if(~exist('gammac','var'))
                self.gammac=0;
            else
                self.gammac=gammac;
            end
            self.alpha_0=alpha_0;
            
            for i=1:self.N
                self.W{i}=dists.expfam.dirichlet(D,repmat(1/2,D,1));
            end
            self.zp=dists.gammagamma(1,1,1,1);
            self.iters=0;            
            if(~scale)
                self.z.loggeomean=0;
                self.z.mean=0;
                self.z.logZ=0
            end
        end
                        
        function [DL,VarExp] = update(self,Y,iters)
            for i=1:iters
                self.iters=self.iters+1;
                L=self.L;
                self.updateLatents(Y);                    
                
                self.updateParms(Y);          
                DL = self.L - L;
                if(DL<0)
                    fprintf(['Warning Lower Bound Decreasing (DL/L = ',num2str(DL/abs(self.L)),') at iteration ',num2str(self.iters),'\n'])
                end
            end
        end
        
        function updateLatents(self,Y,iters)
            if(~exist('iters','var'))
                iters=1;
            end
            Ns=size(Y,2);
            if(size(self.p,3)~=Ns)
                self.logptilde=NaN(self.D,self.N,Ns);
                if(self.gammac)
                    self.c=dists.expfam.gamma(self.alpha_0*ones(self.N,Ns),ones(self.N,Ns));
                else
                    self.c=dists.expfam.dirichletcols(self.N,self.alpha_0/self.N);
                    self.c.updateSS(rand(self.N,Ns));
                end
            end
            
            if(self.scale)
                self.z=dists.expfam.gamma(repmat(self.zp.alphamean,1,Ns),...
                                  repmat(self.zp.alphamean.*self.zp.betamean,1,Ns));
                if(self.gammac)
                    self.z.updateSS(sum(self.c.mean,1),sum(Y,1),1);
                else
                    self.z.updateSS(1,sum(Y,1),1);
                end
            end

            logW=self.ElogWmat;
            lnz=self.z.loggeomean;
            
            
            for n=1:iters
                for i=1:self.D
                    self.logptilde(i,:,:) = bsxfun(@plus,logW(i,:)',self.c.loggeomean);
                end
                self.p=bsxfun(@plus,self.logptilde,-max(self.logptilde,[],2));
                self.p=exp(self.p);
                self.p=bsxfun(@times,self.p,1./sum(self.p,2));

                ENijt=bsxfun(@times,self.p,reshape(Y,self.D,1,Ns));
                
                idx=self.p>0 & self.p<1;
                self.L = sum(self.logptilde(:).*ENijt(:)) - sum(ENijt(idx).*log(self.p(idx)))  ...
                         - sum(gammaln(Y(:)+1)) - Ns*(self.zp.ElogZ) + sum(self.z.logZ) ...
                         - (self.c.logZp) + (self.c.logZ) - self.KLqprior;

                self.SENij=squeeze(sum(ENijt,3));
                
                if(self.gammac)
                    self.c.updateSS(repmat(self.z.mean,self.N,1),squeeze(sum(ENijt)),1)
                else
                    self.c.updateSS(squeeze(sum(ENijt)));
                end 
            end
            
            
        end

        function updateParms(self,Y)
            if(size(self.p,2)~=size(Y,2))
                self.updateLatents(Y,1);
            end
            Ns=size(Y,2);
            
            if(self.scale)
                self.zp.updateSS(mean(self.z.mean),mean(self.z.loggeomean),Ns);  
            end
            for i=1:self.N
                self.W{i}.updateSS(self.SENij(:,i));
            end        
        end
         
        
        function KL = KLqprior(self)            
            KL = self.zp.KLqprior;
            for i=1:self.N
                KL = KL + self.W{i}.KLqprior;
            end
        end
        
        function res = ElogWmat(self)
            res=NaN(self.D,self.N);
            for i=1:self.N
                res(:,i) = self.W{i}.loggeomean;
            end
        end
               
        function res = EWmat(self)
            res=NaN(self.D,self.N);
            for i=1:self.N
                res(:,i) = self.W{i}.mean;
            end
        end
    end
    
end

