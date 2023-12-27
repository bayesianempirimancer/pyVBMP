classdef GIG < handle
    properties
        % GIG(a,b,p)
        a_0
        b_0
        p_0

        a
        b
        p
        
        isUpdated
        
    end
    
    methods
        function self = GIG(a_0,b_0,p_0)
            self.a_0 = a_0;
            self.b_0 = b_0;
            self.p_0 = p_0;

            self.a = a_0;%.*(1+rand(size(a_0)));
            self.b = b_0;%.*(1+rand(size(a_0)));
            self.p = p_0;

        end
        
        function updateSS(self,Ex,Einvx,Elogx,n)  
            if(n>0)
                self.a = self.a_0 + Ex*n;
                self.b = self.b_0 + Einvx*n;
                self.p = self.p_0 + Elogx*n;
            else
                    self.a=self.a_0;
                    self.b=self.b_0;
                    self.p=self.p_0;
            end
        end
        
        function update(self,SEx,SEinvx,SElogx,n)  
                if(n>0)
                    self.a = self.a_0 + SEx;
                    self.b = self.b_0 + SEinvx;
                    self.p = self.p_0 + SElogx;
                else
                    self.a=self.a_0;
                    self.b=self.b_0;
                    self.p=self.p_0;
                end
        end
        
        
        function rawupdate(self,data,p)
            if(~exist('p','var'))
               p=ones(size(data,1),1);
            end
            idx=find(~isnan(sum(data,2)));
            n=sum(p(idx));
            SEx = p(idx)'*data(idx,:);
            SEinvx = p(idx)'*(1./data(idx,:));
            SElogx = p(idx)'*log(data(idx,:));
            self.updateSS(SEx',SEinvx',SElogx',n);
        end
        
        function res = mean(self,i)            
            if(~exist('i','var'))
                idx=self.b>0;
                ab=self.a.*self.b;
                binva=self.b./self.a;            
                res=sqrt(binva).*besselk(self.p+1,sqrt(ab),1)./besselk(self.p,sqrt(ab),1);
                res(~idx) = self.p(~idx)./self.a(~idx)*2;
            else
                if(self.b(i)>0)
                    ab=self.a(i)*self.b(i);
                    binva=self.b(i)/self.a(i);            
                    res=sqrt(binva)*besselk(self.p(i)+1,sqrt(ab),1)/besselk(self.p(i),sqrt(ab),1);
                else
                    res = self.p(i)/self.a(i)*2;
                end
            end
        end
        
        function res = secondmoment(self)
            mu = self.mean;
            idx=self.b>0;
            ab=self.a.*self.b;
            binva=self.b./self.a;
            res = besselk(self.p+2,sqrt(ab),1)./besselk(self.p,sqrt(ab),1).*binva;            
            res(~idx) = self.p(~idx)./self.a(~idx).^2*4 + mu(~idx).^2;
        end
        
        function res = var(self)
            res=self.secondmoment-self.mean.^2;
        end
    
        function res = meaninv(self)
            idx=self.b>0;
            ab=self.a.*self.b;
            ainvb=self.a./self.b;
            res = sqrt(ainvb).*besselk(-self.p+1,sqrt(ab),1)./besselk(-self.p,sqrt(ab),1);
            res(~idx) = self.a(~idx)/2./(self.p(~idx)-1);
            idx=self.b<0 & self.p<1;
            res(idx)=Inf;
        end
        
        function res = loggeomean(self)
            idx=self.b>0;
            ab=self.a.*self.b;
            binva=self.b./self.a;
            res = 0.5*log(binva) + self.logbesselkprime;
            res(~idx) = psi(self.p(~idx)) - log(self.a(~idx)/2);
        end
        
        function res = logbesselkprime(self)
            dnu=0.0001;
            res = (log(besselk(self.p+dnu,sqrt(self.a.*self.b),1))-log(besselk(self.p-dnu,sqrt(self.a.*self.b),1)))/dnu/2;
        end
        
        function res = logZ(self)
            idx=self.b>0;
            res = log(2) + log(besselk(self.p,sqrt(self.a.*self.b),1)) ...
                - sqrt(self.a.*self.b) - 1/2*self.p.*log(self.a./self.b);
            res(~idx) = gammaln(self.p(~idx)) - self.p(~idx).*log(self.a(~idx)/2);
        end

        function res = logZp(self)
            idx=self.b_0>0;
            res = log(2) + log(besselk(self.p_0,sqrt(self.a_0.*self.b_0),1)) ...
                - sqrt(self.a_0.*self.b_0) - 1/2*self.p_0.*log(self.a_0./self.b_0);
            res(~idx) = gammaln(self.p_0(~idx)) - self.p_0(~idx).*log(self.a_0(~idx)/2);            
        end

        function res = entropy(self)
            idx=self.b>0;
            ab=self.a.*self.b;
            res = 0.5*log(self.b./self.a) + log(2) + log(besselk(self.p,sqrt(ab),1))-sqrt(ab) ...
                - (self.p-1).*self.logbesselkprime + ...
                0.5*sqrt(ab)./besselk(self.p,sqrt(ab),1).*(besselk(self.p+1,sqrt(ab),1)+besselk(self.p-1,sqrt(ab),1));
            res(~idx) = self.p(~idx) - log(self.a(~idx)/2) ...
                      + gammaln(self.p(~idx)) + (1-self.p(~idx)).*psi(self.p(~idx));
        end
        
        function res = lnbesselk(p,z)
            res=log(besselk(p,z,1));
            idx=isinf(res);
            res(idx) = -1/2*log(2*pi*z(idx))+log(1-(4*p(idx).^2-1)/8./z(idx));
            sum(idx)
        end
        
    end
end




