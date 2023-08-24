classdef TPB < handle
    properties
        % model parameters
        mean % first moment
        firstInvMoment % first inverse moment
        logGeoMean
        nu
        
        dims 
        
        % hyperparameters
        a
        b
        phi
        omega
        
        Z
        b1
        b2
        b3
        b4
        b5
        GIGparams
        fixphi
    end
    
    methods
        function self = TPB(dims,phi,a,b,fixphi)
            if ~exist('phi','var'), phi = .01; end
            if ~exist('a','var'), a = 1/2; end
            if ~exist('b','var'), b = 1/2; end % a and b are horseshoe prior here
            if ~exist('fixphi','var'), fixphi = 1; end
            
            self.dims = dims;
            
            self.nu = dists.expfam.gammadist(b,phi,1/phi*ones(dims),ones(dims));
            
            self.phi = phi;
            %self.nu = 1/phi*ones(dims);
            self.mean = phi*ones(dims);
            self.firstInvMoment = 1/phi*ones(dims);
            self.logGeoMean = log(phi*ones(dims));
            
            self.omega = 1/self.phi;
            
            self.a = a;
            self.b = b;

            self.fixphi = fixphi;
            
            % fix values of linear interpolations for speed
            self.Z = logspace(-2.5,1.5,1e2);
            self.b1 = besselk(a + 1/2, self.Z);
            self.b2 = besselk(a - 1/2, self.Z);
            self.b3 = besselk(3/2 - a, self.Z);
            self.b4 = besselk(1/2 - a, self.Z);
            self.b5 = besselk(a - 3/2, self.Z);
        end
        
        function update(self,EX2)
            % update variance component (GIG)
            sqrta = (2*self.nu.mean).^(1/2);
            sqrtb = (EX2).^(1/2);
            sqrtab = sqrta.*sqrtb;
            self.GIGparams.sqrta = sqrta; self.GIGparams.sqrtb = sqrtb; 
            self.mean = sqrtb .* interp1(self.Z,self.b1,sqrtab,'nearest','extrap') ...
                ./ ( sqrta .* interp1(self.Z,self.b2,sqrtab,'nearest','extrap'));
            self.firstInvMoment = sqrta .* interp1(self.Z,self.b3,sqrtab,'nearest','extrap') ...
                ./ ( sqrtb .* interp1(self.Z,self.b4,sqrtab,'nearest','extrap'));
            self.logGeoMean = 2*log(sqrtb./sqrta);
            
            % update nu
            self.nu.update(2*self.a*ones(self.dims),2*self.mean);
            %self.nu = (self.a + self.b)./(self.phi + self.mean);

            %update phi
            % TODO. For now, keep phi fixed.
            if ~self.fixphi
                % update phi
                self.phi = (numel(self.nu)*self.b + 1/2)/(self.omega + sum(self.nu(:)));

                % update omega
                self.omega = 1/(self.phi + 1);
            end
        end
        
        function res = entropy(self)
            sqrta = self.GIGparams.sqrta; sqrtb = self.GIGparams.sqrtb;
            sqrtab = sqrta.*sqrtb;
            % note: this is only valid for a = 1/2!
            res = log(sqrtb) - log(sqrta) ...
                + log(2*interp1(self.Z,self.b2,sqrtab,'nearest','extrap')) ...
                + sqrtab./(2*interp1(self.Z,self.b2,sqrtab,'nearest','extrap')) ...
                .* (interp1(self.Z,self.b1,sqrtab,'nearest','extrap') ...
                + interp1(self.Z,self.b5,sqrtab,'nearest','extrap') );
            res = sum(res(:));
        end
        
        function res = expectlogjoint(self)
            GIGp = self.a - 1/2;
            GIGa = self.GIGparams.sqrta.^2;
            GIGb = self.GIGparams.sqrtb.^2;
            sqrtab = sqrt(GIGa.*GIGb);
            % note: this is only valid for a = 1/2!
            
            res = GIGp/2*log(GIGa./GIGb) ...
                - log(2*interp1(self.Z,self.b2,sqrtab,'nearest','extrap')) ...
                + (GIGp - 1) * self.logGeoMean ...
                - GIGa/2 .* self.mean ...
                - GIGb/2 .* self.firstInvMoment;
            res = sum(res(:));
        end
        
        function res = lowerboundcontrib(self)
            res = self.nu.lowerboundcontrib() + self.entropy() ...
                 + self.expectlogjoint();
        end
    end
end