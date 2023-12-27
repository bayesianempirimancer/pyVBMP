classdef GaussGPEF < handle  %COMMENT this shit would be so much easier with pointers
    properties
        % hyperparameters
        D     % total dimension of data matrix
        types % types{k}.dist = 'mvn','normal','poisson','gamma','mult','binary'
              % types{k}.idx indicate the dimensions of the data matrix associated 
              % with that data type.  For the purposes of handling missing data
              % two independent poisson variabels should have a different
              % entries in the types cell array
              % 
              % Things still missing.  Currently the above distributions are 
              % implemented with conjugate priors except for gamma which is wierd.  
              % This aspect needs
              % updating and standardization.  E.g.  poisson should be
              % gammapoisson and a new class for gaussianpoisson and MVNpoisson 
              % should be
              % written.  Same for binomial and multinomial.  For this we
              % need implementations of the dual vb trick.  
              % Similarly normal will become normalwishartnormal, gamma
              % will become gammagamma, mult will becom dirmult, etc....
              %
              
        dists % same length as types containing the relevant distributions
        
    end
    
    methods 
        
        function self = GaussGPEF(types)
            self.types = types;
            self.D = 0;
            for k=1:length(types) 
                self.D = max(self.D,max(types{k}.idx));
                switch types{k}.dist
                    case 'mvn'
                        D=length(types{k}.idx);
                        self.dists{k}=dists.expfam.NW(zeros(D,1),1,eye(D),D+2);                         
                    case 'normal'
                        D=length(types{k}.idx);
                        self.dists{k}=dists.expfam.normalinvgamma(zeros(D,1),ones(D,1),ones(D,1),ones(D,1));                         
                    case 'poisson'
                        D=length(types{k}.idx);
                        self.dists{k}=dists.poissgamma(ones(D,1)/2,ones(D,1)/2);                         
                    case 'gamma'
                        D=length(types{k}.idx);
                        self.dists{k} = dists.gammagamma(ones(D,1)/2,ones(D,1)/2,ones(D,1)/2,ones(D,1)/2);
                    case 'mult'
                        D=length(types{k}.idx);
                        self.dists{k} = dists.expfam.dirichlet(D,ones(D,1)/2);
                    case 'binary'
                        D=length(types{k}.idx);
                        self.dists{k} = dists.expfam.betadist(ones(D,1)/2,ones(D,1)/2);
                    case 'poissonMVN'
                         D=length(types{k}.idx);
                         self.dists{k} = dists.poissonMVN(zeros(D,1),eye(D));
                    otherwise
                        fprintf([types{k}.dist,' not recognized.  Assuming independent normal'])
                        D=length(types{k}.idx);
                        self.dists{k}=dists.expfam.normalinvgamma(zeros(D,1),ones(D,1),ones(D,1),ones(D,1));                         
                end
            end

        end
        
        function update(self,data,iters,p)  
            
            % data is the raw data matrix of size ns by D
            % if present p is an assignment probability (for clustering)
            % and is assumed to be a vector

            % NW
            if(~exist('p','var'))
                p=ones(size(data,1),1); 
            end
            
            for i=1:iters
                if(isempty(data))
                     for k=1:length(self.dists);        
                        self.dists{k}.rawupdate([],0);
                     end
                else
                    for k=1:length(self.dists);        
                        self.dists{k}.rawupdate(data(:,self.types{k}.idx),p);
                    end
                end
            end
        end        

        
        function res = entropy(self)
            res=0;
            for k=1:length(self.dists);        
                res = res + self.dists{k}.entropy;
            end          
        end
        
        function res = KLqprior(self)
            res=0;
            for k=1:length(self.dists);        
                res = res + sum(self.dists{k}.KLqprior);
            end
        end
        
        function res = Eloglikelihood(self,data)
            res=0;
            for k=1:length(self.dists);        
                res = res + self.dists{k}.Eloglikelihood(data(:,self.types{k}.idx));
            end
        end
        
        function res = mean(self)
            res = NaN(1,self.D);
            for k=1:length(self.dists)
                res(self.types{k}.idx)=self.dists{k}.mean;
            end
        end

        function res = var(self)
            res = NaN(1,self.D);
            for k=1:length(self.dists)
                res(self.types{k}.idx)=self.dists{k}.var;
            end
        end

    end
end
