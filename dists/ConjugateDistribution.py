import torch
# This is the template for a conjugate distribution.
# It contains routines that parameterize a prior distribution and associated
# likelihood function with rules for updating the parameters of the posterior
# computing expected log likelihoods and KL divergences as well as 
# ELBO contributions and expectations of the sufficient statistics
# Users can also specify additional expectations that may be of use.
# 
# May wish to consider using dictionaries instead of lists for parms and sufficient statistics

class ConjugateDistribution():
    def __init__(self,event_shape,batch_shape=(),prior_parms = None):
        self.min_event_dim = 0 # smallest allowable event dimension
        self.event_shape = event_shape
        self.batch_shape = batch_shape
        self.event_dim = len(event_shape)
        self.batch_dim = len(batch_shape)

        if prior_parms is None:
            # set default parameters of the prior distribution here
            # each element of this list is a function of the event_shape expanded to the batch_shape
            self.eta_0 = []  
        else: 
            self.nat_parms_0 = prior_parms.copy()
        self.eta = []    # each list element is a random perturbation of the corresponding element of nat_parms_0
        self._ET = []
        self.SS = []

    def to_event(self,n):
        if n < 1:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]        
        return self

    def T(self,X):  # evaluate the sufficient statistic return a list of tensors
        pass

    def ET(self):  # expected value of the sufficient statistic given the natural parameters, self.nat_parms
        if self._ET is None:
            if self.sample_ET is True:
                self._ET = self.ET_sampler()
            else:
                # self._ET = correct function of natual parameters
                pass
        return self._ET

    def log_measure(self,x):
        pass

    def log_partition(self):
        # log of the partition function of the likelihood evaluated
        # at the expected sufficient statistic A(<\eta>) = A(nat_parms)
        pass

    def expected_log_partition(self):  
        pass

    def logZ(self):  # log partition function of the natural parameters often called A(\eta)
        return self._logZ
    
    def logZ_prior(self):
        return self._logZ_prior

    def logZ_ub(self): # upper bound on the log partition function 
        return self._logZ_ub

    def ET_sampler(self):
        # builds ET(eta) by sampling given the current value of the natural parameters
        # and pushes the result to self._ET as well as self._logZ
        pass 

    def conjugate_likelihood(self,X):
        # conjugate likelihood function return SS list
        pass

    def ss_update(self,SS,lr=1.0,beta=None):
        # SS is a list of coefficients from the conjugate likelihood function with same shape as nat_parms
        if beta is not None:
            for i in range(len(self.nat_parms)):
                self.SS[i] = beta*self.SS[i] + SS[i]
        else:
            self.SS = SS.copy()
        for i, parm in self.nat_parms:        
            self.nat_parm[i] = (self.SS[i] + self.nat_parms_0[i])*lr + self.nat_parms[i]*(1-lr)
        self._ET = None
        self._logZ = None

    def raw_update(self,X,p=None,lr=1.0):
        if p is None: 
            EmpT = self.T(X)
        else:  # assumes p is sample by batch
            if(self.batch_dim==0):
                sample_shape = p.shape
            else:
                sample_shape = p.shape[:-self.batch_dim]
            EmpT = self.T(X.view(sample_shape+self.batch_dim*(1,)+self.event_shape))*p.view(p.shape + self.event_dim*(1,)) 
        while EmpT.ndim > self.event_dim + self.batch_dim:
            EmpT = EmpT.sum(0)
        self.ss_update(EmpT,lr)

    def _KL_qprior(self):  
        # returns the KL divergence between prior (nat_parms_0) and posterior (nat_parms)
        pass

    def KL_qprior(self):
        # returns KL divergence with shape (batch_shape)
        KL = self._KL_qprior()
        sum_dim = self.event_dim - self.event_dim_0
        if sum_dim > 0:
            KL = KL.sum(list(range(-sum_dim,0)))

    def _Elog_like(self,X):    
        # reuturns the likelihood of X under the default event_shape
        pass

    def Elog_like(self,X):   
        # retuns the Expected log likelihood with shape (sample_shape + batch_shape)
        # for use in mixing weights calculation
        ELL = self._Elog_like(self,X)
        sum_dim = self.event_dim - self.event_dim_0
        if sum_dim > 0:
            ELL = ELL.sum(list(range(-sum_dim,0)))

    def sample(self,sample_shape=()):
        pass

    # Define other expectations that may be of use:
    # def mean():
    #     pass

