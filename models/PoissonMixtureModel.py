import dists.Mixture as Mixture
import dists.Gamma as Gamma

class PoissonMixtureModel(Mixture):
    def __init__(self, nc, dim):
        dist = Gamma(event_shape = (dim,), batch_shape=(nc,))
        super().__init__(dist, event_shape = (nc,))
