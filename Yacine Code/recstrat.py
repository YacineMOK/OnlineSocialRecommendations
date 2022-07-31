from SocialBandits_torch import SocialBandit_TorchF
from utils import *

class LinREL1(SocialBandit_TorchF):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, delta=0.01, scale=1e-05, warmup=False):
        SocialBandit_TorchF.__init__(self, P, U0, alpha, sigma, lam)
        self.scale = scale
        self.delta = delta
        self.warmup = warmup

    def recommend(self):
        """ Recommend a V using the LinREL1 algorithm """


        # Initialization: 
        #    Pulls "d" (for each user)
        #    Observes rewards 
        if self.warmup and self.i < self.d:
            return SocialBandit_TorchF.recommend(self)
        sqrtZ = sqrtm(self.Z)
        N = self.n * self.d
        b = max(128 * N * torch.log(self.i + 2) * torch.log((self.i + 2) ** 2 / self.delta),
                (8. / 3. * torch.log((self.i + 2) ** 2 / self.delta)) ** 2)*self.scale
        ext = extrema(sqrtZ, torch.sqrt(N * b))
        
        logger.debug("Optimizing over %d possible u extrema." % len(ext))
        L = self.generateL(self.A)
        optval = float("-inf")
        for u in ext:
            ue = u + self.u0est
            ue = torch.reshape(ue, (1, N))
            z = torch.matmul(ue, L)
            v, val = self.getoptv(z)
            # logger.debug("Current value: %f" % val)
            if val > optval:
                logger.debug("recommend found new maximum at value %f" % val)
                optval = val
                optv = v
                optu = u
        return self.vec2mat(optv)