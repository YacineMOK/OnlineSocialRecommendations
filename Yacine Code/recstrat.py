from SocialBandits_torch import SocialBandit_TorchF
from utils import *
import scipy
import logging 

# Sets the logger
logger = logging.getLogger('Social Bandits')


####################### LinREL 1
class LinREL1(SocialBandit_TorchF):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, delta=0.01, scale=1e-05, warmup=False):
        SocialBandit_TorchF.__init__(self, P, U0, alpha, sigma, lam)
        self.scale = scale
        self.delta = delta
        self.warmup = warmup # Pulls arms randomly if "warmup=True"
    def recommend(self):
        """ Recommend a V using the LinREL1 algorithm """

        # Initialization: 
        #    Pulls "d" (for each user)
        #    Observes rewards 
        if self.warmup and self.i < self.d:
            print("HERE")
            return SocialBandit_TorchF.recommend(self)
        
        print("Helloooooo")
        print("self.Z = ", self.Z)
        sqrtZ = torch.Tensor(sqrtm((self.Z)))
        print("sqrtZ OK")
        N = self.n * self.d
        print("N")
        b = max(128 * N * scipy.log(self.i + 2) * scipy.log((self.i + 2) ** 2 / self.delta),
                (8. / 3. * scipy.log((self.i + 2) ** 2 / self.delta)) ** 2)*self.scale
        print("before extrema")
        ext = extrema(sqrtZ, scipy.sqrt(N * b))
        print("after extrema")

        logger.debug("Optimizing over %d possible u extrema." % len(ext))
        L = self.generateL(self.A) # Kron between A(t) and Id
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

class LinREL1FiniteSet(LinREL1):
    """ LinREL class recommending over a finite set"""

    def getoptv(self, z):
        '''
        Redefine the getoptv // Adjust the 
        '''
        options = self.set
        Z = self.vec2mat(z)
        V = torch.zeros((self.n, self.d))
        totval = 0.0
        for i in range(self.n):
            optval = float("-inf")
            for j in range(self.M):
                # val = torch.real(torch.dot(Z[i, :], options[j, :]))
                val = torch.dot(Z[i, :], options[j, :])
                if val > optval:
                    logger.debug("getoptv found new maximimum at value %f" % val)
                    optval = val
                    V[i, :] = options[j, :]
            totval += optval
        return (self.mat2vec(V), totval)

####################### Regression LinREL 1
class RegressionLinREL1(SocialBandit_TorchF):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, delta=0.01, warmup=False):
        SocialBandit_TorchF.__init__(self, P, U0, alpha, sigma, lam)
        self.delta = delta
        self.warmup = warmup

    def recommend(self):
        """ Recommend a V using the LinREL1 algorithm """
        if self.warmup and self.i < self.d:
            return SocialBandit_TorchF.recommend(self)
        sqrtZ = sqrtm(self.Z)
        N = self.n * self.d
        
        L = self.generateL(self.A)
      
        u = torch.reshape(self.u0est, (1, N))
        z = torch.matmul(u, L)
        optv ,_ = self.getoptv(z)
      
        return self.vec2mat(optv)


####################### LinREL 2
class LinREL2(SocialBandit_TorchF):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, delta=0.01, scale=1e-05, warmup=False):
        SocialBandit_TorchF.__init__(self, P, U0, alpha, sigma, lam)
        self.delta = delta
        self.warmup = warmup
        self.scale = scale

    def recommend(self):
        """ Recommend a V using the LinREL2 algorithm """
        if self.warmup and self.i < self.d:
            return SocialBandit_TorchF.recommend(self)
        sqrtZ = sqrtm(self.Z)
        N = self.n * self.d
        b = max(128 * N *torch.log(self.i + 2) * torch.log((self.i + 2) ** 2 / self.delta),
                (8. / 3. * torch.log((self.i + 2) ** 2 / self.delta)) ** 2)*self.scale
        ext = extrema(sqrtZ, torch.sqrt(b))
        logger.debug("Optimizing over %d possible u extrema." % len(ext))
        L = self.generateL(self.A)
        optval = float("-inf")
        for u in ext:
            # Original -> u = (u ** 2 + self.u0est) ** 0.5
            # But it works only better if we set equal to:
            u = u + self.u0est
            u = torch.reshape(u, (1, N))
            z = torch.matmul(u, L)
            v, val = self.getoptv(z)
            if val >= optval:
                logger.debug("recommend found new maximum at value %f" % val)
                optval = val
                optv = v
                optu = u
        return self.vec2mat(optv)

class LinREL2FiniteSet(LinREL2):
    """ LinREL class recommending over a finite set"""
    def getoptv(self, z):
        options = self.set
        Z = self.vec2mat(z)
        V = torch.zeros((self.n, self.d))
        totval = 0.0
        for i in range(self.n):
            optval = float("-inf")
            for j in range(self.M):
                val = torch.dot(Z[i, :], options[j, :])
                if val > optval:
                    logger.debug("getoptv found new maximimum at value %f" % val)
                    optval = val
                    V[i, :] = options[j, :]
            totval += optval
        return (self.mat2vec(V), totval)

####################### Random Bandit
class RandomBanditL2Ball(SocialBandit_TorchF):
    def recommend(self):
        """ Recommend a random V"""
        gaussian = torch.random.randn(self.n, self.d)
        norms = torch.linalg.norm(gaussian, 2, 1)
        norms = torch.reshape(norms, (self.n, 1))
        M = 1. / torch.tile(norms, (1, self.d))
        return torch.multiply(gaussian, M)

####################### Random Bandit - Finite Set
class RandomBanditFiniteSet(SocialBandit_TorchF):
    def recommend(self):
        options = self.set
        recs = []
        for i in range(self.n):
            k = torch.random.randint(0, self.M)
            logger.debug("Selected arm %d out of %d for user %d" % (k, self.M, i))
            recs.append(options[k, :])
        return torch.reshape(torch.concatenate(recs), (self.n, self.d))