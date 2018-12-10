# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:46:18 2018

"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:46:34 2018

"""

import numpy as np
from numpy.linalg import solve as solveLinear, inv, matrix_rank, pinv
import logging
from scipy.linalg import sqrtm
from pprint import pformat
from time import time
import argparse
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger('Social Bandits')


def sbasis(i, n):
    """Standard basis vector e_i of dimension n."""
    arr = np.zeros(n)
    arr[i] = 1.0
    return arr


def extrema(B, c):
    """Return the extreme points of set:

           x :  || B x ||_1 <= c
    """
    Binv = inv(B)
    n, n = B.shape
    basis = [sbasis(i, n) for i in range(n)]
    nbasis = [-e for e in basis]
    pnbasis = basis + nbasis  # 合并两个list
    #print(pnbasis)
    # for y in pnbasis:
    #     print(np.matmul(Binv, y))
    #     print(c * np.matmul(Binv, y))
    return [c * np.matmul(Binv, y) for y in pnbasis]


###############################################################################
'''
B=np.zeros([4,4])

for i in range(4):
    B[i,i]=i+1

print(B)

#c=2
#extrema(B,c)
'''


################################################################################
def clearFile(file):
    """Delete all contents of a file"""
    with open(file, 'w') as f:
        f.write("")


###############################################################################

def generateP(n):
    """Generate an n x n matrix P """
    P = np.random.random((n, n))
    #print('P=', P)
    #print('np.sum(P,1)=', np.sum(P, 1))  # 每行求和
    Prowsums = np.reshape(np.sum(P, 1), (n, 1))
    #print('Prowsums', Prowsums)  # 将原来数据形成一列
    x = [1 / i for i in Prowsums]
    #print(x)
    # print('np.tile(Prowsums,(1,n)=',np.tile(Prowsums,(1,n))
    M = 1. / np.tile(Prowsums, (1, n))  # Prowsums复制n列再每个数据求倒数
    #print(M)
    return np.multiply(P, M)

##############################################################################
# generateP(4)
###############################################################################
def testExpectedRewards(n, d, alpha=0.2):
    """ Test if different methods for computing rewards are correct """
    P = generateP(n)
    U0 = np.random.randn(n, d)
    # np.random.randn这个函数的作用就是从标准正态分布中返回一个或多个样本值。(n,d)表示维度
    V = np.random.randn(n, d)

    sb = SocialBandit(P, U0, alpha)

    A = sb.generateA(2)

    X = sb.generateX(A, V)
    L = sb.generateL(A)

    r1 = sb.expectedRewardsViaX(X)
    r2 = sb.expectedRewardsViaA(A, V)
    print(r1)
    print(r2)
    print("Reward vector difference is:", np.linalg.norm(r1 - r2))

    rtot1 = sb.expectedTotalRewardViaL(L, V)
    rtot2 = sb.expectedTotalRewardViaX(X)
    rtot3 = sb.expectedTotalRewardViaA(A, V)

    print("Total reward differences are: |r1-r2|=%f,|r2-r3|=%f,|r3-r1|=%f" % (
    np.abs(rtot1 - rtot2), np.abs(rtot2 - rtot3), np.abs(rtot3 - rtot1)))


###################################################################################
# testExpectedRewards(10, 8, alpha = 0.2)
###################################################################################

def testRandomStrategy(n, d, t=5, alpha=0.2, sigma=0.001, lam=0.00001):
    """ Test estimation and random strategy. It is better to test this from the main program instead"""
    P = generateP(n)
    U0 = np.random.randn(n, d)

    sb = SocialBandit(P, U0, alpha, sigma, lam)

    sb.run(t)


###############################################################################
# testRandomStrategy(10, 8, t = 5,alpha = 0.2,sigma=0.001,lam = 0.00001)
###############################################################################
class SocialBandit():
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001):
        """Initialize social bandit object. Arguments are:
           P: social influence matrix
           alpha: probability α that inherent interests are used
           U0: inherent interest matrix
           sigma:  noise standard deviation σ, added when generating rewards
           lam: regularization parameter λ used in ridge regression
        """
        self.P = P
        self.alpha = alpha
        self.beta = 1 - alpha
        self.U0 = U0
        self.n, self.d = U0.shape
        self.sigma = sigma
        self.lam = lam
        I = np.identity(self.n)
        self.Ainf = self.alpha * inv(I - self.beta * P)

    def generateFiniteSet(self, M, seed=45):
        rng = np.random.RandomState(seed)
        self.set = rng.randn(M, self.d)
        self.M = M

    def getFiniteSet(self):
        return self.set

    def setFiniteSet(self, fset, M):
        self.set = fset
        self.M = M

    def updateA(self, A):
        """ Update matrix A to the next iteration. Uses formula:
            A(t+1) = A(t) * β P + α I
        """
        return np.matmul(A, self.beta * self.P) + self.alpha * np.identity(self.n)

    def generateA(self, t=0):
        """ Generate matrix A at iteration t. It returns:
            A(t) = α Σ_k=0^t (β P)^k
        """
        A = self.alpha * np.identity(self.n)
        for k in range(t):
            A = self.updateA(A)
        return A

    def getU(self, A):
        """ Return current vector U, via
            U(t) = A(t) * U0
        """
        return np.matmul(A, self.U0)

    def generateL(self, A):
        """ Generate large matrix L, given by the Kronecker product between A.T and the identity"""
        return np.kron(A.T, np.identity(self.d))

    def generateX(self, A, V):
        """ Generate matrix X from A and V"""
        n, d = self.n, self.d
        veclist = [A[i, j] * V[i, :] for i in range(n) for j in range(n)]
        X = np.reshape(np.concatenate(veclist), (n, n * d))
        return X

    def mat2vec(self, M):
        """ Convert a matrix to a vector form (row-wise).
        """
        n, d = self.n, self.d
        return np.reshape(M, (n * d,))

    def vec2mat(self, v):
        """ Convert a vector to a matrix form (row-wise).
        """
        n, d = self.n, self.d
        return np.reshape(v, (n, d))

    def expectedRewardsViaX(self, X):
        """ Compute the expected reward at each node via:
            r(t) = X(t) * u0
            where u0 = vec(U0)
        """
        u0 = self.mat2vec(self.U0)
        return np.matmul(X, u0)

    def expectedRewardsViaA(self, A, V):
        """ Compute the expected reward at each node via:
            r(t) = <A * U0, V> * 1
            where <.> denotes the Hadamard (entry-wise) product
        """
        Z = np.matmul(A, self.U0)
        topped = np.multiply(Z, V)
        return np.sum(topped, 1)  # computes row-wise sum

    def expectedTotalRewardViaL(self, L, V):
        """ Compute the total expected reward  via:
            rtot(t) = u0.T * L(t) * v
        """
        v = self.mat2vec(V)
        u0 = self.mat2vec(self.U0)

        return np.dot(u0, np.matmul(L, v))

    def expectedTotalRewardViaX(self, X):
        """ Compute the total expected reward  via:
            rtot(t) = 1 * X(t) * u0
        """
        return np.sum(self.expectedRewardsViaX(X))

    def expectedTotalRewardViaA(self, A, V):
        """ Compute the total expected reward  via:
            rtot(t) = 1 * X(t) * u0
        """
        return np.sum(self.expectedRewardsViaA(A, V))

    def generateRandomRewards(self, X):
        return self.expectedRewardsViaX(X) + np.random.randn(self.n) * self.sigma

    def generateZ(self, X):
        """Generate Z = XT*X """
        return np.matmul(X.T, X)

    def updateZ(self, Z, X):
        """Update Z = Z+XT*X """
        return Z + np.matmul(X.T, X)

    def generateXTr(self, X, r):
        """Generate XT*r """
        return np.matmul(X.T, r)

    def updateXTr(self, XTr, X, r):
        """Update XT*r with new X and r"""
        return XTr + np.matmul(X.T, r)

    def regress(self, Z, XTr):
        """Solve least squares problem min_u||X*u-r||_2^2"""
        return solveLinear(Z, XTr)

    def recommend(self):
        gaussian = np.random.randn(self.n, self.d)
        norms = np.linalg.norm(gaussian, 2, 1)
        norms = np.reshape(norms, (self.n, 1))
        M = 1. / np.tile(norms, (1, self.d))
        return np.multiply(gaussian, M)

    def initializeRun(self):
        Z = self.lam * np.identity(self.n * self.d)
        XTr = np.zeros(self.n * self.d)
        return (Z, XTr)

    def run(self, t=10):
        rews = np.zeros(t)

        self.Z, self.XTr = self.initializeRun()

        self.u0 = self.mat2vec(self.U0)
        self.u0est = np.zeros(self.n * self.d)
        self.A = self.generateA()
        self.i = 0
        while self.i < t:
            V = self.recommend();

            X = self.generateX(self.A, V)
            r = self.generateRandomRewards(X)
            
            rews[self.i] = self.expectedTotalRewardViaA(self.A, V) # It should be verified
            print(rews[self.i])
            self.Z = self.updateZ(self.Z, X)
            self.XTr = self.updateXTr(self.XTr, X, r)
        
            self.u0est = self.regress(self.Z, self.XTr)
            udiff = np.linalg.norm(self.u0est - self.u0)
            Adiff = np.linalg.norm(np.reshape(self.A - self.Ainf, self.n * self.n))
            logger.info("It. %d: ||u_0-\hat{u}_0||_2 = %f, ||A-Ainf||= %f" % (self.i, udiff, Adiff))
            print("It. %d: ||u_0-\hat{u}_0||_2 = %f, ||A-Ainf||= %f" % (self.i, udiff, Adiff))

            self.i += 1
            self.A = self.updateA(self.A)
            
        return rews
        
###############################################################################
class RandomBanditL2Ball(SocialBandit):
    def recommend(self):
        """ Recommend a random V"""
        gaussian = np.random.randn(self.n, self.d)
        norms = np.linalg.norm(gaussian, 2, 1)
        norms = np.reshape(norms, (self.n, 1))
        M = 1. / np.tile(norms, (1, self.d))
        return np.multiply(gaussian, M)

class RandomBanditFiniteSet(SocialBandit):
    def recommend(self):
        options = self.set
        recs = []
        for i in range(self.n):
            k = np.random.randint(0, self.M)
            logger.debug("Selected arm %d out of %d for user %d" % (k, self.M, i))
            recs.append(options[k, :])
        return np.reshape(np.concatenate(recs), (self.n, self.d))


###############################################################################

class RegressionLinREL1(SocialBandit):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, delta=0.01, warmup=False):
        SocialBandit.__init__(self, P, U0, alpha, sigma, lam)
        self.delta = delta
        self.warmup = warmup

    def recommend(self):
        """ Recommend a V using the LinREL1 algorithm """
        if self.warmup and self.i < self.d:
            return SocialBandit.recommend(self)
        sqrtZ = sqrtm(self.Z)
        N = self.n * self.d
        
        L = self.generateL(self.A)
      
        u = np.reshape(self.u0est, (1, N))
        z = np.matmul(u, L)
        optv ,_ = self.getoptv(z)
      
        return self.vec2mat(optv)

class LinREL1(SocialBandit):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, delta=0.01, scale=1e-05, warmup=False):
        SocialBandit.__init__(self, P, U0, alpha, sigma, lam)
        self.scale = scale
        self.delta = delta
        self.warmup = warmup

    def recommend(self):
        """ Recommend a V using the LinREL1 algorithm """
        if self.warmup and self.i < self.d:
            return SocialBandit.recommend(self)
        sqrtZ = sqrtm(self.Z)
        N = self.n * self.d
        b = max(128 * N * np.log(self.i + 2) * np.log((self.i + 2) ** 2 / self.delta),
                (8. / 3. * np.log((self.i + 2) ** 2 / self.delta)) ** 2)*self.scale
        ext = extrema(sqrtZ, np.sqrt(N * b))
        
        logger.debug("Optimizing over %d possible u extrema." % len(ext))
        L = self.generateL(self.A)
        optval = float("-inf")
        for u in ext:
            ue = u + self.u0est
            ue = np.reshape(ue, (1, N))
            z = np.matmul(ue, L)
            v, val = self.getoptv(z)
            # logger.debug("Current value: %f" % val)
            if val > optval:
                logger.debug("recommend found new maximum at value %f" % val)
                optval = val
                optv = v
                optu = u
        return self.vec2mat(optv)


#####################################################################################

class LinREL1FiniteSet(LinREL1):
    """ LinREL class recommending over a finite set"""

    def getoptv(self, z):
        options = self.set
        Z = self.vec2mat(z)
        V = np.zeros((self.n, self.d))
        totval = 0.0
        for i in range(self.n):
            optval = float("-inf")
            for j in range(self.M):
                val = np.real(np.dot(Z[i, :], options[j, :]))
                if val > optval:
                    logger.debug("getoptv found new maximimum at value %f" % val)
                    optval = val
                    V[i, :] = options[j, :]
            totval += optval
        return (self.mat2vec(V), totval)

class RegressionLinREL1FiniteSet(RegressionLinREL1):
    """ Regression LinREL class recommending over a finite set"""

    def getoptv(self, z):
        options = self.set
        Z = self.vec2mat(z)
        V = np.zeros((self.n, self.d))
        totval = 0.0
        for i in range(self.n):
            optval = float("-inf")
            for j in range(self.M):
                val = np.dot(Z[i, :], options[j, :])
                if val > optval:
                    logger.debug("getoptv found new maximimum at value %f" % val)
                    optval = val
                    V[i, :] = options[j, :]
            totval += optval
        return (self.mat2vec(V), totval)
    
class LinREL1L2Ball(LinREL1):
    """ LinREL class recommending over a finite set"""
    
    def getoptv(self, z):
        Z = self.vec2mat(z)
        norms = np.linalg.norm(Z, 2, 1)
        # logger.debug("Max norm: %f, Min norm: %f, Counts: %d" % (max(norms),min(norms),len(norms)))
        norms = np.reshape(norms, (self.n, 1))
        M = np.nan_to_num(1. / np.tile(norms, (1, self.d)))
        return (self.mat2vec(np.multiply(Z, M)), sum(norms))


#######################################################################################################

class LinOptV1(LinREL1FiniteSet):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, delta=0.01, warmup=False):
        SocialBandit.__init__(self, P, U0, alpha, sigma, lam)
        self.delta = delta
        self.warmup = warmup

    def recommend(self):
        """ Recommend a V using the LinREL1 algorithm """
        if self.warmup and self.i < self.d:
            return SocialBandit.recommend(self)
        N = self.n * self.d
        u0 = self.U0
        u = np.reshape(u0, (1, N))
        L = self.generateL(self.A)
        z = np.matmul(u, L)
        optval = float("-inf")
        v, val = self.getoptv(z)
        # logger.debug("Current value: %f" % val)
        if val > optval:
            logger.debug("recommend found new maximum at value %f" % val)
            optval = val
            optv = v
            optu = u
        return self.vec2mat(optv)

    '''
class LinOptV1(LinREL1FiniteSet):
    def __init__(self,P, U0, alpha=0.2, sigma = 0.0001, lam = 0.001,delta = 0.01,warmup=False):
        SocialBandit.__init__(self,P, U0, alpha,sigma,lam)
        self.delta = delta
        self.warmup=warmup

    def recommend(self):
        """ Recommend a V using the LinREL1 algorithm """
        if self.warmup and self.i<self.d:
            return SocialBandit.recommend(self)       
        sqrtZ = sqrtm(self.Z)
        N = self.n*self.d
        b = max(  128*N*np.log(self.i+2)* np.log((self.i+2)**2/self.delta),(8./3.*np.log((self.i+2)**2/self.delta))**2  )
        ext = extrema(sqrtZ,np.sqrt(N*b))
        logger.debug("Optimizing over %d possible u extrema." % len(ext) )
        L = self.generateL(self.A)
        optval = float("-inf")
       # u0=self.mat2vec(self.U0)
        for u in ext:
            u = u+self.u0
            #print np.linalg.norm(u),max(u),min(u)
            u = np.reshape(u,(1,N))
            z = np.matmul(u, L)
            v,val = self.getoptv(z)
            #logger.debug("Current value: %f" % val)
        if val>optval:
                logger.debug("recommend found new maximum at value %f" % val)
                optval = val
                optv = v
                optu = u
        return self.vec2mat(optv)
'''


class LinREL2(SocialBandit):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, delta=0.01, scale=1e-05, warmup=False):
        SocialBandit.__init__(self, P, U0, alpha, sigma, lam)
        self.delta = delta
        self.warmup = warmup
        self.scale = scale

    def recommend(self):
        """ Recommend a V using the LinREL2 algorithm """
        if self.warmup and self.i < self.d:
            return SocialBandit.recommend(self)
        sqrtZ = sqrtm(self.Z)
        N = self.n * self.d
        b = max(128 * N * np.log(self.i + 2) * np.log((self.i + 2) ** 2 / self.delta),
                (8. / 3. * np.log((self.i + 2) ** 2 / self.delta)) ** 2)*self.scale
        ext = extrema(sqrtZ, np.sqrt(b))
        logger.debug("Optimizing over %d possible u extrema." % len(ext))
        L = self.generateL(self.A)
        optval = float("-inf")
        for u in ext:
            # Original -> u = (u ** 2 + self.u0est) ** 0.5
            # But it works only better if we set equal to:
            u = u + self.u0est
            u = np.reshape(u, (1, N))
            z = np.matmul(u, L)
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
        V = np.zeros((self.n, self.d))
        totval = 0.0
        for i in range(self.n):
            optval = float("-inf")
            for j in range(self.M):
                val = np.dot(Z[i, :], options[j, :])
                if val > optval:
                    logger.debug("getoptv found new maximimum at value %f" % val)
                    optval = val
                    V[i, :] = options[j, :]
            totval += optval
        return (self.mat2vec(V), totval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Social Bandit Simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('strategy', help="Recommendation strategy.",
                        choices=["RandomBanditL2Ball", "RandomBanditFiniteSet", "LinREL1L2Ball", "LinREL1FiniteSet",
                                 "LinREL2FiniteSet", "LinUCB"])
    parser.add_argument('--n', default=100, type=int, help="Number of users")
    parser.add_argument('--fois', default=500, type=int, help="Number of trial")
    parser.add_argument('--d', default=10, type=int, help="Number of dimensions")
    parser.add_argument('--alpha', default=0.05, type=float, help='α value. β is set to 1 - α ')
    parser.add_argument('--sigma', default=0.05, type=float, help='Standard deviation σ of noise added to responses ')
    parser.add_argument('--lam', default=0.01, type=float, help='Regularization parameter λ used in ridge regression')
    parser.add_argument('--delta', default=0.1, type=float, help='δ value. Used by LinREL ')
    parser.add_argument('--M', default=100, type=float,
                        help='Size M of finite set. Used by all finite set strategies. ')
    parser.add_argument('--maxiter', default=50, type=int, help='Maximum number of iterations')
    parser.add_argument('--debug', default='INFO', help='Verbosity level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--logfile', default='SB.log', help='Log file')
    parser.set_defaults(screen_output=True)
    parser.add_argument('--noscreenoutput', dest="screen_output", action='store_false', help='Suppress screen output')
    parser.add_argument("--randseed", type=int, default=42, help="Random seed")
    parser.add_argument("--test", type=bool, default=False)

    args = parser.parse_args()

    level = "logging." + args.debug
    logger.setLevel(eval(level))
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(eval(level))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    if args.screen_output:
        logger.addHandler(logging.StreamHandler())
    logger.info("Starting with arguments: " + str(args))
    logger.info('Level set to: ' + str(level))

    BanditClass = eval(args.strategy)

    np.random.seed(args.randseed)
    random.seed(args.randseed)

    P = generateP(args.n)
    U0 = np.random.randn(args.n, args.d)

    
    if "LinREL" in args.strategy:
        sb = BanditClass(P, U0, args.alpha, args.sigma, args.lam, args.delta)
    else:
        sb = BanditClass(P, U0, args.alpha, args.sigma, args.lam)
        
    if "Finite" in args.strategy:
        sb.generateFiniteSet(args.M)

    sb.run(args.maxiter)
