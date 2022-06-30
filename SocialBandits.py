#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from numpy.linalg import solve as solveLinear, inv, matrix_rank, pinv
import logging
from scipy.linalg import sqrtm
from pprint import pformat
import time
import argparse
import math
import random
import cvxopt
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
    Prowsums = np.reshape(np.sum(P, 1), (n, 1))
    x = [1 / i for i in Prowsums]
    M = 1. / np.tile(Prowsums, (1, n))  # Prowsums复制n列再每个数据求倒数
    return np.multiply(P, M)

def generatePFromFile(n,fileName):
    """Generate an n x n matrix P with 1/n on each cell"""
    """in order to smooth the probabilities"""
    P = np.full((n,n),1./n)
    f = open(fileName)
    for line in f.readlines():
        line = line.strip().split()
        line_len = len(line)
        val = float(line[2]) if line_len>2 else 1.
        P[int(line[0])][int(line[1])] = val
    f.close()
    #smoothing the probabilities
    S = P.sum(axis=1,dtype='float')
    for i in range(n):
        for j in range(n):
            P[i][j]=P[i][j]/S[i]
    return P

def generatePFromNetworkxGraph(n,G):
    """Generate an n x n matrix P with 1/n on each cell"""
    """in order to smooth the probabilities"""
    P = np.full((n,n),1./n)
    for e in G.edges():
        P[int(e[0])][int(e[1])] = 1.
        P[int(e[1])][int(e[0])] = 1.
    #smoothing the probabilities
    S = P.sum(axis=1,dtype='float')
    for i in range(n):
        for j in range(n):
            P[i][j]=P[i][j]/S[i]
    return P

def getU0FromFile(n,d,fileName):
    U0s = []
    f = open(fileName)
    for line in f.readlines():
        line = [float(x) for x in line.strip().split()]
        assert len(line)==d
        U0s.append(line)
    f.close()
    assert len(U0s)==n
    return np.array(U0s)


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
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, scale=0.00001):
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
        self.scale = scale
        I = np.identity(self.n)
        self.Ainf = self.alpha * inv(I - self.beta * P)

    def generateFiniteSet(self, M, seed=45):
        rng = np.random.RandomState(seed)
        self.set = rng.randn(int(M), self.d)
        self.M = int(M)

    def readFiniteSetFromFile(self, M, fileName):
        f = open(fileName)
        its = []
        for line in f.readlines():
            line = [float(x) for x in line.strip().split()]
            assert len(line)==self.d
            its.append(line)
        assert len(its)==M
        self.set = np.array(its)
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

    def updateZ(self, Zinv, X, sigma):
        """Update via Thompson sampling"""
        return LA.inv(Zinv+np.divide(np.matmul(X.T,X),sigma))

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
        rews = []

        self.Z, self.XTr = self.initializeRun()

        self.u0 = self.mat2vec(self.U0)
        self.u0est = np.zeros(self.n * self.d)
        self.A = self.generateA()
        self.i = 0
        while self.i < t:
            stat = {}
            t0 = time.perf_counter()
            V = self.recommend();

            X = self.generateX(self.A, V)
            r = self.generateRandomRewards(X)
            
            stat['reward']=self.expectedTotalRewardViaA(self.A, V) # It should be verified
            #print(rews[self.i])
            self.Z = self.updateZ(self.Z, X)
            self.XTr = self.updateXTr(self.XTr, X, r)
        
            self.u0est = self.regress(self.Z, self.XTr)
            udiff = np.linalg.norm(self.u0est - self.u0)
            Adiff = np.linalg.norm(np.reshape(self.A - self.Ainf, self.n * self.n))
            logger.info("It. %d: ||u_0-\hat{u}_0||_2 = %f, ||A-Ainf||= %f" % (self.i, udiff, Adiff))
            #print("It. %d: ||u_0-\hat{u}_0||_2 = %f, ||A-Ainf||= %f" % (self.i, udiff, Adiff))

            self.i += 1
            self.A = self.updateA(self.A)
            t1 = time.perf_counter() - t0
            stat['u0diff']=udiff
            stat['Adiff']=Adiff
            stat['time']=t1
            rews.append(stat)
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

class ThompsonSampling(SocialBandit):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, delta=0.01, scale=1e-05, warmup=False):
        SocialBandit.__init__(self, P, U0, alpha, sigma, lam)
        self.scale = scale
        self.delta = delta
        self.warmup = warmup
        self.sigma = sigma

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

    def run(self, t=10):
        rews = []

        self.Z, self.XTr = self.initializeRun()

        self.u0 = self.mat2vec(self.U0)
        self.u0est = np.random.normal(size=self.n * self.d)
        self.A = self.generateA()
        self.i = 0
        while self.i < t:
            stat = {}
            t0 = time.perf_counter()
            V = self.recommend();

            X = self.generateX(self.A, V)
            r = self.generateRandomRewards(X)
            
            stat['reward']=self.expectedTotalRewardViaA(self.A, V)
            
            Zinv = LA.inv(self.Z)
            s = np.random.normal(0,self.sigma,self.n)
            self.Z = self.updateZ(Zinv, X, self.sigma)
            self.XTr = self.updateXTr(self.XTr, X, np.divide(r+s,self.sigma))
        
            self.u0est = np.matmul(self.Z,np.matmul(Zinv,self.u0est)+self.XTr)

            udiff = np.linalg.norm(self.u0est - self.u0)
            Adiff = np.linalg.norm(np.reshape(self.A - self.Ainf, self.n * self.n))
            logger.info("It. %d: ||u_0-\hat{u}_0||_2 = %f, ||A-Ainf||= %f" % (self.i, udiff, Adiff))

            self.i += 1
            self.A = self.updateA(self.A)
            t1 = time.perf_counter() - t0
            stat['u0diff']=udiff
            stat['Adiff']=Adiff
            stat['time']=t1
            rews.append(stat)
        return rews
 

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

class RegressionFiniteSet(RegressionLinREL1):
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

class ThompsonSamplingFiniteSet(ThompsonSampling):
    
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

class ThompsonSamplingL2Ball(ThompsonSampling):
    
    def getoptv(self, z):
        Z = self.vec2mat(z)
        norms = np.linalg.norm(Z, 2, 1)
        # logger.debug("Max norm: %f, Min norm: %f, Counts: %d" % (max(norms),min(norms),len(norms)))
        norms = np.reshape(norms, (self.n, 1))
        M = np.nan_to_num(1. / np.tile(norms, (1, self.d)))
        return (self.mat2vec(np.multiply(Z, M)), sum(norms)) 

class RegressionL2Ball(RegressionLinREL1):
    """ LinREL class recommending over a finite set"""
    
    def getoptv(self, z):
        Z = self.vec2mat(z)
        norms = np.linalg.norm(Z, 2, 1)
        # logger.debug("Max norm: %f, Min norm: %f, Counts: %d" % (max(norms),min(norms),len(norms)))
        norms = np.reshape(norms, (self.n, 1))
        M = np.nan_to_num(1. / np.tile(norms, (1, self.d)))
        return (self.mat2vec(np.multiply(Z, M)), sum(norms))
    
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

class LinOptFiniteSet(LinREL1FiniteSet):
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
        v, val = self.getoptv(z)
        return self.vec2mat(v)


class LinOptL2Ball(LinREL1L2Ball):
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
        v, val = self.getoptv(z)
        return self.vec2mat(v)



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




class LinUCB(SocialBandit):
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001,\
            delta=0.01, scale=1e-05, warmup=False):
        SocialBandit.__init__(self, P, U0, alpha, sigma, lam)
        self.delta = delta
        self.warmup = warmup
        self.scale = scale
        #building matrices
        N = self.n*self.d
        r = []
        c = []
        v = []
        for i in range(self.n):
            for j in range(self.d):
                r.append(i)
                c.append((self.d*(N+2))*i+(N+2)*j)
                v.append(1.)
        r.append(self.n)
        c.append((N+1)*(N+1)-1)
        v.append(1.)
        self.G = cvxopt.spmatrix(v,c,r,((N+1)*(N+1),self.n+1))
        self.c = cvxopt.matrix(np.multiply(-1.,np.ones((self.n+1,1))))
        self.G_0 = cvxopt.matrix(np.identity(self.n+1))
        self.h_0 = cvxopt.matrix(np.zeros((self.n+1,1)))

    def recommend(self):
        """ Recommend a V using the LinREL2 algorithm """
        if self.warmup and self.i < self.d:
            return SocialBandit.recommend(self)
        sqrtZ = sqrtm(self.Z)
        N = self.n * self.d
        
        b = max(128 * N * np.log(self.i + 2) * np.log((self.i + 2) ** 2 / self.delta),
                (8. / 3. * np.log((self.i + 2) ** 2 / self.delta)) ** 2)*self.scale
        #logger.debug("Optimizing over %d possible u extrema." % len(ext))
       
        L = self.generateL(self.A)

        #values for LinUCB
        B = np.multiply(np.matmul(L.T,self.u0est),0.5)
        H_0 = np.multiply(b,np.matmul(np.matmul(L.T,self.Z),L))

        #matrix H for SDP relaxation
        H_up = np.concatenate((H_0,np.array([B]).T),axis=1)
        H_down = np.concatenate((B,[0]))
        H = np.concatenate((H_up,np.array([H_down])),axis=0)
        
        v, val = self.getoptv(H)

        return self.vec2mat(v)

    def getoptv(self, z):
        #constructing the matrices
        H = cvxopt.matrix(np.multiply(-1.,z))
        #solving the SDP
        sol = cvxopt.solvers.sdp(c=self.c,Gl=self.G_0,hl=self.h_0,\
                Gs=[self.G],hs=[H])
        Y = sol['zs'][0]
        if LA.matrix_rank(Y)==1:
            ev, evec = LA.eig(Y)
            y = np.multiply(evec[0],math.sqrt(ev[0]))
            return y[:-1].real*y[-1].real,0
        else:
            Z = LA.cholesky(Y)
            s = np.random.normal(size=self.n*self.d+1)
            sg = np.sign(np.matmul(Z,s))
            D = np.power(np.diag(np.diag(Y)),0.5)
            v = np.matmul(D,sg)
            return v[:-1]*v[-1],0

        #TODO get the final solution from Y when not rank 1

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


### Stochastic Versions of LinRel1 and LinRel2
""" Stochastically pdate matrix A to the next iteration.
    Uses formula:
        A(t+1) = A(t) * β P + α I
    Asumes that β = 1-α I
"""

def stochasticUpdate(A,alpha,P,n):
    dice = random.random()
    if dice<=1-alpha:
        return np.matmul(A,P)
    else:
        return np.identity(n)

class StochasticLinOptFiniteSet(LinOptFiniteSet):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

class StochasticLinOptL2Ball(LinOptL2Ball):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

class StochasticRandomBanditFiniteSet(RandomBanditFiniteSet):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

class StochasticRandomBanditL2Ball(RandomBanditL2Ball):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

class StochasticRegressionFiniteSet(RegressionFiniteSet):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

class StochasticRegressionL2Ball(RegressionL2Ball):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

class StochasticLinREL1FiniteSet(LinREL1FiniteSet):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

class StochasticLinREL1L2Ball(LinREL1L2Ball):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

class StochasticLinREL2FiniteSet(LinREL2FiniteSet):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

class StochasticLinUCB(LinUCB):
    def updateA(self, A):
        return stochasticUpdate(A,self.alpha,self.P,self.n)

### Ainf Versions of LinRel1 and LinRel2

class InfiniteLinOptFiniteSet(LinOptFiniteSet):
    def updateA(self, A):
        return self.Ainf

class InfiniteLinOptL2Ball(LinOptL2Ball):
    def updateA(self, A):
        return self.Ainf

class InfiniteRandomBanditFiniteSet(RandomBanditFiniteSet):
    def updateA(self, A):
        return self.Ainf

class InfiniteRandomBanditL2Ball(RandomBanditL2Ball):
    def updateA(self, A):
        return self.Ainf

class InfiniteRegressionFiniteSet(RegressionFiniteSet):
    def updateA(self, A):
        return self.Ainf

class InfiniteRegressionL2Ball(RegressionL2Ball):
    def updateA(self, A):
        return self.Ainf

class InfiniteLinREL1FiniteSet(LinREL1FiniteSet):
    def updateA(self, A):
        return self.Ainf

class InfiniteLinREL1L2Ball(LinREL1L2Ball):
    def updateA(self, A):
        return self.Ainf

class InfiniteLinREL2FiniteSet(LinREL2FiniteSet):
    def updateA(self, A):
        return self.Ainf

class InfiniteLinUCB(LinUCB):
    def updateA(self, A):
        return self.Ainf



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Social Bandit Simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('strategy', help="Recommendation strategy.",
                        choices=["RandomBanditL2Ball", "RandomBanditFiniteSet",\
                                "LinREL1L2Ball", "LinREL1FiniteSet",\
                                "LinREL2FiniteSet", "LinUCB",\
                                "LinOptFiniteSet","LinOptL2Ball",\
                                "RegressionFiniteSet","RegressionL2Ball",\
                                "ThompsonSamplingFiniteSet",\
                                "ThompsonSamplingL2Ball"])
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
    parser.add_argument('--graphtype', default='cmp',\
            help='Type of synthetic graph', choices=['cmp','erdos-renyi',\
            'barabasi-albert'])
    parser.add_argument('--networkfile', default='None', help='Edge list file')
    parser.add_argument('--itemfile', default='None', help='Item list file')
    parser.add_argument('--outfile', default='None', help='File output')
    parser.add_argument('--u0file', default='None', help='U0 file')
    parser.add_argument('--logfile', default='SB.log', help='Log file')
    parser.set_defaults(screen_output=True)
    parser.add_argument('--noscreenoutput', dest="screen_output",\
            action='store_false', help='Suppress screen output')
    parser.add_argument('--scale', default=1, type=float, help=\
            'scale of the β(t)  value. Used by LinREL') 
    parser.add_argument("--randseed", type=int, default=42, help="Random seed")
    parser.add_argument("--stochastic", type=bool, default=False,\
            help="Use stochastic choice of profile")
    parser.add_argument('--infinite', type=bool, default=False,\
            help="Use Ainf at every step")

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

    if args.stochastic==True:
        args.strategy = "Stochastic"+args.strategy
    elif args.infinite==True:
        args.strategy = "Infinite"+args.strategy

    BanditClass = eval(args.strategy)

    np.random.seed(args.randseed)
    random.seed(args.randseed)

    if args.networkfile=='None':
        #generate complete file randomly
        if args.graphtype=='cmp' or args.n<=3:
            P = generateP(args.n)
        else:
            G = nx.Graph()
            if args.graphtype=='erdos-renyi':
                prob = math.log(float(args.n))/\
                        float(args.n)
                G = nx.fast_gnp_random_graph(args.n,prob)
            elif args.graphtype=='barabasi-albert':
                m = int(math.log(float(args.n)))
                G = nx.barabasi_albert_graph(args.n,m)
            P = generatePFromNetworkxGraph(args.n,G)

    else:
        P = generatePFromFile(args.n,args.networkfile)


    if args.u0file=='None':
        U0 = np.random.randn(args.n, args.d)
    else:
        U0 = getU0FromFile(args.n, args.d, args.u0file)

   
    if "LinREL" in args.strategy:
        sb = BanditClass(P, U0, args.alpha, args.sigma, args.lam, args.delta,\
                args.scale)
    else:
        sb = BanditClass(P, U0, args.alpha, args.sigma, args.lam,\
                args.scale)
        
    if "Finite" in args.strategy:
        if args.itemfile=='None':
            sb.generateFiniteSet(args.M)
        else:
            sb.readFiniteSetFromFile(args.M,args.itemfile)

    res = sb.run(args.maxiter)

    if args.outfile!='None':
        pd.DataFrame(res).to_csv(args.outfile)
