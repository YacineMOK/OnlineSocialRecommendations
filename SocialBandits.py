# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import solve as solveLinear,inv,matrix_rank
import logging
#from scipy.sparse import coo_matrix,csr_matrix
from pprint import pformat
from time import time
import argparse


def generateP(n):
    P=np.random.random((n,n))
    Prowsums = np.reshape(np.sum(P,1),(n,1))
    M = 1./np.tile(Prowsums,(1,n))
    return np.multiply(P,M)
     
def testExpectedRewards(n, d, alpha = 0.2):
    P = generateP(n)
    U0 = np.random.randn(n,d)
    V = np.random.randn(n,d)

    sb = SocialBandit(P,U0,alpha)
 
    A = sb.generateA(2)
    
     
    X = sb.generateX(A,V)
    L = sb.generateL(A)

    r1 = sb.expectedRewardsViaX(X)
    r2 = sb.expectedRewardsViaA(A,V)
    print r1
    print r2
    print "Reward vector difference is:",np.linalg.norm(r1-r2)

    rtot1= sb.expectedTotalRewardViaL(L,V)
    rtot2= sb.expectedTotalRewardViaX(X)
    rtot3= sb.expectedTotalRewardViaA(A,V)

    print "Total reward differences are: |r1-r2|=%f,|r2-r3|=%f,|r3-r1|=%f" % (np.abs(rtot1-rtot2),np.abs(rtot2-rtot3),np.abs(rtot3-rtot1))

def testRandomStrategy(n, d, t = 5,alpha = 0.2,sigma=0.001,lam = 0.00001):
    P = generateP(n)
    U0 = np.random.randn(n,d)
    
    sb = SocialBandit(P,U0,alpha,sigma)
    u0 = sb.mat2vec(U0)

    i = 0
    A=sb.generateA()
    Z = lam*np.identity(n*d)
    XTr = np.zeros(n*d)
    while i<t:
        V = np.random.randn(n,d)
        X = sb.generateX(A,V)
	
        r = sb.generateRandomRewards(X)     
        
        Z = sb.updateZ(Z,X)
        XTr = sb.updateXTr(XTr,X,r)

        u0est = sb.regress(Z,XTr)
        Adiff = np.reshape(A-sb.Ainf,n*n)
        print "It. %d: ||u_0-\hat{u}_0||_2 = %f, ||A-Ainf||= %f" % (i,np.linalg.norm(u0est-u0),np.linalg.norm(Adiff)) 
        i += 1
        A = sb.updateA(A)

class SocialBandit():
    def __init__(self,P, U0, alpha=0.2, sigma = 0.0):
        """Initialize social bandit object. Arguments are:
           P: social influence matrix
           alpha: probability α that inherent interests are used
           U0: inherent interest matrix 
           sigma:  noise standard deviation σ, added when generating rewards
        """
        self.P = P
        self.alpha = alpha
        self.beta = 1-alpha
        self.U0 = U0
        self.n,self.d = U0.shape
        self.sigma = sigma
        I = np.identity(self.n)
        self.Ainf = self.alpha*inv(I-self.beta*P)

    def updateA(self,A):
        """ Update matrix A to the next iteration. Uses formula:
            A(t+1) = A(t) * β P + α I
        """
        return np.matmul(A,self.beta*self.P)+self.alpha*np.identity(self.n) 

    def generateA(self,t=0):
        """ Generate matrix A at iteration t. It returns:
            A(t) = α Σ_k=1^t (β P)^k 
        """
        A = self.alpha*np.identity(self.n)
        for k in range(t):
            A = self.updateA(A)
        return A

    def getU(self,A):
        """ Return current vector U, via
            U(t) = A(t) * U0
        """
	return np.matmul(A,self.U0)


 
    def generateL(self,A):
        """ Generate large matrix L, given by the Kronecker product between A.T and the identity"""
        return np.kron(A.T,np.identity(self.d))

    def generateX(self,A,V):
        """ Generate matrix X from A and V"""
        n,d = self.n,self.d
        veclist = [ A[i,j]*V[i,:]  for i in range(n) for j in range(n)]
        X = np.reshape(np.concatenate(veclist),(n,n*d))
        return X


    def mat2vec(self,M):
        """ Convert a matrix to a vector form (row-wise).
        """
        n,d = self.n,self.d
        return np.reshape(M,(n*d,))
   
    def vec2mat(self,v):
        """ Convert a vector to a matrix form (row-wise).
        """
        n,d = self.n,self.d
        return np.reshape(v,(n,d))
   
    def expectedRewardsViaX(self,X):
        """ Compute the expected reward at each node via:
            r(t) = X(t) * u0
            where u0 = vec(U0)
        """
        u0=self.mat2vec(self.U0)
        return np.matmul(X,u0)

    def expectedRewardsViaA(self,A,V):
        """ Compute the expected reward at each node via:
            r(t) = <A * U0, V> * 1
            where <.> denotes the Hadamard (entry-wise) product
        """
        Z=np.matmul(A,self.U0)
        topped = np.multiply(Z,V)
        return np.sum(topped,1) #computes row-wise sum
 

    def expectedTotalRewardViaL(self,L,V):
        """ Compute the total expected reward  via:
            rtot(t) = u0.T * L(t) * v
        """
        v = self.mat2vec(V)
        u0 = self.mat2vec(self.U0)

        return np.dot(u0,np.matmul(L,v)) 
   

    def expectedTotalRewardViaX(self,X):
        """ Compute the total expected reward  via:
            rtot(t) = 1 * X(t) * u0
        """
	return np.sum(self.expectedRewardsViaX(X))

    def expectedTotalRewardViaA(self,A,V):
        """ Compute the total expected reward  via:
            rtot(t) = 1 * X(t) * u0
        """
	return np.sum(self.expectedRewardsViaA(A,V))
        
    def generateRandomRewards(self,X):
	"""Produce rewards with noise"""
        return self.expectedRewardsViaX(X)+np.random.randn(self.n)*self.sigma

    def generateZ(self,X):
        """Generate Z = XT*X """
        return np.matmul(X.T,X)

    def updateZ(self,Z,X):
        """Update Z = Z+XT*X """
        return Z+np.matmul(X.T,X)

    def generateXTr(self,X,r):
        """Generate XT*r """
        return np.matmul(X.T,r)


    def updateXTr(self,XTr,X,r):
        """Update XT*r with new X and r"""
        return XTr+np.matmul(X.T,r)


    def regress(self,Z,XTr):
        """Solve least squares problem min_u||X*u-r||_2^2"""
        return solveLinear(Z,XTr)
        
