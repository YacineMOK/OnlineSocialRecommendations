# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import solve as linearSystemSolver,inv,matrix_rank
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
    r3 = sb.expectedRewardsViaA2(A,V)
    print r1
    print r2
    print r3
    print "Reward vector difference is:",np.linalg.norm(r1-r2),np.linalg.norm(r2-r3),np.linalg.norm(r3-r1)

    rtot1= sb.expectedTotalRewardViaL(L,V)
    rtot2= sb.expectedTotalRewardViaX(X)

    print "Total reward difference is:",np.abs(rtot1-rtot2)


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
            r(t) = <U0,A.T * V> * 1
            where <.> denotes the Hadamard (entrywise) product
        """
        Z=np.matmul(A.T,V)
        topped = np.multiply(self.U0,Z)
        print A.T
        print V
        print Z
        print self.U0
        print topped
        return np.sum(topped,1) #computes row-wise sum

    def expectedRewardsViaA2(self,A,V):
        """ Compute the expected reward at each node via:
            r(t) = <A *U0, V> * 1
            where <.> denotes the Hadamard (entrywise) product
        """
        Z=np.matmul(A,self.U0)
        topped = np.multiply(Z,V)
        print A
        print self.U0
        print Z
        print V
        print topped
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
     
