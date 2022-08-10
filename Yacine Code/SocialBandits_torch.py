#!/usr/bin/python
# -*- coding: latin-1 -*-

import torch
import logging
import time


# Sets the logger
logger = logging.getLogger('Social Bandits')

# Class
class SocialBandit_TorchF():
    def __init__(self, P, U0, alpha=0.2, sigma=0.0001, lam=0.001, scale=0.00001):
        """Initialize social bandit object. Arguments are:
           P: social influence matrix
           alpha: probability alpha that inherent interests are used
           U0: inherent interest matrix
           sigma:  noise standard deviation sigma, added when generating rewards
           lam: regularization parameter lambda used in ridge regression
        """
        self.P = P
        self.alpha = alpha
        self.beta = 1 - alpha
        self.U0 = U0
        self.n, self.d = U0.shape
        self.sigma = sigma
        self.lam = lam
        self.scale = scale
        I = torch.eye(self.n)
        self.Ainf = self.alpha * torch.inverse(I - self.beta * P)
###### RECOMMENDATION SET / CATALOG
    def generateFiniteSet(self, M, seed=45):
        """
        Generates a finit set of items (A catalogue)
            M : Number of items
        """
        ## Sets the seed localy
        # rng = np.random.RandomState(seed)
        # self.set = rng.randn(int(M), self.d)

        ## An equivalent way on PyTorch is to use the "Generators"
        # Info : https://discuss.pytorch.org/t/is-there-a-randomstate-equivalent-in-pytorch-for-local-random-generator-seeding/37131
        gen = torch.Generator()
        gen = gen.manual_seed(seed)
        self.set = torch.rand((int(M), self.d), generator=gen)
        self.M = int(M)

    def readFiniteSetFromFile(self, M, fileName):
        f = open(fileName)
        its = []
        for line in f.readlines():
            line = [float(x) for x in line.strip().split()]
            assert len(line)==self.d
            its.append(line)
        assert len(its)==M
        self.set = torch.Tensor(its)
        self.M = M

    def getFiniteSet(self):
        return self.set

    def setFiniteSet(self, fset, M):
        self.set = fset
        self.M = M

###### GENERATE MATRICES FOR COMPUTATIONS   &
    def updateA(self, A):
        """ Update matrix A to the next iteration. Uses formula:
            A(t+1) = A(t) * β P + α I
        """
        return torch.matmul(A, self.beta * self.P) + self.alpha * torch.eye(self.n)

    def generateA(self, t=0):
        """ Generate matrix A at iteration t. It returns:
            A(t) = α Σ_k=0^t (β P)^k
        """
        A = self.alpha * torch.eye(self.n)
        for k in range(t):
            A = self.updateA(A)
        return A

    def getU(self, A):
        """ Return current vector U, via
            U(t) = A(t) * U0
        """
        return torch.matmul(A, self.U0)

    def generateL(self, A):
        """ Generate large matrix L, given by the Kronecker product between A.T and the identity"""
        # print("Size of A:", A.size())
        # print("Size of dsds:",torch.eye(self.d).size())

        return torch.kron((A.T).contiguous(), torch.eye(self.d))

    def generateX(self, A, V):
        """ Generate matrix X from A and V"""
        n, d = self.n, self.d
        veclist = [A[i, j] * V[i, :] for i in range(n) for j in range(n)]

        ## Why did i chose "torch.view" rather than "torch.reshape"
        # -> https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
        ## torch.cat
        # -> https://discuss.pytorch.org/t/concatenate-torch-tensor-along-given-dimension/2304
        X = torch.reshape(torch.cat(veclist), (n, n * d))
        return X

###### mat2vec AND vec2mat
    def mat2vec(self, Mat):
        """ Convert a matrix to a vector form (row-wise).
        """
        n, d = self.n, self.d
        return torch.reshape(Mat, (n * d,))

    def vec2mat(self, vec):
        """ Convert a vector to a matrix form (row-wise).
        """
        n, d = self.n, self.d
        return torch.reshape(vec, (n, d))

###### COMPUTING THE REWARDS
    def expectedRewardsViaX(self, X):
        """ Compute the expected reward at each node via:
            r(t) = X(t) * u0
            where u0 = vec(U0)
        """
        u0 = self.mat2vec(self.U0)
        return torch.matmul(X, u0)
    
    def expectedRewardsViaA(self, A, V):
        """ Compute the expected reward at each node via:
            r(t) = <A * U0, V> * 1
            where <.> denotes the Hadamard (entry-wise) product
        """
        Z = torch.matmul(A, self.U0)
        topped = torch.multiply(Z, V)
        return torch.sum(topped, 1)  # computes row-wise sum
    
    def expectedTotalRewardViaL(self, L, V):
        """ Compute the total expected reward  via:
            rtot(t) = u0.T * L(t) * v
        """
        v = self.mat2vec(V)
        u0 = self.mat2vec(self.U0)

        # I hesitate between two things torch.mm and torch.dot
        res = 0
        try : 
            res = torch.mm(u0, torch.matmul(L, v))
        except RuntimeError:
            res = torch.dot(u0, torch.matmul(L, v))
        return res 
    
    def expectedTotalRewardViaX(self, X):
        """ Compute the total expected reward  via:
            rtot(t) = 1 * X(t) * u0
        """
        return torch.sum(self.expectedRewardsViaX(X))

    def expectedTotalRewardViaA(self, A, V):
        """ Compute the total expected reward  via:
            rtot(t) = 1 * X(t) * u0
        """
        return torch.sum(self.expectedRewardsViaA(A, V))

    def generateRandomRewards(self, X):
        return self.expectedRewardsViaX(X) + torch.rand(self.n) * self.sigma

    def generateZ(self, X):
        """Generate Z = XT*X """
        return torch.matmul(X.T, X)

    def updateZ(self, Z, X):
        """
        Following the LinREL algorithm:
            Update Z = Z+XT*X 
        """
        return Z + torch.matmul(X.T, X)

    def updateZ(self, Zinv, X, sigma):
        """Update via Thompson sampling"""
        return torch.inverse(Zinv+torch.divide(torch.matmul(X.T,X),sigma))

    def generateXTr(self, X, r):
        """Generate XT*r """
        return torch.matmul(X.T, r)
    
    def updateXTr(self, XTr, X, r):
        """Update XT*r with new X and r"""
        return XTr + torch.matmul(X.T, r)

    def regress(self, Z, XTr):
        """Solve least squares problem min_u||X*u-r||_2^2"""
        ## https://www.tutorialspoint.com/pytorch-torch-linalg-solve-method
        return torch.linalg.solve(Z, XTr)

    def recommend(self):
        print("Line1")
        gaussian = torch.randn(self.n, self.d)
        print("Line2")
        norms = torch.linalg.norm(gaussian, 2, 1)
        print("Line3")
        norms = torch.reshape(norms, (self.n, 1))
        print("Line4")
        M = 1. / torch.tile(norms, (1, self.d))
        print("Before return")
        return torch.multiply(gaussian, M)

    def initializeRun(self):
        Z = self.lam * torch.eye(self.n * self.d)
        XTr = torch.zeros(self.n * self.d)
        return (Z, XTr)

    def run(self, t=10):
        rews = []

        self.Z, self.XTr = self.initializeRun()

        self.u0 = self.mat2vec(self.U0)
        self.u0est = torch.zeros(self.n * self.d)
        self.A = self.generateA()
        self.i = 0
        while self.i < t:

            print("--- Itteration n°:", self.i)

            stat = {}
            t0 = time.perf_counter()
            
            print("\t--- Start")
            V = self.recommend();
            print("Rec")
            X = self.generateX(self.A, V)
            print("X")
            r = self.generateRandomRewards(X)
            print("random rew")

            stat['reward']=self.expectedTotalRewardViaA(self.A, V) # It should be verified
            #print(rews[self.i])
            
            
            self.Z = self.updateZ(self.Z, X, self.sigma)
            print("Z")
            self.XTr = self.updateXTr(self.XTr, X, r)
            print("XTr")
            self.u0est = self.regress(self.Z, self.XTr)
            print("U0_hat")
            udiff = torch.linalg.norm(self.u0est - self.u0)
            print("U diff")
            Adiff = torch.linalg.norm(torch.reshape(self.A - self.Ainf, (self.n * self.n, 1)))
            print("A Diff")
            logger.info("It. %d: ||u_0-\hat{u}_0||_2 = %f, ||A-Ainf||= %f" % (self.i, udiff, Adiff))
            print("It. %d: ||u_0-\hat{u}_0||_2 = %f, ||A-Ainf||= %f" % (self.i, udiff, Adiff))

            self.i += 1
            self.A = self.updateA(self.A)
            t1 = time.perf_counter() - t0
            stat['u0diff']=udiff
            stat['Adiff']=Adiff
            stat['time']=t1
            rews.append(stat)
        return rews