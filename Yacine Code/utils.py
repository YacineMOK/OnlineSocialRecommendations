import torch
import pandas as pd
import numpy as np 

# Utils libraries
from scipy.linalg import sqrtm

# Generate a square matrix of size "n" such as :
#   1. All values are floats from 0 to 1
#   2. The sum of each row equals 1
def generateP(n):
    """Generate an n x n matrix P
    This matrix represents/simulates the social influence probabilities
    """
    mat = torch.rand(n, n)
    matrowsums = 1./torch.sum(mat, 1).reshape(n,1)
    return mat.mul(matrowsums) 

####################### Compute extreme points of a set
def sbasis(i, n):
    """Standard basis vector e_i of dimension n."""
    arr = torch.zeros(n)
    arr[i] = 1.0
    return arr

def extrema(B, c):
    """Return the extreme points of set:

           x :  || B x ||_1 <= c
    """
    Binv = torch.linalg.inv(B)
    n, n = B.shape
    basis = [sbasis(i, n) for i in range(n)]
    nbasis = [-e for e in basis]
    pnbasis = basis + nbasis 
    return [c * torch.matmul(Binv, y) for y in pnbasis]

####################### From files (inherent profiles, probabilities, ...)

def getU0FromFile(n,d,fileName):
    """
        Returns a torch.Tensor object
    """
    U0s = []
    f = open(fileName)
    for line in f.readlines():
        line = [float(x) for x in line.strip().split()]
        assert len(line)==d
        U0s.append(line)
    f.close()
    assert len(U0s)==n
    return torch.Tensor(U0s)

def generatePFromFile(n,fileName):
    """Generate an n x n matrix P with 1/n on each cell"""
    """in order to smooth the probabilities"""
    P = torch.from_numpy(np.full((n,n),1./n))
    f = open(fileName)
    for line in f.readlines():
        line = line.strip().split()
        line_len = len(line)
        val = float(line[2]) if line_len>2 else 1.
        P[int(line[0])][int(line[1])] = val
    f.close()
    #smoothing the probabilities
    S = P.sum(axis=1,dtype=torch.float)
    for i in range(n):
        for j in range(n):
            P[i][j]=P[i][j]/S[i]
    return P