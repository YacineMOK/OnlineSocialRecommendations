import torch

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
    Binv = inv(B)
    n, n = B.shape
    basis = [sbasis(i, n) for i in range(n)]
    nbasis = [-e for e in basis]
    pnbasis = basis + nbasis 
    return [c * torch.matmul(Binv, y) for y in pnbasis]
