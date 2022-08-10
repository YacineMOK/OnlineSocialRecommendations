from SocialBandits_torch import SocialBandit_TorchF
from utils import *

def testExpectedRewards(n, d, alpha=0.2):
    """ Test if different methods for computing rewards are correct """
    P = generateP(n)
    U0 = torch.randn(n, d)
    
    sb = SocialBandit_TorchF(P, U0, alpha)
    sb.run(20)


if __name__ == "__main__":
    n = 50
    d = 10
    testExpectedRewards(n,d)
