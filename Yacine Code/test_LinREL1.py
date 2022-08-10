from recstrat import LinREL1, LinREL1FiniteSet
import logging
from utils import *

# Sets the logger
logger = logging.getLogger('Social Bandits')


def testExpectedRewards(n, d, alpha=0.2):
    """ Test if different methods for computing rewards are correct """
    P = generateP(n)
    U0 = torch.randn(n, d)
    
    print(" ---- Test LinREL1 ---")
    # sb = LinREL1(P, U0, alpha)
    # sb.run(20)

    print(" ---- Test LinREL1 (finite set)---")
    sb = LinREL1FiniteSet(P, U0, alpha)
    sb.run(20)


if __name__ == "__main__":
    n = 50
    d = 10
    testExpectedRewards(n,d)
