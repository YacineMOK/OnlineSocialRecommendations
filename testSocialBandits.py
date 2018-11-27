# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from SocialBandits import *

def testSocialBandits(P, U0, H, alpha, sigma, lam, delta):

    BanditStrategies = ['LinOptV1', 'RandomBanditFiniteSet', 'LinREL1L2Ball'] #, 'LinREL1L2Ball', 'RandomBanditFiniteSet']
    rewards = np.zeros((H, len(BanditStrategies)))
    regrets = []
    M = 100 # Number of finite set 
    
    for i, strategy in enumerate(BanditStrategies):
        BanditClass = eval(strategy)
        if "LinREL" in strategy:
            sb = BanditClass(P, U0, alpha, sigma, lam, delta)
        else:
            sb = BanditClass(P, U0, alpha, sigma, lam)
            
        sb.generateFiniteSet(M)
        rewards[:,i] = sb.run(H)

        if i > 0:
           regrets.append({'bandit': strategy, 'regret': np.cumsum(rewards[:,i] - rewards[:,0])})
    df = pd.DataFrame(regrets)
    df.to_csv('out.csv') # it should be renamed
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Social Bandit Test Runner',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n',default=100,type=int,help ="Number of users") 
    parser.add_argument('--h',default=500,type=int,help ="Horizon")
    parser.add_argument('--d',default=10,type=int,help ="Number of dimensions") 
    parser.add_argument('--alpha',default=0.05,type=float, help='alpha value. beta is set to 1 - alpha')
    parser.add_argument('--sigma',default=0.05,type=float, help='Standard deviation σ of noise added to responses ')
    parser.add_argument('--lam',default=0.01,type=float, help='Regularization parameter λ used in ridge regression')
    parser.add_argument('--delta',default=0.1,type=float, help='δ value. Used by LinREL')
    parser.add_argument('--M',default=100,type=float, help='Size M of finite set. Used by all finite set strategies. ')
    parser.add_argument('--maxiter',default=50,type=int, help='Maximum number of iterations')
    parser.add_argument('--debug',default='INFO', help='Verbosity level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument('--logfile',default='SB.log',help='Log file')
    parser.set_defaults(screen_output=True)
    parser.add_argument('--noscreenoutput',dest="screen_output",action='store_false',help='Suppress screen output')
    parser.add_argument("--randseed",type=int,default=42,help="Random seed")
    
    args = parser.parse_args()

    np.random.seed(args.randseed)
    random.seed(args.randseed)

    P = generateP(args.n)
    U0 = np.random.randn(args.n,args.d)

    testSocialBandits(P, U0, args.h, args.alpha, args.sigma, args.lam, args.delta)

    
    
    
