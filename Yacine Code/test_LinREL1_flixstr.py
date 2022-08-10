from recstrat import LinREL1, LinREL1FiniteSet, LinREL2FiniteSet
import logging
from utils import *
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Social Bandit Simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('strategy', help="Recommendation strategy.",
                        choices=["LinREL1FiniteSet", "LinREL2FiniteSet"])
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

    print("\n--- The chosen stategy: ",args.strategy, " ---\n")

    # Generate P from a file
    P = generatePFromFile(args.n,args.networkfile)
    
    print("Génération de P")
    # Inherent Profiles
    if args.u0file=='None':
        U0 = torch.random.randn(args.n, args.d)
    else:
        U0 = getU0FromFile(args.n, args.d, args.u0file)

    print("Obtention de U0")

   # Strategies
    BanditClass = eval(args.strategy)
    print("Stratégie OK")
    if "LinREL" in args.strategy:
        sb = BanditClass(P, U0, args.alpha, args.sigma, args.lam, args.delta,\
                args.scale)
    else:
        sb = BanditClass(P, U0, args.alpha, args.sigma, args.lam,\
                args.scale)
    
    print("Social Bandit Class OK")
    # Finite catalogue (?)
    if "Finite" in args.strategy:
        if args.itemfile=='None':
            sb.generateFiniteSet(args.M)
        else:
            sb.readFiniteSetFromFile(args.M,args.itemfile)

    print("Finite  OK")
    # Run the social bandit algorithm
    res = sb.run(args.maxiter)

    print("Fun - RUN")

    # Writing results in a pandas dataframe if outfile args != none
    if args.outfile!='None':
        pd.DataFrame(res).to_csv(args.outfile)


     

