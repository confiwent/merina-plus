import argparse

# fmt:off
def get_args(rest_args):
    parser = argparse.ArgumentParser(description='Plot resutls in TMM')
    # load data
    parser.add_argument('--res-folder', default='test', help='the name of result folder')
    parser.add_argument('--tr-folder', default='puffer', help='the name of traces folder')
    parser.add_argument('--tf', action='store_true', help='Use FCC traces')
    parser.add_argument('--tfh', action='store_true', help='Use FCCand3GP traces')
    parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
    parser.add_argument('--to', action='store_true', help='Use Oboe traces')
    parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
    parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')
    parser.add_argument('--tp3', action='store_true', help='Use Puffer3 traces')

    # load baselines
    parser.add_argument('--iml', action='store_true', help='Show the results of IMRL without MI')
    parser.add_argument('--comyco', action='store_true', help='Show the results of Comyco')
    parser.add_argument('--geser', action='store_true', help='Show the results of Geser')
    parser.add_argument('--mpc', action='store_true', help='Show the results of RobustMPC')
    parser.add_argument('--pensieve', action='store_true', help='Show the results of Penseive')
    parser.add_argument('--imrl', action='store_true', help='Show the results of IMRL')
    parser.add_argument('--ppo', action='store_true', help='Show the results of PPO')
    parser.add_argument('--mppo', action='store_true', help='Show the results of MPPO')
    parser.add_argument('--oracle', action='store_true', help='Show the results of MPC-Oracle')
    parser.add_argument('--bola', action='store_true', help='Show the results of BOLA')
    parser.add_argument('--adp', action='store_true', help='Show the results of adaptation')
    parser.add_argument('--fugo', action='store_true', help='Show the results of FUGU')
    parser.add_argument('--bayes', action='store_true', help='Show the results of BayesMPC')
    parser.add_argument('--maml', action='store_true', help='Show the results of A2BR')
    parser.add_argument('--mmrl', action='store_true', help='Show the results of MERINA of MM22')
    parser.add_argument('--nacrl', action='store_true', help='Show the results of MERINA+ without ac')
    parser.add_argument('--nmirl', action='store_true', help='Show the results of MERINA+ without mi')

    # set comparison baseline
    parser.add_argument('--baseline', default='mpc', help='the name of baseline alg')

    #-------- set pic xlim ----------
    parser.add_argument('--xlim-min', nargs='?', const=-99, default=-99, type=float, help='minimum of xlim for cdf')
    parser.add_argument('--xlim-max', nargs='?', const=-99, default=-99, type=float, help='maximum of xlim for cdf')

    parser.add_argument('--nplot', action='store_true', help='Not plot the results')

    return parser.parse_args(rest_args)
