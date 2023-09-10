import argparse

# fmt:off
def get_args(rest_args):
    parser = argparse.ArgumentParser(description='MeRL-based ABR with vmaf')
    parser.add_argument('--test', action='store_true', help='Evaluate only')
    parser.add_argument('--proba', action='store_true', help='Use probabilistic policy')
    parser.add_argument('--non-acp', action='store_true', help='Not use action pruning')
    parser.add_argument('--mm22', action='store_true', help='mm22 Version')

    parser.add_argument('--adp', action='store_true', help='adaptation')
    parser.add_argument('--fscra', action='store_true', help='adaptation from scratch')
    parser.add_argument('--il', action='store_true', help='Only use imitation learning')
    parser.add_argument('--nmi', action='store_true', help='Not use mutual information loss')
    parser.add_argument('--from-il', action='store_true', help='Use the pre-training models learned by imitation learning')
    parser.add_argument('--name', default='merina', help='the name of result folder')
    parser.add_argument('--epochs', nargs='?', const=1050, default=1050, type=int, help='imitation training epochs')
    parser.add_argument('--init', action='store_true', help='Load the pre-train model parameters')
    parser.add_argument('--vap', action='store_true', help='Approximate the value function')
    parser.add_argument('--epochT', nargs='?', const=1e7, default=1e7, type=int, help='max training epochs')

    parser.add_argument('--mpc-h', nargs='?', const=5, default=5, type=int, help='The MPC planning horizon')
    parser.add_argument('--valid-i',nargs='?', const=100, default=100, type=int, help='The valid interval')
    
    ## Latent encoder 
    parser.add_argument('--latent-dim', nargs='?', const=16, default=16, type=int, help='The dimension of latent space')
    parser.add_argument('--kld-beta', nargs='?', const=0.01, default=0.01, type=float, help='The coefficient of kld in the VAE loss function')
    parser.add_argument('--kld-lambda', nargs='?', const=1.1, default=1.1, type=float, help='The coefficient of kld in the VAE recon loss function') ## control the strength of over-fitting of reconstruction, KL divergence between the prior P(D) and the distribution of P(D|\theta)
    parser.add_argument('--vae-gamma', nargs='?', const=0.7, default=0.7, type=float, help='The coefficient of reconstruction loss in the VAE loss function')
    
    ## Policy loss 
    parser.add_argument('--lc-alpha', nargs='?', const=5, default=5, type=float, help='The coefficient of cross entropy in the actor loss function')
    parser.add_argument('--lc-beta', nargs='?', const=0.25, default=0.25, type=float, help='The coefficient of entropy in the imitation loss function')
    parser.add_argument('--lc-mu', nargs='?', const=0.1, default=0.1, type=float, help='The coefficient of cross entropy in the actor loss function')
    parser.add_argument('--lc-gamma', nargs='?', const=0.10, default=0.10, type=float, help='The coefficient of mutual information in the actor loss function')
    parser.add_argument('--sp-n', nargs='?', const=10, default=10, type=int, help='The sample numbers of the mutual information')
    parser.add_argument('--gae-gamma', nargs='?', const=0.99, default=0.99, type=float, help='The gamma coefficient for GAE estimation')
    parser.add_argument('--gae-lambda', nargs='?', const=0.95, default=0.95, type=float, help='The lambda coefficient for GAE estimation')
    
    ## PPO configures 
    parser.add_argument('--batch-size', nargs='?', const=128, default=128, type=int, help='Mini-batch size for training')
    parser.add_argument('--ppo-ups', nargs='?', const=5, default=5, type=int, help='Update numbers in each epoch for PPO')
    parser.add_argument('--explo-num', nargs='?', const=20, default=20, type=int, help='Exploration steps for roll-out')
    parser.add_argument('--ro-len', nargs='?', const=25, default=25, type=int, help='Length of roll-out')
    parser.add_argument('--clip', nargs='?', const=0.02, default=0.02, type=float, help='Clip value of ppo')
    parser.add_argument('--anneal-p', nargs='?', const=0.95, default=0.95, type=float, help='Annealing parameters for entropy regularization')
    parser.add_argument('--vap-e', nargs='?', const=250, default=250, type=float, help='the number of epochs to approximate the value function')
    parser.add_argument('--ema', nargs='?', const=0.999, default=0.999, type=float, help='ema value')
    
    ## choose datasets for throughput traces 
    parser.add_argument('--res-folder', default='test', help='the name of result folder')
    parser.add_argument('--tr-folder', default='puffer', help='the name of traces folder')

    return parser.parse_args(rest_args)
