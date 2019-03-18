# python imports
import argparse

# local imports
from NNDP.train_nndp import train
from utils.prepare_data import prepare_data
from NNDP.nndp import NNDP

# library imports
import torch

def CreateArgsParser():
    parser = argparse.ArgumentParser(description='NNDP Pytorch')

    parser.add_argument('--iters', type=int, default=1000, metavar='N',
                    help='number of iterations to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
    parser.add_argument('--file', required=True, metavar='F')

    return parser

def main():
    args = CreateArgsParser().parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    graph = prepare_data(args.file)
    n_verticies = len(graph)

    nndp_model = NNDP(n_verticies)

    train(args, nndp_model, graph, n_verticies, device)


if __name__ == '__main__':
    main()