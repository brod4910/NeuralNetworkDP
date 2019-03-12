# python imports
import argparse

# local imports
from NNDP.train_nndp import train

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
    train(args)


if __name__ == '__main__':
    main()