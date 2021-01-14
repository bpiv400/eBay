import argparse
import torch
from sim.SimTrainer import SimTrainer
from utils import set_gpu
from constants import DROPOUT_GRID
from featnames import MODELS, DISCRIM_MODEL


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        choices=MODELS + [DISCRIM_MODEL])
    parser.add_argument('--dropout', type=int, required=True,
                        choices=range(len(DROPOUT_GRID)))
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    # set gpu and cpu affinity
    set_gpu(args.gpu)

    # initialize trainer
    trainer = SimTrainer(name=args.name)

    # compute dropout
    dropout = DROPOUT_GRID[args.dropout]

    # train model
    trainer.train_model(dropout=dropout, log=not args.nolog)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
