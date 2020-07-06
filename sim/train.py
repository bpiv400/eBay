import argparse
from sim.SimTrainer import SimTrainer
from utils import set_gpu_workers
from constants import SMALL, TRAIN_RL, TRAIN_MODELS, DISCRIM_MODELS

DROPOUT_GRID = []
for j in range(8):
    for i in range(j+1):
        if j - i <= 3:
            DROPOUT_GRID.append((i / 10, j / 10))


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dropout', type=int, required=True)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    # set gpu and cpu affinity
    set_gpu_workers(args.gpu)

    # partition to train on
    if args.dev:
        train_part = SMALL
    elif args.name in DISCRIM_MODELS:
        train_part = TRAIN_RL
    else:
        train_part = TRAIN_MODELS

    # initialize trainer
    trainer = SimTrainer(name=args.name,
                         train_part=train_part,
                         dev=args.dev)

    # compute dropout
    dropout = DROPOUT_GRID[args.dropout-1]

    # train model
    trainer.train_model(dropout=dropout)


if __name__ == '__main__':
    main()
