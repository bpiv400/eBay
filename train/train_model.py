import argparse
from train.Trainer import Trainer
from train.train_consts import INT_DROPOUT, DROPOUT_GRID, NORM_TYPE
from constants import SMALL, TRAIN_RL, TRAIN_MODELS, VALIDATION, DISCRIM_MODELS


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dropout', type=int, required=True)
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()

    # partition to train on
    if args.dev:
        train_part = SMALL
    elif args.name in DISCRIM_MODELS:
        train_part = TRAIN_RL
    else:
        train_part = TRAIN_MODELS

    # initialize trainer
    trainer = Trainer(args.name, train_part, VALIDATION, dev=args.dev)

    # compute dropout
    dropout = tuple([float(i / INT_DROPOUT) for i in DROPOUT_GRID[args.dropout-1]])

    # normalization
    norm = 'batch' if args.name not in NORM_TYPE else NORM_TYPE[args.name]

    # train model
    trainer.train_model(dropout=dropout, norm=norm)


if __name__ == '__main__':
    main()
