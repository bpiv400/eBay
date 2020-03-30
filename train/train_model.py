import argparse
from train.Trainer import Trainer
from train.train_consts import INT_DROPOUT, DROPOUT_GRID
from constants import SMALL, TRAIN_RL, TRAIN_MODELS, VALIDATION, DISCRIM_MODELS


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dropout', type=int, required=True)
    parser.add_argument('--norm', type=str)
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()

    # error checking normalization
    if args.norm is not None:
        assert args.norm in ['batch', 'layer', 'weight']

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

    # train model
    trainer.train_model(dropout=dropout, norm=args.norm)


if __name__ == '__main__':
    main()
