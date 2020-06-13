import argparse
from train.Trainer import Trainer
from train.const import INT_DROPOUT, DROPOUT_GRID
from constants import SMALL, TRAIN_RL, TRAIN_MODELS, VALIDATION, \
    DISCRIM_MODELS, INIT_VALUE_MODELS


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
    elif args.name in DISCRIM_MODELS + INIT_VALUE_MODELS:
        train_part = TRAIN_RL
    else:
        train_part = TRAIN_MODELS

    # initialize trainer
    trainer = Trainer(args.name, train_part, VALIDATION, dev=args.dev)

    # compute dropout
    dropout = tuple([float(i / INT_DROPOUT) for i in DROPOUT_GRID[args.dropout-1]])

    # train model
    trainer.train_model(dropout=dropout)


if __name__ == '__main__':
    main()
