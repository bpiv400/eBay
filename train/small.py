import argparse
from train.Trainer import Trainer
from train.train_consts import INT_DROPOUT, DROPOUT_GRID
from constants import SMALL, VALIDATION


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str, help='model name')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--grid', action='store_true')
    args = parser.parse_args()

    # compute dropout
    if args.grid:
        assert args.dropout > 0 and args.dropout in DROPOUT_GRID
        dropout = [i  / INT_DROPOUT for i in DROPOUT_GRID[args.dropout]]
    else:
        assert 0 <= args.dropout < INT_DROPOUT
        dropout = args.dropout / INT_DROPOUT

    # initialize trainer
    trainer = Trainer(args.name, SMALL, VALIDATION, dev=True)

    # training loop
    trainer.train_model(dropout=dropout)


if __name__ == '__main__':
    main()
