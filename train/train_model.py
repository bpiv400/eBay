import argparse
from scipy.optimize import minimize_scalar
from train.Trainer import Trainer
from train.train_consts import INT_DROPOUT, DROPOUT_GRID
from constants import TRAIN_RL, TRAIN_MODELS, VALIDATION, DISCRIM_MODELS


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dropout', type=int, default=0)
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
    train_part = TRAIN_RL if args.name in DISCRIM_MODELS else TRAIN_MODELS
    trainer = Trainer(args.name, train_part, VALIDATION)

    # # use grid search to find regularization hyperparameter
    # multiplier = GAMMA_MULTIPLIER[args.name]
    # if multiplier == 0:
    #     trainer.train_model(gamma=0, dropout=args.dropout)
    # else:
    #     result = minimize_scalar(lambda g: trainer.train_model(gamma=g),
    #                              method='bounded',
    #                              bounds=(0, GAMMA_MAX * multiplier),
    #                              options={'xatol': GAMMA_TOL * multiplier,
    #                                       'disp': 3})
    #     print(result)

    # train model
    trainer.train_model(dropout=dropout)


if __name__ == '__main__':
    main()
