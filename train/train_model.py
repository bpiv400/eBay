import argparse
from scipy.optimize import minimize_scalar
from train.Trainer import Trainer
from train.train_consts import GAMMA_TOL, GAMMA_MAX, GAMMA_MULTIPLIER, INT_DROPOUT
from constants import TRAIN_RL, TRAIN_MODELS, VALIDATION, DISCRIM_MODELS


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dropout', type=int, default=0)
    args = parser.parse_args()

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

    # train
    dropout = (args.dropout - 1) / INT_DROPOUT
    trainer.train_model(dropout=dropout)


if __name__ == '__main__':
    main()
