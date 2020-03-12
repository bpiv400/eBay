import argparse
from scipy.optimize import minimize_scalar
from train.Trainer import Trainer
from train.train_consts import GAMMA_TOL, GAMMA_MAX, GAMMA_MULTIPLIER
from constants import TRAIN_RL, TRAIN_MODELS, VALIDATION


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dropout', action='store_true')
    args = parser.parse_args()

    # initialize trainer
    train_part = TRAIN_RL if args.name in ['listings', 'threads'] else TRAIN_MODELS
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
    result = minimize_scalar(lambda p: trainer.train_model(dropout=p),
                             method='bounded',
                             bounds=(0, 1),
                             options={'xatol': 0.05, 'disp': 3})
    print(result)


if __name__ == '__main__':
    main()
