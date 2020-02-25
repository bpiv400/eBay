import argparse
from scipy.optimize import minimize_scalar
from train.ConTrainer import Trainer
from train.train_consts import GAMMA_TOL, GAMMA_MAX
from constants import TRAIN_RL, TRAIN_MODELS, VALIDATION
from featnames import CON


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    name = parser.parse_args().name

    # initialize trainer
    train_part = TRAIN_RL if name in ['listings', 'threads'] else TRAIN_MODELS
    trainer = Trainer(name, train_part, VALIDATION)

    # use grid search to find regularization hyperparameter
    result = minimize_scalar(lambda g: trainer.train_model(gamma=g),
                             method='bounded',
                             bounds=(0, GAMMA_MAX),
                             options={'xatol': GAMMA_TOL, 'disp': 3})
    print(result)


if __name__ == '__main__':
    main()
