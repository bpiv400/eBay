import argparse
import numpy as np
from train.ConTrainer import Trainer
from train.train_consts import GRID_INC
from constants import TRAIN_RL, TRAIN_MODELS, VALIDATION


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    name = parser.parse_args().name

    # initialize trainer
    train_part = TRAIN_RL if name in ['listings', 'threads'] else TRAIN_MODELS
    trainer = Trainer(name, train_part, VALIDATION)

    # use grid search to find regularization hyperparameter
    gamma, best = 0, np.inf
    while True:
        loss = trainer.train_model(gamma=gamma)

        # stop if loss is worse than best
        if loss > best:
            break

        # reset best and increment gamma
        best = loss
        gamma += GRID_INC


if __name__ == '__main__':
    main()
