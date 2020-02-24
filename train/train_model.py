import argparse
from scipy.optimize import minimize_scalar
from train.ConTrainer import Trainer
from constants import TRAIN_RL, TRAIN_MODELS, VALIDATION


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    name = parser.parse_args().name

    # initialize trainer
    train_part = TRAIN_RL if name in ['listings', 'threads'] else TRAIN_MODELS
    trainer = Trainer(name, train_part, VALIDATION)

    # use univariate optimizer to find regularization hyperparameter
    loss = lambda g: trainer.train_model(gamma=g)
    result = minimize_scalar(loss, method='bounded', bounds=(0, 1), 
        options={'xatol': 0.1, 'disp': 3})
    print(result)


if __name__ == '__main__':
    main()
