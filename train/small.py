import sys, os, argparse
from train.Trainer import Trainer
from constants import SMALL, VALIDATION


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='model name')
    name = parser.parse_args().name

    # initialize trainer
    trainer = Trainer(name, SMALL, VALIDATION)

    # training
    trainer.train_model(gamma=1)