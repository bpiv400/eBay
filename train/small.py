import argparse
from train.ConTrainer import Trainer
from constants import SMALL, VALIDATION


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='model name')
    parser.add_argument('--gamma', type=float, default=1)
    args = parser.parse_args()
    name, gamma = args.name, args.gamma

    # initialize trainer
    trainer = Trainer(name, SMALL, VALIDATION, dev=True)

    # training
    trainer.train_model(gamma=gamma)


if __name__ == '__main__':
    main()
