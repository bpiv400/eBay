import argparse
from train.Trainer import Trainer
from constants import SMALL, VALIDATION


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str, help='model name')
    parser.add_argument('--dropout', type=float, default=0)
    args = parser.parse_args()

    # initialize trainer
    trainer = Trainer(args.name, SMALL, VALIDATION, dev=True)

    # training
    trainer.train_model(dropout=args.dropout)


if __name__ == '__main__':
    main()
