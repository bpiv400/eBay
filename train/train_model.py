import sys, os, argparse
import numpy as np
from datetime import datetime as dt
from scipy.optimize import minimize_scalar
from train.Trainer import Trainer


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    name, smoothing = args.name, args.smoothing

    # experiment number
    expid = dt.now().strftime('%y%m%d-%H%M')

    # initialize trainer
    train_part = 'train_rl' if name in ['listings', 'threads'] else 'train_models'
    trainer = Trainer(name, train_part, 'test_rl', expid)

    # use univariate optimizer to find regularization hyperparameter
    loss = lambda logx: -trainer.train_model(gamma=10 ** logx)
    result = minimize_scalar(loss, method='bounded', bounds=(-5, 1), 
        options={'xatol': 0.1, 'disp': 3})
    print(result)