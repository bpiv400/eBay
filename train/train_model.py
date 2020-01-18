import sys, os, argparse
import numpy as np
from datetime import datetime as dt
from scipy.optimize import minimize_scalar
from train.Trainer import Trainer


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--smoothing', action='store_true')
    args = parser.parse_args()
    name, smoothing = args.name, args.smoothing

    # experiment number
    expid = dt.now().strftime('%y%m%d-%H%M')

    # initialize trainer
    train_part = 'train_rl' if name in ['listings', 'threads'] else 'train_models'
    trainer = Trainer(name, train_part, 'test_rl', expid)

    # use univariate optimizer to find smoothing hyperparameter
    if smoothing:
        assert name in ['arrival', 'delay_byr', 'delay_slr']
        loss = lambda logx: -trainer.train_model(smoothing=10 ** logx)
        result = minimize_scalar(loss, method='bounded', bounds=(1, 4), 
            options={'xatol': 0.1, 'disp': 3})
        print(result)

    # run training loop once with default of smoothing=0
    else:
        trainer.train_model()