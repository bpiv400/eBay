import sys, os, argparse
import numpy as np
from datetime import datetime as dt
from scipy.optimize import minimize_scalar
from train.Trainer import Trainer


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    name = parser.parse_args().name
    assert name in ['arrival', 'delay_byr', 'delay_slr']

    # experiment number
    expid = dt.now().strftime('%y%m%d-%H%M')

    # initialize trainer
    trainer = Trainer(name, 'train_models', 'train_rl', expid)

    # wrapper function
    loss = lambda logx: -trainer.train_model(smoothing=10 ** logx)

    result = minimize_scalar(loss, method='bounded', bounds=(0, 4), 
        options={'xatol': 0.1, 'disp': 3})