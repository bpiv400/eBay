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

    # experiment number
    expid = dt.now().strftime('%y%m%d-%H%M')

    # initialize trainer
    trainer = Trainer(name, 'small', 'test_rl', expid)

    # wrapper function
    loss = lambda logx: -trainer.train_model(smoothing=10 ** logx)

    result = minimize_scalar(loss, 
        method='bounded', bounds=(0, 4), xatol=0.1)