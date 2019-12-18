import sys, os, argparse
import numpy as np
from compress_pickle import load, dump
from scipy.optimize import minimize_scalar
from models.inputs import eBayDataset
from models.trainer import Trainer
from constants import *


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dropout', action='store_true', default=False)
    args = parser.parse_args()
    name, dropout = args.name, args.dropout

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, name))
    print(sizes)

    # load datasets
    train = eBayDataset('train_models', name)
    test = eBayDataset('train_rl', name)

    # initialize trainer object
    trainer = Trainer(name, sizes, train, test)

    # pretraining
    if trainer.iter == 0:
        trainer.train_model()
    
    # find optimal regularization coefficient
    if dropout:
        wrapper = lambda x: trainer.train_model(x)
        result = minimize_scalar(wrapper, 
            method='Bounded', bounds=(0,10), options={'xatol': 0.5})
        dump(result, EXPS_DIR + '%s.pkl' % name)
    