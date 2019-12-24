import sys, os, argparse
import numpy as np
from compress_pickle import load, dump
from scipy.optimize import minimize_scalar
from models.FeedForwardDataset import FeedForwardDataset
from models.RecurrentDataset import RecurrentDataset
from models.trainer import Trainer
from utils import input_partition
from constants import *


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dropout', action='store_true', default=False)
    args = parser.parse_args()
    name, dropout = args.name, args.dropout

    # load model sizes
    sizes = load('{}/inputs/sizes/{}.pkl'.format(PREFIX, name))
    print('Parameters: {}'.format(sizes))

    # load datasets
    dataset = RecurrentDataset if 'x_time' in sizes else FeedForwardDataset
    print('Loading training data')
    train = dataset('train_models', name)
    print('Loading validation data')
    test = dataset('train_rl', name)

    # initialize trainer object
    print('Training model')
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
    