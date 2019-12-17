import sys, os, argparse
import numpy as np
from compress_pickle import load, dump
from scipy.optimize import minimize_scalar
from models.simulator.interface import Inputs
from models.simulator.model import Simulator
from models.simulator.trainer import Trainer
from constants import *


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    model = args.model

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
    print(sizes)

    # load datasets
    train = Inputs('small', model)
    test = Inputs('train_rl', model)

    # initialize trainer object
    trainer = Trainer(model, sizes, train, test)

    # pretraining
    if trainer.iter == 0:
        trainer.train_model()
    
    # find optimal gamma
    wrapper = lambda x: trainer.train_model(x)
    result = minimize_scalar(wrapper, 
        method='Bounded', bounds=(0,10), options={'xatol': 0.5})

    # save result
    dump(result, EXPS_DIR + '%s.pkl' % model)
    