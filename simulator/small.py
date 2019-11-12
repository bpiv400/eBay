import sys, os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from torch.utils.data import DataLoader
from interface import *
from simulator import Simulator
from constants import *

EPOCHS = 100


def run_loop(simulator, data, isTraining=False):
    # collate function
    f = collateRNN if data.isRecurrent else collateFF

    # load batches
    batches = DataLoader(data, batch_sampler=Sample(data, isTraining),
        collate_fn=f, num_workers=NUM_WORKERS, pin_memory=True)

    # loop over batches, calculate log-likelihood
    lnL = 0
    t0 = dt.now()
    for b, batch in enumerate(batches):
        t1 = dt.now()
        lnL += simulator.run_batch(batch, isTraining)
        print('gpu time: %d microseconds' % (dt.now() - t1).microseconds)
        print('total time: %d microseconds' % (dt.now() - t0).microseconds)
        t0 = dt.now()

    return lnL / data.N_labels


def train_model(simulator, train):
    # loop over epochs, record log-likelihood
    for i in range(EPOCHS):
        t0 = dt.now()

        # training loop
        print('Training epoch %d:' % i)
        run_loop(simulator, train, isTraining=True)

        # epoch duration
        dur = np.round((dt.now() - t0).seconds)


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str, 
        help='One of arrival, delay_byr, delay_slr, con_byr, con_slr.')
    args = parser.parse_args()
    model = args.model

    # load model sizes
    print('Loading parameters')
    sizes = pickle.load(
        open('%s/inputs/sizes/%s.pkl' % (PREFIX, model), 'rb'))
    print(sizes)

    # load neural net parameters
    params = {'layers': 2, 'hidden': 32}
    print(params)
    
    # initialize neural net
    simulator = Simulator(model, params, sizes, 
        device='cuda' if torch.cuda.is_available() else 'cpu')
    print(simulator.net)

    # load data
    train = Inputs('small', model)

    # train model
    train_model(simulator, train)
