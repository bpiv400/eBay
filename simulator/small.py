import sys, os, argparse
import torch, torch.optim as optim
import numpy as np, pandas as pd
from datetime import datetime as dt
from compress_pickle import load
from simulator.interface import Inputs, run_loop
from simulator.model import Simulator
from constants import *


def train_model(simulator, optimizer, data):
    # loop over epochs, record log-likelihood
    for epoch in range(1, 6):
        print('Epoch %d' % epoch)

        # training loop
        t0 = dt.now()
        lnL = run_loop(simulator, data, optimizer)
        sec = (dt.now() - t0).total_seconds()

        # summarize loop
        print('\tElapsed time: %d seconds' % sec)
        print('\tlnL: %9.4f' % lnL)


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(description='Training development.')
    parser.add_argument('--model', type=str, 
        help='One of arrival, hist, [delay/con/msg]_[byr/slr].')
    args = parser.parse_args()
    model = args.model

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
    print(sizes)

    # initialize neural net
    simulator = Simulator(model, sizes)
    print(simulator.net)
    print(simulator.loss)

    # initialize optimizer with default learning rate
    if (model == 'arrival') or ('delay' in model):
        optimizer = optim.Adam([
            {'params': simulator.net.h0.nn0.parameters()},
            {'params': simulator.net.h0.nn1.parameters()},
            {'params': simulator.net.h0.nn2.parameters()},
            {'params': simulator.net.c0.nn0.parameters()},
            {'params': simulator.net.c0.nn1.parameters()},
            {'params': simulator.net.c0.nn2.parameters()},
            {'params': simulator.net.rnn.parameters()}])
    else:
        optimizer = optim.Adam([
            {'params': simulator.net.nn0.parameters()},
            {'params': simulator.net.nn1.parameters()},
            {'params': simulator.net.nn2.parameters()}])
    print(optimizer)

    # load data
    data = Inputs('small', model)

    # train model
    train_model(simulator, optimizer, data)

    # save model
    torch.save(simulator.net.state_dict(), MODEL_DIR + '%s.net' % model)