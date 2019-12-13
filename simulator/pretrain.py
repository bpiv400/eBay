import sys, os, argparse
import torch, torch.optim as optim
import numpy as np, pandas as pd
from compress_pickle import load, dump
from datetime import datetime as dt
from simulator.interface import Inputs, run_loop
from simulator.model import Simulator
from constants import *


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
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
    print(sizes)

    # initialize neural net without dropout
    simulator = Simulator(model, sizes, dropout=False)

    # initialize optimizer
    optimizer = optim.Adam(simulator.net.parameters())

    # print modules
    print(simulator.net)
    print(simulator.loss)
    print(optimizer)

    # load datasets
    train = Inputs('train_models', model)

    # training without variational dropout
    epoch, last = 0, np.inf
    while True:
        print('Epoch %d' % epoch)

        # training
        t0 = dt.now()
        loss = run_loop(simulator, train, optimizer)
        print('\tloss: %d' % loss)

        lnL = -loss / train.N_labels
        print('\tlnL: %1.4f' % lnL)         

        sec = (dt.now() - t0).total_seconds()
        print('\ttime: %d sec' % sec)

        # save model
        torch.save(simulator.net.state_dict(),
            MODEL_DIR + 'pretrain/%s.net' % model)

        # stopping condition
        if loss > last * FTOL:
            break

        # increment epochs and update last
        epoch += 1
        last = loss