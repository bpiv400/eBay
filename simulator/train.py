import sys, os, argparse, math
import torch, torch.optim as optim
import numpy as np, pandas as pd
from compress_pickle import load
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import minimize_scalar
from simulator.interface import Inputs, run_loop
from simulator.model import Simulator
from constants import *


def weight_reset(m):
    if isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def train_model(log10_lr, folder, simulator, train, test):
    # reset model weights
    simulator.net.apply(weight_reset)

    # parameter initialization
    epoch, best = 1, -np.inf

    # folder and round number
    
    if os.path.isdir(folder):
        i = len([f for f in os.listdir(folder) if os.path.isfolder(f)]) + 1
    else:
        i = 1
    print('Round %d' % i)

    # initialize optimizer with given learning rate
    lr = math.pow(10, log10_lr)
    optimizer = optim.Adam(simulator.net.parameters(), lr=lr)
    print(optimizer)

    # initialize tensorboard writer  
    writer = SummaryWriter('%s/exp%d' % (folder, i))
    writer.add_scalar('lr', lr, i)

    # loop over epochs, record log-likelihood
    while True:
        print('Epoch %d' % epoch)

        # training loop
        t0 = dt.now()
        lnL_train = run_loop(simulator, train, optimizer)
        sec_train = (dt.now() - t0).total_seconds()

        # calculate log-likelihood on validation set
        with torch.no_grad():
            t0 = dt.now()
            lnL_test = run_loop(simulator, test)
            sec_test = (dt.now() - t0).total_seconds()

        # summarize loop
        print('\tTraining time: %d seconds' % sec_train)
        print('\tValidation time: %d seconds' % sec_test)
        print('\tlnL_train: %9.4f' % lnL_train)
        print('\tlnL_test: %9.4f' % lnL_test)

        # tensorboard logging
        info = {'lnL_train': lnL_train, 'lnL_test': lnL_test}
        for k, v in info.items():
            writer.add_scalar(k, v, epoch)

        # stop if training loss worsens
        if lnL_train > best:
            best = lnL_train
        else:
            return -lnL_train

        # increment epoch
        epoch += 1


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

    # initialize neural net
    simulator = Simulator(model, sizes, DROPOUT)
    print(simulator.net)
    print(simulator.loss)

    # load datasets
    train = Inputs('train_models', model)
    test = Inputs('train_rl', model)

    # empty tensorboard folder
    folder = '%s/%s' % (LOG_DIR, model)

    # optimize learning rate
    optimize_lr = lambda x: train_model(x, simulator, train, test)
    result = minimize_scalar(optimize_lr,
        method='bounded', bounds=(-4,-1), options={'xatol': 0.01})

    # save model
    torch.save(simulator.net.state_dict(), MODEL_DIR + '%s.net' % model)