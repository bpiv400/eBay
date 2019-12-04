import sys, os, argparse, math
import torch, torch.optim as optim
import numpy as np, pandas as pd
from datetime import datetime as dt
from compress_pickle import load
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import minimize_scalar
from simulator.interface import Inputs, Sample, collateFF, collateRNN
from simulator.model import Simulator
from constants import *


def run_loop(simulator, data, optimizer=None):
    # training or validation
    isTraining = optimizer is not None

    # collate function
    f = collateRNN if data.isRecurrent else collateFF

    # sampler
    sampler = Sample(data, isTraining)

    # load batches
    batches = DataLoader(data, batch_sampler=sampler,
        collate_fn=f, num_workers=NUM_WORKERS, pin_memory=True)

    # loop over batches, calculate log-likelihood
    lnL = 0.0
    for batch in batches:
        lnL += simulator.run_batch(batch, optimizer)

    return lnL / data.N_labels


def train_model(log10_lr, simulator, data, writer):
    # initialize model weights
    torch.manual_seed(SEED)
    init_weights = lambda m: torch.nn.init.xavier_uniform_(m.weight.data)
    simulator.net.apply(init_weights)

    # initialize optimizer with given learning rate
    optimizer = optim.Adam(simulator.net.parameters(), 
        lr=math.pow(10, log10_lr))
    print(optimizer)

    # loop over epochs, record log-likelihood
    c, best = 0, -np.inf
    while c < K:
        t0 = dt.now()

        # training loop
        print('Epoch %d' % (i+1))
        t0 = dt.now()
        lnL, gpu_time = run_loop(simulator, data, optimizer)
        sec = (dt.now() - t0).total_seconds()

        # summarize loop
        print('\tElapsed time: %d seconds' % sec)
        print('\tlnL: %9.4f' % lnL)

        # tensorboard logging
        info = {'lnL_train': lnL, 'sec': sec}
        for k, v in info.items():
            writer.add_scalar(k, v, i+1)

        # update stopping parameters
        if lnL > best:
            c = 0
            best = lnL
        else:
            c += 1

    return -lnL


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
    simulator = Simulator(model, sizes)
    print(simulator.net)
    print(simulator.loss)

    # load data
    data = Inputs('small', model)

    # initialize tensorboard writer
    writer = SummaryWriter(LOG_DIR + '%s' % model)

    # wrapper for optimization
    optimize_lr = lambda x: train_model(x, simulator, data, writer)

    # optimize learning rate
    result = minimize_scalar(optimize_lr,
        method='bounded', bounds=(-5,-1), options={'xatol': 0.01})
    writer.close()

    # save model
    torch.save(simulator.net.state_dict(), MODEL_DIR + '%s.net' % model)