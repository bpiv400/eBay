import sys, os, argparse, math
import torch, torch.optim as optim
import numpy as np, pandas as pd
from datetime import datetime as dt
from compress_pickle import load
from torch.utils.data import DataLoader
from simulator.interface import Inputs, Sample, collateFF, collateRNN
from simulator.model import Simulator
from constants import *


def run_loop(simulator, data, optimizer=None):
    # training or validation
    isTraining = optimizer is not None

    # collate function
    f = collateRNN if data.isRecurrent else collateFF

    # minibatch size
    mbsize = simulator.mbsize if isTraining else MBSIZE_VALIDATION

    # sampler
    sampler = Sample(data, mbsize, isTraining)

    # load batches
    batches = DataLoader(data, batch_sampler=sampler,
        collate_fn=f, num_workers=NUM_WORKERS, pin_memory=True)

    # loop over batches, calculate log-likelihood
    lnL, gpu_time = 0.0, 0.0
    for batch in batches:
        t1 = dt.now()
        lnL += simulator.run_batch(batch, optimizer)
        gpu_time += (dt.now() - t1).total_seconds()

    return lnL / data.N_labels, gpu_time


def train_model(simulator, optimizer, data):
    # loop over epochs, record log-likelihood
    for i in range(EPOCHS):
        t0 = dt.now()

        # training loop
        print('Epoch %d' % (i+1))
        t0 = dt.now()
        lnL, gpu_time = run_loop(simulator, data, optimizer)
        print('GPU time: %d seconds' % gpu_time)
        print('Total time: %d seconds' % (dt.now() - t0).seconds)
        print('lnL: %9.4f' % lnL)


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str, 
        help='One of arrival, delay_byr, delay_slr, con_byr, con_slr.')
    parser.add_argument('--id', type=int, help='Experiment ID.')
    args = parser.parse_args()
    model = args.model
    paramsid = args.id

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
    print(sizes)

    # load experiment parameters
    params = pd.read_csv('outputs/params.csv', 
        index_col=0).loc[paramsid].to_dict()
    print(params)

    # initialize neural net
    simulator = Simulator(model, params, sizes, 
        device='cuda' if torch.cuda.is_available() else 'cpu')
    print(simulator.net)
    print(simulator.loss)

    # initialize optimizer
    optimizer = optim.Adam(simulator.net.parameters(), 
        betas=(0.9, 1-math.pow(10, params['b2'])),
        lr=math.pow(10, params['lr']))
    print(optimizer)

    # load data
    data = Inputs('small', model)

    # time epoch
    train_model(simulator, optimizer, data)

    # # save model
    # torch.save(simulator.net.state_dict(), 
    #     MODEL_DIR + '%s_%d.net' % (model, paramsid))