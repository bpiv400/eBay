import sys, os, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from compress_pickle import load
from torch.utils.data import DataLoader
from interface import *
from simulator import Simulator
from constants import *


def run_loop(simulator, data):
    # collate function
    f = collateRNN if data.isRecurrent else collateFF
    # sampler
    sampler = Sample(data, simulator.mbsize, False)
    # load batches
    batches = DataLoader(data, batch_sampler=sampler,
        collate_fn=f, num_workers=NUM_WORKERS, pin_memory=True)
    # loop over batches, calculate log-likelihood
    print('%d batches.' % len(batches))
    t0 = dt.now()
    lnL, gpu_time, total_time = 0.0, 0.0, 0.0
    for batch in batches:
        t1 = dt.now()
        lnL += simulator.run_batch(batch, True)
        gpu_time += (dt.now() - t1).total_seconds() 
        total_time += (dt.now() - t0).total_seconds()
        t0 = dt.now()

    return lnL / data.N_labels, gpu_time, total_time


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
    params = pd.read_csv('%s/inputs/params.csv' % PREFIX, 
        index_col=0).loc[paramsid].to_dict()
    print(params)

    # initialize neural net
    simulator = Simulator(model, params, sizes, 
        device='cuda' if torch.cuda.is_available() else 'cpu')
    print(simulator.net)

    # load data
    data = load('%s/inputs/small/%s.gz' % (PREFIX, model))

    # time epoch
    lnL, gpu_time, total_time = run_loop(simulator, data)

    # print
    print('lnL: %.4f' % lnL)
    print('GPU time: %d seconds' % int(gpu_time))
    print('Total time: %d seconds' % int(total_time))