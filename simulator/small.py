import sys, os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from torch.utils.data import DataLoader
from simulator.interface import *
from simulator.simulator import Simulator
from constants import *


def run_loop(simulator, data, f, mbsize):
    # sampler
    sampler = Sample(data, mbsize, False)
    # load batches
    batches = DataLoader(data, batch_sampler=sampler,
        collate_fn=f, num_workers=NUM_WORKERS, pin_memory=True)
    # loop over batches, calculate log-likelihood
    for batch in batches:
        simulator.run_batch(batch, True)


def find_mbsize(simulator, data):
    # collate function
    f = collateRNN if data.isRecurrent else collateFF

    # initialize mbsize
    mbsize = 128
    last = np.inf
    while True:
        # run loop and print total time
        t0 = dt.now()
        run_loop(simulator, data, f, mbsize)
        total_time = (dt.now() - t0).total_seconds()
        print('Minibatch size %d: %d sec' % (mbsize, total_time))
        # break if total time exceeds last run
        if total_time > last:
            break
        # increment mbsize
        mbsize *= 2
        # set last time to total_time
        last = total_time

# return mbsize from previous run
return int(mbsize / 2)



if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str, 
        help='Name of model (e.g., delay_byr)')
    args = parser.parse_args()
    model = args.model

    # load model sizes
    print('Loading parameters')
    sizes = pickle.load(
        open('%s/inputs/sizes/%s.pkl' % (PREFIX, model), 'rb'))
    print(sizes)

    # load neural net parameters
    params = {'layers': LAYERS, 'hidden': HIDDEN}
    print(params)
    
    # initialize neural net
    simulator = Simulator(model, params, sizes, 
        device='cuda' if torch.cuda.is_available() else 'cpu')
    print(simulator.net)

    # load data
    data = load('%s/inputs/small/%s.gz' % (PREFIX, model))

    # train model
    mbsize = find_mbsize(simulator, data)
