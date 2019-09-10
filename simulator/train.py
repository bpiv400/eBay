import sys
sys.path.append('repo/')
sys.path.append('repo/simulator/')
import os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from simulator import Simulator
from constants import *
from utils import *


def train_model(simulator):
    time0 = dt.now()

    # initialize array for log-likelihoods by epoch
    lnL = []

    # loop over epochs, record log-likelihood
    i = 0
    while True:
        start = dt.now()

        # iterate over minibatches
        lnL.append(simulator.run_epoch())

        # print log-likelihood and duration
        print('Epoch %d: %dsec. lnL: %1.4f.' %
            (i+1, (dt.now() - start).seconds, lnL[i]))

        # break if no improvement for 100 epochs
        if (i >= 100) and (lnL[i] <= lnL[i-100]):
            break
        i += 1

    # return loss history and total duration
    return lnL, dt.now() - time0


# loads data and calls functions in utils.py to construct training inputs
def process_inputs(model, outcome):
    # initialize output dictionary
    d = {}

    # outcome
    y = pickle.load(open(TRAIN_PATH + 'y.pkl', 'rb'))
    d['y'] = y[model][outcome]
    del y

    # arrival models are all feed-forward
    x = pickle.load(open(TRAIN_PATH + 'x.pkl', 'rb'))
    if model == 'arrival':
        if outcome == 'days':
            d['x_fixed'] = parse_fixed_feats_days(x, d['y'].index)
        else:
            d['x_fixed'] = parse_fixed_feats_arrival(outcome, x)

    # byr and slr models are RNN-like
    else:
        if outcome == 'delay':
            z = pickle.load(open(TRAIN_PATH + 'z.pkl', 'rb'))
            d['x_fixed'] = parse_fixed_feats_delay(model, x)
            d['x_time'] = parse_time_feats_delay(model, d['y'].index, z)
        else:
            d['x_fixed'] = parse_fixed_feats_role(x)
            d['x_time'] = parse_time_feats_role(model, outcome, x['offer'])

    # dictionary of feature names, in order
    featnames = {k: v.columns for k, v in d.items() if k.startswith('x')}

    return convert_to_tensors(d), featnames


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str,
        help='One of: arrival, byr, slr.')
    parser.add_argument('--outcome', type=str, help='Outcome to predict.')
    parser.add_argument('--id', type=int, help='Experiment ID.')
    args = parser.parse_args()

    # extract parameters from CSV
    params = parse_params(args)
    print(params)

    # create inputs to model
    print('Creating model inputs')
    train, featnames = process_inputs(args.model, args.outcome)

    # print feature names
    for k, v in featnames.items():
        print(k)
        for c in v:
            print('\t%s' % c)

    # initialize neural net
    simulator = Simulator(args.model, args.outcome, train, params)
    print(simulator.net)

    # train model
    print('Training')
    lnL, duration = train_model(simulator)

    # save simulator parameters and other output
    prefix = MODEL_DIR + '_'.join([args.model, args.outcome, str(args.id)])
    torch.save(simulator.net.state_dict(), prefix + '.pt')
    pickle.dump([featnames, lnL, duration], open(prefix + '.pkl', 'wb'))