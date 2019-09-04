import sys
sys.path.append('../')
import os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from simulator import Simulator
from parsing_funcs import *
from constants import *


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


def process_inputs(model, outcome, data):
    x, y, z = [data[k] for k in ['x', 'y' ,'z']]
    # initialize output dictionary
    d = {}
    # outcome
    d['y'] = y[model][outcome]
    # arrival models are all feed-forward
    if model == 'arrival':
        if outcome == 'days':
            d['x_fixed'] = parse_fixed_feats_days(x, d['y'].index)
        else:
            d['x_fixed'] = parse_fixed_feats_arrival(outcome, x)
    # byr and slr models are RNN-like
    else:
        if outcome == 'delay':
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

    # load data
    print('Loading data')
    data = pickle.load(open(TRAIN_PATH, 'rb'))

    # create inputs to model
    print('Creating model inputs')
    train, featnames = process_inputs(args.model, args.outcome, data)

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