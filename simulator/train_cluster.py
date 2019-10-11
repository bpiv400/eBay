import sys, os, pickle, argparse
import torch
from torch.nn.utils import rnn
import numpy as np, pandas as pd
from compress_pickle import load, dump
from datetime import datetime as dt

sys.path.append('repo/simulator')
from simulator import Simulator

sys.path.append('repo/')
from constants import *

EPOCHS = 1000
HIDDEN = 256
LAYERS = 3
MODELS = [['arrival', 'bin'],
          ['arrival', 'loc'],
          ['arrival', 'hist'],
          ['arrival', 'sec'],
          ['slr', 'accept'],
          ['slr', 'reject'],
          ['slr', 'con'],
          ['slr', 'msg'],
          ['slr', 'round'],
          ['slr', 'nines'],
          ['byr', 'accept'],
          ['byr', 'reject'],
          ['byr', 'con'],
          ['byr', 'msg'],
          ['byr', 'round'],
          ['byr', 'nines']]


def get_minibatch(d, idx):
    data = {}
    data['y'] = torch.from_numpy(d['y'][idx])
    data['x_fixed'] = torch.from_numpy(d['x_fixed'][idx,:])
    if 'x_time' in d:
        data['x_time'] = torch.from_numpy(d['x_time'][idx,:,:])
        data['x_time'] = rnn.pack_padded_sequence(data['x_time'], 
            torch.sum(data['y'] > -1, dim=1), 
            batch_first=True, enforce_sorted=False)
    return data


def get_batches(d, randomize=True):
    N = np.shape(d['x_fixed'])[0]
    v = [i for i in range(N)]
    if randomize:
        np.random.shuffle(v)
    return np.array_split(v, 1 + N // MBSIZE)


def train_model(simulator, outfile, train, test):
    # loop over epochs, record log-likelihood
    for i in range(EPOCHS):
        t0 = dt.now()

        # split training data into minibatches
        batches = get_batches(train)

        # loop over batches
        lnL_train = 0
        for idx in batches:
            # create dictionary and run batch
            data = get_minibatch(train, idx)
            lnL_train += simulator.run_batch(data, idx)
            print(lnL_train)

        # training duration
        dur = np.round((dt.now() - t0).seconds)

        # calculate log-likelihood on holdout
        batches = get_batches(test, randomize=False)
        lnL_test = 0
        for idx in batches:
            data = get_minibatch(test, idx)
            loss, _ = simulator.evaluate_loss(data, train=False)
            lnL_test -= loss.item()

        # write to file
        f = open(outfile, 'a')
        f.write('%d,%d,%.4f,%.4f\n' % (i+1, dur, lnL_train, lnL_test))
        f.close()


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--id', type=int, help='Experiment ID.')
    args = parser.parse_args()

    # model and outcome
    model, outcome = MODELS[args.id-1]
    print('%s: %s' % (model, outcome))

    # input prefix
    if torch.cuda.is_available():
        prefix = '/data/eBay'
    else:
        prefix = 'data'

    # load sizes
    sizefile = '%s/inputs/sizes/%s_%s.pkl' % (prefix, model, outcome)
    sizes = pickle.load(open(sizefile, 'rb'))
    params = {'ff_layers': LAYERS, 'rnn_layers': LAYERS, 
              'ff_hidden': HIDDEN, 'rnn_hidden': HIDDEN, 'K': 10}

    # initialize neural net
    simulator = Simulator(model, outcome, params, sizes)
    print(simulator.net)

    # load data
    print('Loading data')
    trainfile = '%s/inputs/train_models/%s_%s.gz' % (prefix, model, outcome)
    train = load(trainfile)

    testfile = '%s/inputs/train_rl/%s_%s.gz' % (prefix, model, outcome)
    test = load(testfile)

    # create outfile
    outfile = 'outputs/cluster/%s_%s.csv' % (model, outcome)
    f = open(outfile, 'w')
    f.write('epoch,seconds,lnL_train,lnL_holdout\n')
    f.close()

    # train model
    print('Training: %d epochs.' % EPOCHS)
    train_model(simulator, outfile, train, test)
    