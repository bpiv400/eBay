import sys
import os
import pickle
import torch
import torch.nn.utils.rnn as rnn
import numpy as np
import pandas as pd
from datetime import datetime as dt
from models import Simulator
from utils import *
from constants import *


def process_mb(simulator, optimizer, d):
    # zero the gradient
    optimizer.zero_grad()

    # forward pass to get model output
    p, a, b = simulator(d['x_fixed'], d['x_offer'])

    # calculate total loss over minibatch
    criterion = simulator.get_criterion()
    if 'g' in d:
        loss = criterion(p, a, b, d['y'], d['g'])
    else:
        loss = criterion(p, d['y'])

    # backward pass and update model weights
    loss.backward()
    optimizer.step()

    # return gamma and lnL
    g = get_gamma(a.detach(), b.detach(), d['y']) if 'g' in d else None
    return -loss.item(), g


def prepare_batch(train, g, idx):
    batch = {}
    batch['y'] = train['y'][:, idx]
    batch['x_fixed'] = train['x_fixed'][:, idx, :]
    batch['x_offer'] = rnn.pack_padded_sequence(
        train['x_offer'][:, idx, :], train['turns'][idx])
    if g is not None:
        batch['g'] = g[:, idx, :]
    return batch


def run_epoch(simulator, optimizer, train, g, mbsize):
    lnL = 0
    indices = get_batch_indices(train['y'].size()[1], mbsize)
    for i in range(len(indices)):
        idx = indices[i]
        batch = prepare_batch(train, g, idx)
        lnL_i, g_i = process_mb(simulator, optimizer, batch)
        lnL += lnL_i
        if g is not None:
            g[:, idx, :] = g_i
    return lnL / torch.sum(train['turns']).item(), g


def train_model(simulator, train, params):
    # initialize lnL vector
    time0 = dt.now()
    lnL= np.full(EPOCHS, np.nan)

    # initialize gamma
    g = initialize_gamma(train['y'].size(), simulator.get_K())

    # initialize optimizer
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params.lr)

    for epoch in range(EPOCHS):
        start = dt.now()

        # iterate over minibatches
        lnL[epoch], g = run_epoch(simulator, optimizer, train, g, params.mbsize)

        # update loss history
        print('Epoch %d: %dsec. lnL: %1.4f.' %
            (epoch+1, (dt.now() - start).seconds, lnL[epoch]))

    # return loss history and duration
    return {'lnL': lnL, 'duration': dt.now() - time0}


if __name__ == '__main__':
    # extract parameters from command line
    args = get_args()

    # extract parameters from spreadsheet
    params = pd.read_csv(EXP_PATH, index_col=0, dtype=TYPES).loc[args.id]
    print(params)

    # training data
    train = process_inputs(params.model)

    # initialize neural net
    N_fixed = train['x_fixed'].size()[2]
    N_offer = train['x_offer'].size()[2]
    simulator = Simulator(N_fixed, N_offer, params)
    print(simulator)
    sys.stdout.flush()

    # check gradient
    if args.gradcheck:
        print('Checking gradient')
        check_gradient(simulator, train)

    # training loop
    print('Training')
    output = train_model(simulator, train, params)

    # save simulator parameters and other output
    prefix = BASEDIR + '%s/%d' % (args.model, args.id)
    torch.save(simulator.state_dict(), prefix + '.pt')
    pickle.dump(output, open(prefix + '.pkl', 'wb'))
