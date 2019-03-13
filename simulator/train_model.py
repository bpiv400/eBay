import sys, os, argparse, pickle
import torch, torch.nn.utils.rnn as rnn
import numpy as np, pandas as pd
from datetime import datetime as dt
from models import *
from utils import *

INPUT_PATH = {name: '../../data/%s/simulator_input.pkl' % name
    for name in ['train', 'test']}
EXP_PATH = 'experiments.csv'
OPTIMIZER = torch.optim.Adam


def process_mb(simulator, optimizer, criterion, train, idx):
    # zero the gradient
    optimizer.zero_grad()

    # variables
    y_i = train['y'][:, idx]
    x_fixed_i = train['x_fixed'][:, idx, :]
    x_offer_i = rnn.pack_padded_sequence(
            train['x_offer'][:, idx, :], train['turns'][idx])

    # forward pass to get model output
    theta = simulator(x_fixed_i, x_offer_i)

    # calculate total loss over minibatch
    loss = criterion(theta, y_i)

    # update parameters
    loss.backward()
    optimizer.step()


def get_avg_ll(simulator, criterion, d):
    x_offer = rnn.pack_padded_sequence(d['x_offer'], d['turns'])
    ll = -criterion(simulator(d['x_fixed'], x_offer), d['y'])
    return ll.item() / torch.sum(d['turns']).item()


def run_epoch(simulator, optimizer, criterion, train, mbsize, N_threads):
    batches, indices = get_batch_indices(N_threads, mbsize)
    for i in range(batches):
        process_mb(simulator, optimizer, criterion, tensors['train'], indices[i])


def train_model(simulator, optimizer, criterion, tensors, mbsize, epochs):
    ll = {'train': np.full(epochs, np.nan), 'test': np.full(epochs, np.nan)}
    N_threads = tensors['train']['y'].size()[1]
    for epoch in range(1,epochs+1):
        start = dt.now()

        # iterate over minibatches
        run_epoch(simulator, optimizer, criterion, tensors['train'], mbsize, N_threads)

        # calculate average log-likelihood
        for key, val in tensors.items():
            ll[key][epoch-1] = get_avg_ll(simulator, criterion, val)

        print('Epoch %d: %s' % (epoch, dt.now() - start))
        print('\tAvg lnL in train: %1.6f' % ll['train'][epoch-1])
        print('\tAvg lnL in test: %1.6f' % ll['test'][epoch-1])
        #compute_example(simulator)
        sys.stdout.flush()

    return ll


def load_data(model):
    tensors = {}
    for name in ['train', 'test']:
        print('Loading ' + name + ' data')
        data = pickle.load(open(INPUT_PATH[name], 'rb'))
        data['y'] = trim_threads(data['y'][model])
        data['x_fixed'] = get_x_fixed(data['T'])
        data['x_offer'] = expand_x_offer(data['x_offer'], model)
        tensors[name] = convert_to_tensors(data)
        if name == 'train':
            N_fixed = tensors[name]['x_fixed'].size()[2]
            N_offer = tensors[name]['x_offer'].size()[2]
    return tensors, N_fixed, N_offer

if __name__ == '__main__':
    # extract parameters from command line
    args = get_args()

    # extract parameters from spreadsheet
    params = pd.read_csv(EXP_PATH, index_col=0).loc[args.id]

    # output layer and loss function
    print('Model: ' + args.model)
    if args.model == 'delay':
        N_out = 3
        criterion = DelayLoss.apply
    elif args.model == 'con':
        N_out = 5
        criterion = ConLoss.apply
    elif args.model == 'msg':
        N_out = 1
        criterion = MsgLoss.apply
    else:
        print('Error:', args.model, 'is not a valid model.')
        exit()

    # load data
    tensors, N_fixed, N_offer = load_data(args.model)

    # initialize neural net and optimizer
    simulator = Simulator(N_fixed, N_offer, params.hidden, N_out,
        params.layers, params.dropout, params.lstm)
    print(simulator)
    optimizer = OPTIMIZER(simulator.parameters(), lr=params.lr)
    sys.stdout.flush()

    # check gradient
    if args.gradcheck:
        print('Checking gradient')
        check_gradient(simulator, criterion, tensors['train'])

    # training loop
    print('Training')
    ll = train_model(simulator, optimizer, criterion, tensors, params.mbsize, params.epochs)

    # save simulator parameters and loss history
    path_prefix = '../../data/simulator/%d_' % args.num
    torch.save(simulator.state_dict(), path_prefix + 'pth.tar')
    pickle.dump(loss_hist[1:], open(path_prefix + 'loss.pkl', 'wb'))