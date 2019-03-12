import sys, os, argparse, pickle
import torch, torch.nn.utils.rnn as rnn
import numpy as np, pandas as pd
from datetime import datetime as dt
from models import *
from utils import *

#INPUT_PATH = '../../data/train/simulator_input.pkl'
INPUT_PATH = '../../data/chunks/1_simulator.pkl'
EXP_PATH = 'experiments.csv'
OPTIMIZER = torch.optim.Adam


def process_mb(simulator, optimizer, criterion, y_i, x_fixed_i, x_offer_i):
    # zero the gradient
    optimizer.zero_grad()

    # forward pass to get model output
    theta = simulator(x_fixed_i, x_offer_i)

    # calculate total loss over minibatch
    loss = criterion(theta, y_i)

    # update parameters
    loss.backward()
    optimizer.step()

    return loss


def train(simulator, optimizer, criterion, x_offer, x_fixed, y, turns, mbsize, epochs):
    loss_hist = np.array([np.inf]) # loss in each epoch
    N_turns = torch.sum(turns).item() # total number of turns

    print('Training')
    for epoch in range(1,epochs+1):
        start = dt.now()

        # iterate over minibatches
        batches, indices = get_batch_indices(y.size()[1], mbsize)
        loss = 0
        for i in range(batches):
            idx = indices[i]
            x_offer_i = rnn.pack_padded_sequence(x_offer[:, idx, :], turns[idx])

            loss += process_mb(simulator, optimizer, criterion,
                        y[:, idx], x_fixed[:, idx, :], x_offer_i)

        # update loss history
        loss_hist = np.append(loss_hist, torch.div(loss, N_turns).item())
        print('Epoch %d: %1.6f (%s)' % (epoch, loss_hist[epoch], dt.now() - start))
        #compute_example(simulator)
        sys.stdout.flush()

    return loss_hist


if __name__ == '__main__':
    # extract parameters from command line
    args = get_args()

    # extract parameters from spreadsheet
    params = pd.read_csv(EXP_PATH, index_col=0).loc[args.id]

    # output layer and loss function
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

    # load data and extract components
    print('Loading Data')
    data = pickle.load(open(INPUT_PATH, 'rb'))
    data['y'] = data['y'][args.model]
    data['x_offer'] = expand_x_offer(data['x_offer'], args.model)
    data['x_fixed'] = get_x_fixed(data['T'])
    x_offer, x_fixed, y, turns = convert_to_tensors(data)
    del data

    # create neural net
    N_fixed = x_fixed.size()[2]
    N_offer = x_offer.size()[2]
    simulator = Simulator(N_fixed, N_offer, params.hidden, N_out,params.layers, params.dropout, params.lstm)
    print(simulator)
    sys.stdout.flush()

    # check gradient
    if args.gradcheck:
        print('Checking gradient')
        check_gradient(simulator, criterion, x_fixed, x_offer, y, turns)

    # initialize optimizer
    optimizer = OPTIMIZER(simulator.parameters(), lr=params.lr)

    # training loop iteration
    loss_hist = train(simulator, optimizer, criterion,
        x_offer, x_fixed, y, turns, params.mbsize, params.epochs)

    # save simulator parameters and loss history
    path_prefix = '../../data/simulator/%d_' % args.num
    torch.save(simulator.state_dict(), path_prefix + 'pth.tar')
    pickle.dump(loss_hist[1:], open(path_prefix + 'loss.pkl', 'wb'))