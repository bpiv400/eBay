import sys, os, argparse, pickle
import torch, torch.nn.utils.rnn as rnn
import numpy as np
from datetime import datetime as dt
from models import *
from utils import *

OPTIMIZER = torch.optim.Adam
CRITERION = BetaMixtureLoss.apply

def process_mb(simulator, optimizer, y_i, x_fixed_i, x_offer_i):
    # zero the gradient
    optimizer.zero_grad()

    # forward pass to get parameters of beta mixture
    p, a, b = simulator(x_fixed_i, x_offer_i)

    # calculate total loss over minibatch
    loss = CRITERION(p, a, b, y_i)

    # update parameters
    loss.backward()
    optimizer.step()

    return loss


def train(simulator, optimizer, x_offer, x_fixed, y, turns, mbsize, tol):
    loss_hist = np.array([np.inf]) # loss in each epoch
    N_turns = torch.sum(turns).item() # total number of turns

    print('Training')
    while True:
        start = dt.now()
        epoch = len(loss_hist)
        if epoch > 1 and loss_hist[epoch-2] - loss_hist[epoch-1] < tol:
            break

        # iterate over minibatches
        batches, indices = get_batch_indices(y.size()[1], mbsize)
        loss = 0
        for i in range(batches):
            idx = indices[i]
            x_offer_i = rnn.pack_padded_sequence(x_offer[:, idx, :], turns[idx])

            loss += process_mb(simulator, optimizer, y[:, idx, :],
                                x_fixed[:, idx, :], x_offer_i)

        #compute_example(simulator)

        # update loss history
        loss_hist = np.append(loss_hist, torch.div(loss, N_turns).item())
        print('Epoch %d: %1.6f (%s)' % (epoch, loss_hist[epoch], dt.now() - start))
        sys.stdout.flush()

    return loss_hist


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--mbsize', default=128, type=int,
        help='Number of samples per minibatch.')
    parser.add_argument('--tol', default=1e-6, type=float,
        help='Stopping criterion: improvement in log-likelihood.')
    parser.add_argument('--hidden', default=100, type=int,
        help='Number of nodes in each hidden layer.')
    parser.add_argument('--dropout', default=0.5, type=float,
        help='Dropout rate.')
    parser.add_argument('--step_size', default=0.001, type=float,
        help='Learning rate for optimizer.')
    parser.add_argument('--layers', default=2, type=int,
        help='Number of recurrent layers.')
    parser.add_argument('--check_grad', default=False, type=bool,
        help='Boolean flag to check gradients.')
    args = parser.parse_args()

    # load data and extract comonents
    print('Loading Data')
    sys.stdout.flush()
    data = pickle.load(open('../../data/chunks/0_simulator.pkl', 'rb'))
    x_offer, x_fixed, y = [data[i] for i in ['x_offer','x_fixed','y']]
    turns = 3 - torch.sum(torch.isnan(y[:,:,0]), 0)
    sys.stdout.flush()
    del data

    # create neural net
    N_fixed = x_fixed.size()[2]
    N_offer = x_offer.size()[2]
    simulator = Simulator(N_fixed, N_offer, args.hidden, args.layers, args.dropout)
    print(simulator)
    sys.stdout.flush()

    # check gradient
    if args.check_grad:
        print('Checking gradient')
        check_gradient(simulator, CRITERION, x_fixed, x_offer, y, turns)

    # initialize optimizer
    optimizer = OPTIMIZER(simulator.parameters(), lr=args.step_size)

    # training loop iteration
    loss = train(simulator, optimizer, x_offer, x_fixed, y, turns,
        args.mbsize, args.tol)

    # save simulator parameters and loss history
    torch.save(simulator.state_dict(), '../../data/simulator/0.pth.tar')
    pickle.dump(loss[1:], open('../../data/simulator/0_loss.pkl', 'wb'))