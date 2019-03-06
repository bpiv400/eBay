import sys, os, argparse, pickle
import torch, torch.nn.utils.rnn as rnn
import numpy as np
from datetime import datetime as dt
from models import *
from utils import *

INPUT_PATH = '../../data/train/simulator_input.pkl'
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


def train(simulator, optimizer, x_offer, x_fixed, y, turns, mbsize, epochs):
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

            loss += process_mb(simulator, optimizer, y[:, idx, :],
                                x_fixed[:, idx, :], x_offer_i)

        # update loss history
        loss_hist = np.append(loss_hist, torch.div(loss, N_turns).item())
        print('Epoch %d: %1.6f (%s)' % (epoch, loss_hist[epoch], dt.now() - start))
        compute_example(simulator)
        sys.stdout.flush()

    return loss_hist


if __name__ == '__main__':
    # extract parameters from command line
    args = get_args()

    # load data and extract comonents
    print('Loading Data')
    sys.stdout.flush()
    data = pickle.load(open(INPUT_PATH, 'rb'))
    x_offer, x_fixed, y, turns = convert_to_tensors(data)
    sys.stdout.flush()
    del data

    # create neural net
    N_fixed = x_fixed.size()[2]
    N_offer = x_offer.size()[2]
    simulator = Simulator(N_fixed, N_offer,
        args.hidden, args.layers, args.dropout, args.is_lstm)
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
        args.mbsize, args.epochs)

    # save simulator parameters and loss history
    torch.save(simulator.state_dict(), '../../data/simulator/0.pth.tar')
    pickle.dump(loss[1:], open('../../data/simulator/0_loss.pkl', 'wb'))