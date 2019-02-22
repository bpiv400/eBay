import sys, os, argparse, pickle
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd
from datetime import datetime as dt
from models import Model

CRITERION = nn.MSELoss(reduction='sum')
OPTIMIZER = optim.ADAM()

def processMinibatch(model, optimizer, x_offer, x_fixed, y, k, batch_ind_i):
    # zero the gradient
    optimizer.zero_grad()

    # subset and sort data
    k_i, sort_ind = torch.sort(k[batch_ind_i], dim=0, descending=True)
    indices = [batch_ind_i[sort_ind[j]] for j in range(len(sort_ind))]
    x_offer_i = x_offer[:, indices, :]
    x_fixed_i = x_fixed[:, indices, :]
    y_i = y[:, indices]

    # generate packed sequence
    x_offer_i = torch.nn.utils.rnn.pack_padded_sequence(x_offer_i,k_i)

    # forward pass
    y_hat_i = model(x_fixed_i, x_offer_i)

    # calculate total loss over minibatch and update gradients
    loss = criterion(y_hat_i, y_i)

    # update parameters
    loss.backward()
    optimizer.step()

    return loss


def getBatchIndices(samples, mbsize):
    # create matrix of randomly sampled minibatch indices
    shuffled = random.sample(range(samples), samples)
    batches = math.ceil(samples / mbsize)
    batchInd = [shuffled[mbsize*i:mbsize*(i+1)] for i in range(batches)]
    return batches, batchInd


def train(mbsize, tol, model, criterion, optimizer, x_offer, x_fixed, y, k):
    loss_hist = [float('Inf')] # loss in each epoch
    N_seq = k.sum().item() # number of sequences
    while True:
        start = dt.now()
        epoch = len(loss_hist)
        if epoch > 1 and loss_hist[epoch-2] - loss_hist[epoch-1] < tol:
            break

        # iterate over minibatches
        batches, indices = getBatchIndices(x_offer.shape[1], mbsize)
        loss = 0
        for i in range(batches):
            # get error for minibatch, back prop, and update model
            loss += processMinibatch(model, criterion, optimizer, x_offer, x_fixed, y, k, indices[i])

        # update loss history
        loss_hist.append(loss.item() / N_seq)
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
        help='Stopping criterion: improvement in average loss.')
    parser.add_argument('--hidden', default=100, type=int,
        help='Number of nodes in each hidden layer.')
    parser.add_argument('--dropout', default=0.01, type=float,
        help='Dropout rate.')
    args = parser.parse_args()

    # load data and extract comonents
    print('Loading Data')
    sys.stdout.flush()
    data = pickle.load('../data/exps/rnn2_cat/train_data.pickle', 'rb')

    x_offer = torch.from_numpy(data['offr_vals']).float()
    x_fixed = torch.from_numpy(data['const_vals']).float()
    y = torch.from_numpy(data['target_vals']).float()
    k = torch.from_numpy(data['length_vals']).long()
    print('Done Loading')
    sys.stdout.flush()
    del data

    # calculate parameter values for the model from data
    N_fixed = x_fixed.shape[2]
    N_offer = x_offer.shape[2]

    # create LSTM and optimizer
    model = Model(N_fixed, N_offer, args.hidden, args.dropout)
    optimizer = OPTIMIZER(model.parameters())
    print(model)
    sys.stdout.flush()

    # training loop iteration
    loss = train(args.mbsize, args.tol, model, optimizer, x_offer, x_fixed, y, k)

    # save model parameters and loss history
    torch.save(model.state_dict(), 'data/model.pth.tar')
    pickle.dump(loss[1:], open('data/loss.pkl', 'wb'))