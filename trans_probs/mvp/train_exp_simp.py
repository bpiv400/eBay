import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import pickle
import math
import sys
import os


class Net(nn.Module):
    def __init__(self, num_feat, num_units, num_classes, classes):
        super(Net, self).__init__()
        # print('feats %d' % num_feat)
        # print('units %d' % num_units)
        # print('num_classes %d' % num_classes)
        self.fc1 = nn.Linear(num_feat, num_units)
        self.fc2 = nn.Linear(num_units, num_units)
        self.expect = torch.from_numpy(classes).float()
        self.expect.requires_grad_(False)
        self.expect = self.expect.view(-1, 1)

    def forward(self, x):
        x = self.fc1(x)
        # print(x.size())
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        # print(x)
        # print(x.size())
        # print(self.expect.size())
        x = torch.mm(x, self.expect)
        return x


if __name__ == '__main__':
    # set number of hidden units per layer
    num_units = 30
    # set the number of mini batches
    batch_size = 32
    # set the number of batches as a fraction of
    # training examples
    # I'm thinking of this as "epochs"
    num_batches = 5

    # parse parameters
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    parser.add_argument('--turn', action='store', type=str)
    parser.add_argument('--exp', action='store', type=str)
    args = parser.parse_args()
    name = args.name
    exp_name = args.exp.strip()
    turn = args.turn.strip()
    if len(turn) != 2:
        raise ValueError('turn should be two 2 characters')
    turn_num = turn[1]
    turn_type = turn[0]
    turn_num = int(turn_num)

    # load data
    filename = name + '_concat.csv'
    print('Loading Data')
    sys.stdout.flush()
    df = pd.read_csv('data/curr_exp/%s/%s' % (turn, filename))
    print('Done Loading')
    sys.stdout.flush()
    # grab response variable
    if turn_type == 'b':
        resp_turn = 's' + str(turn_num)
    else:
        resp_turn = 'b' + str(turn_num + 1)
    resp_col = 'offr_' + resp_turn
    targ = df[resp_col].values
    df.drop(columns=resp_col, inplace=True)
    cols = df.columns
    colix = {}
    counter = 0
    for i in cols:
        colix[i] = counter
        counter = counter + 1

    data = df.values
    del df

    num_feats = data.shape[1]
    num_batches = int(data.shape[0] / batch_size * num_batches)
    classes = np.unique(targ)
    num_classes = classes.size

    net = Net(num_feats, num_units, num_classes, classes)

    # some debugging nonsense
    # print(list(net.parameters()))
    # print(len(list(net.parameters())))
    # params = list(net.parameters())
    # print(params[2].size())
    # print(dict(net.named_parameters()))

    # set loss function
    # use MSE for now
    criterion = nn.MSELoss(size_average=True, reduce=True)
    optimizer = optim.Adam(net.parameters(), weight_decay=math.pow(10, -5))
    loss_hist = []
    loss = criterion
    print('Training')
    for i in range(num_batches):
        print('Batch: %d of %d' % (i, num_batches))
        sys.stdout.flush()
        optimizer.zero_grad()
        # extract label from batch
        sample_inds = np.random.random_integers(
            0, (targ.size - 1), size=batch_size)
        sample_input = data[sample_inds, :]
        sample_input = torch.from_numpy(sample_input).float()
        sample_targ = targ[sample_inds]
        sample_targ = torch.from_numpy(sample_targ).float()
        sample_targ = sample_targ.view(-1, 1)
        output = net(sample_input)
        loss = criterion(output, sample_targ)
        loss_hist.append(loss)
        loss.backward()
        optimizer.step()
    # practice saving the model
    # save the model
    print('Done Training')
    sys.stdout.flush()
    print('Pickling')
    sys.stdout.flush()
    torch.save(net.state_dict(), 'data/models/%s/model_%s.pth.tar' %
               (exp_name, turn))

    loss_pickle = open('data/models/%s/loss_%s.pickle' %
                       (exp_name, turn), 'wb')
    pickle.dump(loss_hist, loss_pickle)
    loss_pickle.close()

    feat_dict_pick = open('data/models/%s/featdict_%s.pickle' %
                          (exp_name, turn), 'wb')
    pickle.dump(colix, feat_dict_pick)
    feat_dict_pick.close()
