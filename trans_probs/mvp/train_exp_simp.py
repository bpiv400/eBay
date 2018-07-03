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
        self.fc2 = nn.Linear(num_units, num_classes)
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


def get_prep_type(exp_name):
    '''
    Description: uses the experiment name to determine
    where to load data from
    '''
    prep_type = exp_name[len(exp_name) - 1]
    if int(prep_type) == 1:
        prep_type = 'mvp1'
    elif int(prep_type) == 2:
        prep_type = 'mvp2'
    elif int(prep_type) == 3:
        prep_type = 'mvp3'
    elif int(prep_type) == 4:
        prep_type = 'mvp4'
    return prep_type


def get_class_series(df, resp_turn):
    '''
    Description: replaces the prediction variable with
    class ids from 0 to n-1 where n = number of classes
    and returns the data frame with the converted column
    and a series where index = original category
    and values give the new index, which can be used as a 'map'
    for outputted columns if necessary later

    Input:
        df: data frame containing 'resp_turn' col
        resp_turn: string identifying the name of the prediction
        variable in df
    Output: tuple of (data frame, class series 'map')
    '''
    raw_classes = np.unique(df[resp_turn].values)
    class_ids = np.arange(0, len(raw_classes))
    class_series = pd.Series(class_ids, index=raw_classes)
    return class_series


def get_resp_turn(turn):
    if len(turn) != 2:
        raise ValueError('turn should be two 2 characters')
    turn_num = turn[1]
    turn_type = turn[0]
    turn_num = int(turn_num)
    if turn_type == 'b':
        resp_turn = 's' + str(turn_num)
    else:
        resp_turn = 'b' + str(turn_num + 1)
    resp_col = 'offr_' + resp_turn
    return resp_col


if __name__ == '__main__':
    # set number of hidden units per layer
    num_units = 30
    # set the number of mini batches
    batch_size = 32
    # set the number of batches as a fraction of
    # training examples
    # parse parameters
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    parser.add_argument('--turn', action='store', type=str)
    parser.add_argument('--exp', action='store', type=str)
    parser.add_argument('--batches', action='store', type=float)

    args = parser.parse_args()
    name = args.name
    exp_name = args.exp.strip()
    turn = args.turn.strip()
    num_batches = args.batches

    if len(turn) != 2:
        raise ValueError('turn should be two 2 characters')
    turn_num = turn[1]
    turn_type = turn[0]
    turn_num = int(turn_num)

    # load data
    filename = name + '_concat.csv'
    print('Loading Data')
    sys.stdout.flush()
    prep_type = get_prep_type(exp_name)
    load_loc = 'data/exps/%s/normed/%s_concat_%s.csv' % (prep_type, name, turn)
    df = pd.read_csv(load_loc)
    print('Done Loading')
    sys.stdout.flush()

    resp_col = get_resp_turn(turn)

    # grab target
    class_series = get_class_series(df, resp_col)

    targ = df[resp_col].values
    # drop response variable
    df.drop(columns=resp_col, inplace=True)

    # creating a dictionary which maps all column names to
    # indices in the input data matrix
    cols = df.columns
    colix = {}
    counter = 0
    for i in cols:
        colix[i] = counter
        counter = counter + 1

    data = df.values
    del df

    # calculating parameter values for the model from data
    num_feats = data.shape[1]
    num_batches = int(data.shape[0] / batch_size * num_batches)
    classes = class_series.index.values
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
        if i % 500 == 0 and i > 0:
            print('Batch: %d of %d' % (i, num_batches))
            print(loss)

        sys.stdout.flush()
        optimizer.zero_grad()
        # extract label from batch
        sample_inds = np.random.random_integers(
            0, (targ.size - 1), size=batch_size)

        sample_input = data[sample_inds, :]
        print(sample_input.isna.any())
        sample_input = torch.from_numpy(sample_input).float()
        sample_targ = targ[sample_inds]
        sample_targ = torch.from_numpy(sample_targ).float()
        sample_targ = sample_targ.view(-1, 1)
        output = net(sample_input)
        if i % 2000 == 0 and i > 0:
            print('resid: %.2f' % (sample_targ - output).mean().view(-1, 1))
            # print(torch.cat([sample_targ.view(-1, 1), output.view(-1, 1),
            #                ((sample_targ - output)**2).view(-1, 1)], 1))
            print(output)
            print(sample_inds[0:5])
            print(sample_input[0:5, :].sum(1))

        loss = criterion(output, sample_targ)
        loss.backward()
        optimizer.step()
    # practice saving the model
    # save the model
    print('Done Training')
    sys.stdout.flush()
    print('Pickling')
    sys.stdout.flush()
    torch.save(net.state_dict(), 'models/exps/%s/model_%s.pth.tar' %
               (exp_name, turn))

    loss_pickle = open('models/exps/%s/loss_%s.pickle' %
                       (exp_name, turn), 'wb')
    pickle.dump(loss_hist, loss_pickle)
    loss_pickle.close()

    class_series.to_csv(
        'models/exps/%s/class_series_%s.csv' % (exp_name, turn))

    feat_dict_pick = open('models/exps/%s/featdict_%s.pickle' %
                          (exp_name, turn), 'wb')
    pickle.dump(colix, feat_dict_pick)
    feat_dict_pick.close()
