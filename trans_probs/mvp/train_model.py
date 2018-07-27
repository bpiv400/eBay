import sys
import os
sys.path.append(os.path.abspath('repo/trans_probs/mvp/'))

from models import *
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
from datetime import datetime as dt
import re
import copy


def get_model_class(exp_name):
    '''
    Description: Uses experiment name to grab the associated model
    from the models module and aliases it as net
    Input: string giving experiment name
    Output: class of model to be trained
    '''
    if 'cross' in exp_name:
        if 'simp' in exp_name:
            print('Model: Cross Simp')
            net = Cross_simp
        elif 'bet' in exp_name:
            print('Model: Cross between')
            net = Cross_simp
        else:
            net = Cross_comp
            print('Model cross comp')
    else:
        if 'simp' in exp_name:
            print('model exp simp')
            net = Exp_simp
        elif 'bet' in exp_name:
            print('model exp between')
            net = Exp_simp
        else:
            print('model exp comp')
            net = Exp_comp
    return net


def get_class_series(midpoints):
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
    Output: tuple of(data frame, class series 'map')
    '''
    raw_classes = midpoints
    class_ids = np.arange(0, len(raw_classes))
    class_series = pd.Series(class_ids, index=raw_classes)
    return class_series


def get_resp_offr(turn):
    '''
    Description: Determines the name of the response column given the name of the last observed turn
    '''
    turn_num = turn[1]
    turn_type = turn[0]
    if turn != 'start_price_usd':
        turn_num = int(turn_num)
    if turn_type == 'b':
        resp_turn = 's' + str(turn_num)
    elif turn == 'start_price_usd':
        resp_turn = 'b0'
    elif turn_type == 's':
        resp_turn = 'b' + str(turn_num + 1)
    resp_col = 'offr_' + resp_turn
    return resp_col


def get_resp_time(turn):
    '''
    Description: Determines the name of the response column given the 
    name of the last observed turn
    for time models
    '''
    turn_num = turn[1]
    turn_type = turn[0]
    turn_num = int(turn_num)
    if turn_type == 'b':
        resp_turn = 's' + str(turn_num)
    elif turn == 'start_price_usd':
        resp_turn = None
    elif turn_type == 's':
        resp_turn = 'b' + str(turn_num + 1)
    resp_col = 'time_%s' % resp_turn
    return resp_col


def get_prep_type(exp_name):
    '''
    Description: uses the experiment name to determine
    where to load data from
    '''
    # grab prep type (the integer contained in the
    # experiment name)
    prep_type = re.findall(r'\d+', exp_name)
    prep_type = prep_type[0]
    prep_type = int(prep_type)

    if int(prep_type) == 1:
        if 'norm' in exp_name:
            prep_type = 'norm1'
        else:
            prep_type = 'mvp1'
    elif int(prep_type) == 2:
        if 'norm' in exp_name:
            prep_type = 'norm2'
        else:
            prep_type = 'mvp2'
    elif int(prep_type) == 3:
        prep_type = 'mvp3'
    elif int(prep_type) == 4:
        prep_type = 'mvp4'
    print('Prep type: %s' % prep_type)
    return prep_type


def get_num_units(exp_name):
    '''
    Description: gets the number of units in each
    nonlinear activation layer using the experiment name
    '''
    if 'simp' in exp_name:
        num_units = 30
    elif 'bet' in exp_name:
        num_units = 100
    else:
        num_units = 100
    print('Num units: %s' % num_units)
    return num_units


def get_resp_turn_classes(df, resp_turn, class_series):
    '''
    Description: Converts data frame with raw class
    target to indices representing class index for use
    in classifier NN
    '''
    org_col = df[resp_turn].values
    converted_classes = class_series.loc[org_col].values
    df[resp_turn] = converted_classes
    return df


def update_loss_hist(loss_hist, recent_hist, curr_batch=None, num_batches=None):
    if num_batches is not None:
        print('Batch: %d of %d' % (curr_batch, num_batches))
    else:
        print('Batche: %d' % curr_batch)
    # convert recent hist to numpy array
    recent_hist = recent_hist.detach().cpu().numpy()
    # recent_hist bug
    print('Recent hist bug: %r' % np.any(recent_hist == 0))
    # get the mean of the recent history
    recent_hist = np.mean(recent_hist)
    # append this to the long-run loss history
    loss_hist.append(recent_hist)
    print('Recent Loss Mean: %.2f' % recent_hist)
    sys.stdout.flush()
    return loss_hist


def process_minibatch(num_samps, net, optimizer, data, targ, batch_size, device, criterion):
    # grab sample indices
    sample_inds = torch.randint(
        low=0, high=num_samps, size=(batch_size,), device=device).long()

    # grab sample data
    sample_input = data[sample_inds, :]

    # grab corresponding target
    sample_targ = targ[sample_inds]

    # get sample output
    if 'cross' in exp_name:
        output = net(sample_input)
    else:
        output = net(sample_input).view(-1)

    # calculate loss and backpropogate gradient
    loss = criterion(output, sample_targ)
    loss.backward()
    # update parameters
    optimizer.step()
    return loss


def process_valid(net, valid_data, valid_targ, criterion):
    # get sample output
    if 'cross' in exp_name:
        output = net(valid_data)
    else:
        output = net(valid_data).view(-1)

    # calculate loss
    loss = criterion(output, valid_targ)
    # return as numpy
    loss = loss.detach().cpu().numpy()
    return loss


def is_valid_dec(valid_hist, hist_len):
    '''
    Description: Returns whether the validation error history
    DOES not satisfy the stopping criterion
    Inputs:
        valid_hist: a list containing numpy.float where each
        element gives the validation error calculated at some 
        step
        hist_len: integer giving the number of consecutive 
        validation error increases required to satisfy the stopping
        criterion
    Output: boolean giving whether training should continue
    '''
    # get the number of iterations
    num_it = len(valid_hist)
    # if there is not sufficient history for the validation error
    # check, vacuously the stopping criterion fails
    if num_it < (hist_len + 1):
        return True
    else:
        # set the flag for whether any decreases have been wittnessed
        any_dec_flag = False
        # iterate over the loss history
        for i in range(hist_len):
            # get the current loss
            curr_loss = valid_hist[num_it - (1 + i)]
            # get the previous loss
            prev_loss = valid_hist[num_it - (2 + i)]
            if curr_loss < prev_loss:
                any_dec_flag = True
        # return whether any decreases have been observed in the
        # duration of the history we're interested in
        return any_dec_flag


def batch_count_loop(num_batches, net, optimizer, data, targ, batch_size, device, criterion):
    # get the total number of samples in the training data
    num_samps = data.shape[0]
    # create the initial copy of the recent_history tensory
    recent_hist = torch.zeros(500, device=device)
    # create an index counter to track where loss should be input to recent_hist tensor
    hist_ind = 0
    # create an empty list to store the mean of the last 500 batch losses
    loss_hist = []
    # iterate over the number of batches assigned
    for i in range(1, num_batches):
        # zero the gradient
        optimizer.zero_grad()

        # get error for minibatch, back prop, and update model
        loss = process_minibatch(
            num_samps, net, optimizer, data, targ, batch_size, device, criterion)

        # add loss to the current recent loss history and update the index counter
        recent_hist[hist_ind] = loss
        hist_ind = hist_ind + 1

        # update the long run loss history every 500 iterations
        if i % 500 == 0:
            loss_hist = update_loss_hist(
                loss_hist, recent_hist, i, num_batches)
            # reset recent history and associated index
            recent_hist = torch.zeros(500, device=device)
            hist_ind = 0
    # return the state dictionary for the model after the last update
    return net.state_dict(), loss_hist


def valid_loop(valid_dur, valid_data, valid_targ, hist_len, net, optimizer, data, targ, batch_size, device, criterion):
    # initialize counter for the total number of minibatches
    batch_count = 1
    # get the total number of samples in the training data
    num_samps = data.shape[0]
    # initialize recent history tensor and index counter to track where loss should be added to it
    recent_hist = torch.zeros(500, device=device)
    hist_ind = 0
    # create an empty list to track validation history
    valid_hist = []
    # create an empty list to store the mean of the last 500 batch losses
    loss_hist = []
    # initialize the flag that determines whether validation error is decreasing
    valid_dec = True
    # initialize validation dictionary to contain dictionaries of model parameters so we can revert to last model before
    # error increased
    valid_dict = {}
    # in the dictionary, higher integer keys indicate more recent model parameters
    for i in range(hist_len + 1):
        valid_dict[i] = None

    # iterate while validation error is decreasing
    while valid_dec:
        # zero the gradient
        optimizer.zero_grad()

        # get error for minibatch, back prop, and update model
        loss = process_minibatch(
            num_samps, net, optimizer, data, targ, batch_size, device, criterion)

        # add loss to the current recent loss history and update the index counter
        recent_hist[hist_ind] = loss
        hist_ind = hist_ind + 1

        # update the long run loss history every 500 iterations
        if batch_count % 500 == 0:
            loss_hist = update_loss_hist(
                loss_hist, recent_hist, batch_count)
            # reset recent history and associated index
            recent_hist = torch.zeros(500, device=device)
            hist_ind = 0

        # update valid history every valid_dur iterations
        if batch_count % valid_dur == 0:
            # get loss on the validation set
            valid_loss = process_valid(net, valid_data, valid_targ, criterion)
            valid_hist.append(valid_loss)
            print('Valid Loss %d: %.2f' % (len(valid_hist), valid_loss))
            # move state history back 1 iteration
            for i in range(hist_len):
                valid_dict[i] = copy.deepcopy(valid_dict[i + 1])
            # make the most recent entry equal state dict
            valid_dict[hist_len] = copy.deepcopy(net.state_dict())

            # check the stopping criterion that the validation error has
            # increased on the last two iterations
            valid_dec = is_valid_dec(valid_hist, hist_len)
        batch_count = batch_count + 1
    # exiting while loop indicates stopping criterion has been reached
    # return the model state stored in the oldest recorded dictionary (valid_dict[0])
    return valid_dict[0], loss_hist, valid_hist


def get_optimizer(net, exp_name):
    '''
    Description: Parsing experiment name to extract optimizer
    to be used
    '''
    if 'unr' in exp_name:
        optimizer = optim.Adam(net.parameters(), weight_decay=0)
    else:
        print('regularized')
        optimizer = optimizer = optim.Adam(
            net.parameters(), weight_decay=math.pow(10, -5))
    return optimizer


def main():
    # parse parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    sys.stdout.flush()
    start_time = dt.now()

    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--turn', action='store', type=str)
    parser.add_argument('--exp', action='store', type=str)
    # set the number of batches as a fraction of
    # training examples
    parser.add_argument('--batches', action='store', type=float)
    # set the number of mini batches
    parser.add_argument('--batch_size', action='store', type=int)
    # whether a validation set should be used to determine stopping
    # value should give the percentage of the training sample to be used
    # for the validation step
    parser.add_argument('--valid_size', action='store',
                        type=float, default=None)
    # gives the number of mini-batches that should take place before evaluating
    # the validation set
    parser.add_argument('--dur_valid', action='store', type=int, default=None)
    # gives the number of consecutive error increases required to
    # trigger the stopping criterion
    parser.add_argument('--hist_len', action='store', type=int, default=None)
    # extract parameters
    args = parser.parse_args()
    global exp_name
    exp_name = args.exp.strip()
    turn = args.turn.strip()
    num_batches = args.batches
    batch_size = args.batch_size

    # set a flag for whether the stopping criterion being used is
    # a validation error set
    valid = 'val' in exp_name

    # check whether the validation flag has been activated
    # indicating the stopping criterion should be based on
    # performance of a validation set
    if valid:
        valid_size = args.valid_size
        # check whether a duration (ie number of batches between validation check)
        # has been given, if not, set to 1000
        if args.dur_valid is not None:
            valid_dur = args.dur_valid
        else:
            valid_dur = 1000
        # check whether a history length has been given (ie the number of consecutive)
        # validation error increases required to trigger the stopping criterion
        # if not, set to 1
        if args.hist_len is not None:
            hist_len = args.hist_len
        else:
            hist_len = 1

    # set flag indicating whether this is an offer model
    if 'time' in exp_name:
        offr_mod = False
    else:
        offr_mod = True

    # get response col
    if offr_mod:
        resp_col = get_resp_offr(turn)
    else:
        if turn != 'start_price_usd':
            resp_col = get_resp_time(turn)
        else:
            raise ValueError('Cannot train a model for time until the' +
                             'buyer makes their first offer')

    # get data type
    prep_type = get_prep_type(exp_name)
    # get model class
    Net = get_model_class(exp_name)

    print('Loading Data')
    sys.stdout.flush()

    # load data frame containing training data
    load_loc = 'data/exps/%s/normed/train_concat_%s.csv' % (
        prep_type, turn)
    df = pd.read_csv(load_loc)

    if offr_mod:
        # bin location
        bin_loc = 'data/exps/%s/%s/bins.pickle' % (prep_type, turn)
        # load bins dictionary from pickle
        with open(bin_loc, 'rb') as f:
            bin_dict = pickle.load(f)
        f.close()
        # extract midpoints from bin dictionary
        midpoints = bin_dict['midpoints']
    else:
        # bin location
        bin_loc = 'data/exps/%s/%s/time_bins.pickle' % (prep_type, turn)
        # load bins dictionary from pickle
        with open(bin_loc, 'rb') as f:
            bin_dict = pickle.load(f)
        f.close()
        midpoints = bin_dict['time_midpoints']

    #########################################################
    # TEMPORARY FIX UNTIL ROUNDING ERROR HAS BEEN SOLVED IN BIN
    if 'norm' in exp_name:
        midpoints = np.around(midpoints, 2)
        count = 0
        for point in midpoints:
            count = np.around(count, 2)
            if point != count:
                raise ValueError('Midpoints not rounded correctly')
            count = count + .01

    ##########################################################
    # delete the dictionary itself
    del bin_dict
    print('Done Loading')
    sys.stdout.flush()

    # remove  refrerence columns from data frame
    extra_cols = ['ref_old', 'ref_rec', 'ref_resp']
    # add the response column corresponding to the
    # other kind of model (time_ji for offr_mod)
    # offr_ji otherwise
    if offr_mod:
        if turn != 'start_price_usd':
            extra_cols.append(get_resp_time(turn))
    else:
        extra_cols.append(get_resp_offr(turn))

    for col in extra_cols:
        if col in df:
            print('Dropping %s' % col)
            df.drop(columns=col, inplace=True)

    # get class series (output later)
    class_series = get_class_series(midpoints)

    # convert raw targets to classes if using cross entropy
    if 'cross' in exp_name:
        df = get_resp_turn_classes(df, resp_col, class_series)

    # if stopping criterion is based ontest performance
    # initialize validation set
    if valid:
        # get the number of samples associated with the percentage size
        valid_size = int(len(df.index) * valid_size)
        # valid_size samples from the range of integers in df.index (since this should
        # be a simple step index)
        valid_inds = np.random.random_integers(
            0, (len(df.index) - 1), valid_size)
        # grab the corresponding data then drop the rows from the data frame
        valid_df = df.loc[valid_inds].copy()
        df.drop(index=valid_inds, inplace=True)
        # grab the target, drop the column, then export the rest as a
        # numpy matrix for training
        valid_targ = valid_df[resp_col].values
        valid_df.drop(columns=resp_col, inplace=True)
        valid_data = valid_df.values
        del valid_df
        # convert both valid_data and valid_targ to tensors for training
        valid_targ = torch.from_numpy(valid_targ)
        # convert data to a float
        valid_data = torch.from_numpy(valid_data).float()
        # converts valid targ to the appropriate type depending on kind of model
        # float for exp
        # long for crossent
        if 'cross' in exp_name:
            valid_targ = valid_targ.long()
        else:
            valid_targ = valid_targ.float()
        # move both the correct device
        valid_targ, valid_data = valid_targ.to(device), valid_data.to(device)

    # grab target
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
    classes = class_series.index.values
    num_classes = classes.size
    num_units = get_num_units(exp_name)

    if not valid:
        num_batches = int(data.shape[0] / batch_size * num_batches)

    # initialize model with appropriate arguments
    if 'cross' in exp_name:
        net = Net(num_feats, num_units, num_classes)
        criterion = nn.CrossEntropyLoss(size_average=True, reduce=True)
    else:
        classes = torch.from_numpy(classes).float()
        classes = classes.to(device)
        net = Net(num_feats, num_units, num_classes, classes)
        criterion = nn.MSELoss(size_average=True, reduce=True)

    # move net to appropriate device
    net.to(device)

    # convert data and target to tensors and place them on the appropriate
    # device
    data = torch.from_numpy(data).float()
    targ = torch.from_numpy(targ)
    # convert targ to appropriate type
    if 'cross' in exp_name:
        targ = targ.long()
    else:
        targ = targ.float()
    # change dimension
    targ = targ.view(-1)

    # move to device
    data, targ = data.to(device), targ.to(device)

    # training loop prep
    loss_hist = []
    optimizer = get_optimizer(net, exp_name)

    # training loop iteration
    if not valid:
        # standard num iterations stopping criterion
        state_dict, loss_hist = batch_count_loop(
            num_batches, net, optimizer, data, targ, batch_size, device, criterion)
    else:
        # validation set stopping criterion
        state_dict, loss_hist, val_hist = valid_loop(valid_dur, valid_data, valid_targ, hist_len, net,
                                                     optimizer, data, targ, batch_size, device, criterion)

    print('Done Training')
    sys.stdout.flush()
    print('Pickling')
    sys.stdout.flush()

    # save outputs
    # save model parameters
    torch.save(state_dict, 'models/exps/%s/model_%s.pth.tar' %
               (exp_name, turn))

    # save loss history and validation loss history if its given
    loss_pickle = open('models/exps/%s/loss_%s.pickle' %
                       (exp_name, turn), 'wb')
    loss_dict = {}
    loss_dict['train_loss'] = loss_hist

    # save validation history if validation set is used as stopping criterion
    if valid:
        loss_dict['valid_loss'] = val_hist

    # dump and close pickle
    pickle.dump(loss_dict, loss_pickle)
    loss_pickle.close()

    # save class_series
    class_series.to_csv(
        'models/exps/%s/class_series_%s.csv' % (exp_name, turn))
    # save feature dictionary
    feat_dict_pick = open('models/exps/%s/featdict_%s.pickle' %
                          (exp_name, turn), 'wb')
    pickle.dump(colix, feat_dict_pick)
    feat_dict_pick.close()
    end_time = dt.now()
    print('Total Time: ' + str(end_time - start_time))


if __name__ == '__main__':
    main()
