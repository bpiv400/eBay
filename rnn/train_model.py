import sys
import os
sys.path.append(os.path.abspath('repo/rnn/models.py'))

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


def get_prep_type(exp_name, is_rnn=False):
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


def rnn_loss_wrapper(loss_tensor, lengths):
    '''
    Computes the average loss per sequence entry for each sequence
    then averages this over all sequences in the batch

    Args:
        loss_tensor: tensor.torch.float with dimensions (seq_length, batch_size)
        lengths: 1 dimensional tensor.torch.float with size=batch_size

    Returns:
        Average per sequence entry loss, normalized by sequence length
    '''
    # loss output should have shape (seq_len, batch_size)
    # compute the sum of each losses for each sequence
    loss_tensor = loss_tensor.sum(dim=0)
    # compute the average of losses for each sequence
    # ie the average loss per turn for each sequence
    loss_tensor = loss_tensor / lengths
    # compute the mean sequence loss
    loss_tensor = loss_tensor.mean()
    return loss_tensor


def process_minibatch_rnn(num_samps, net, optimizer, offr_vals, const_vals, targ_vals, length_vals, batch_size, device, criterion):
    # grab sample indices
    sample_inds = torch.randint(
        low=0, high=num_samps, size=(batch_size,), device=device).long()

    # grab sample data
    sample_offr_vals = offr_vals[:, sample_inds, :]
    sample_const_vals = const_vals[:, sample_inds, :]
    sample_targ_vals = targ_vals[:, sample_inds]
    sample_lengths = length_vals[sample_inds]

    # sort lengths and retrieved sorted indices as well
    sample_lengths, sorted_inds = torch.sort(
        sample_lengths, dim=0, descending=True)

    # apply same sorting to other inputs
    sample_offr_vals = sample_offr_vals[:, sorted_inds, :]
    sample_const_vals = sample_const_vals[:, sorted_inds, :]
    sample_targ_vals = sample_targ_vals[:, sorted_inds]

    # generate packed sequence
    sample_offr_vals = torch.nn.utils.rnn.pack_padded_sequence(sample_offr_vals,
                                                               sample_lengths)
    # forward pass
    output = net(x=sample_offr_vals, h_0=sample_const_vals)
    # output should have shape (seq_len, num_classes, batch_size)
    # calculate loss on each entry, wrap to calculate mean loss
    loss = rnn_loss_wrapper(
        criterion(output, sample_targ_vals), sample_lengths)

    # average loss per sequence entry averaged over all sequences
    loss.backward()
    # update parameters as necessary
    optimizer.step()

    return loss


def process_valid_rnn(net, valid_offr_vals, valid_const_vals,
                      valid_targs, valid_length_vals, criterion):
    '''
    All data components are pre-sorted by length and valid_offr_vals
    has already been packed
    '''
    # get sample output
    output = net(x=valid_offr_vals, h_0=valid_const_vals)

    # calculate loss
    loss = rnn_loss_wrapper(criterion(output, valid_targs),
                            valid_length_vals)
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


def batch_count_loop(num_batches, net, optimizer, offr_vals, const_vals, targ_vals, length_vals, batch_size, device, criterion):
    # get the total number of samples in the training data
    num_samps = offr_vals.shape[1]
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
        loss = process_minibatch_rnn(
            num_samps, net, optimizer, offr_vals, const_vals, targ_vals, length_vals, batch_size, device, criterion)

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


def valid_loop(valid_dur, valid_offr_vals, valid_const_vals,
               valid_targs, valid_length_vals, hist_len, net,
               optimizer, offr_vals, const_vals, targ_vals, length_vals,
               batch_size, device, criterion):
    # initialize counter for the total number of minibatches
    batch_count = 1
    # get the total number of samples in the training data
    num_samps = offr_vals.shape[1]
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
        loss = process_minibatch_rnn(
            num_samps, net, optimizer, offr_vals, const_vals,
            targ_vals, length_vals, batch_size, device, criterion)

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
            valid_loss = process_valid_rnn(net, valid_offr_vals, valid_const_vals,
                                           valid_targs, valid_length_vals, criterion)
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


def unpickle(path):
    '''
    Extracts an abritrary object from the pickle located at path and returns
    that object

    Args:
        path: string denoting path to pickle

    Returns:
        arbitrary object contained in pickle
    '''
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def main():
    # parse parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    sys.stdout.flush()
    start_time = dt.now()

    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
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

    # get data type
    # TODO: I don't think prep type is necessary here due to effects of make_seq
    # prep_type = get_prep_type(exp_name, is_rnn=True)
    # get model class
    Net = get_model_class(exp_name)

    print('Loading Data')
    sys.stdout.flush()

    # load data pickle
    data_loc = 'data/exps/%s/train_data.pickle' % exp_name
    data_dict = unpickle(data_loc)
    feats_loc = 'data/exps/%s/feats.pickle' % exp_name
    feats_dict = unpickle(feats_loc)

    # extract necessary components from data_dict
    const_vals = data_dict['const_vals']
    offr_vals = data_dict['offr_vals']
    targ_vals = data_dict['target_vals']
    midpoint_ser = data_dict['midpoint_ser']
    length_vals = data_dict['length_vals']

    # delete the dictionary itself
    del data_dict
    print('Done Loading')
    sys.stdout.flush()

    # TODO: Consider removing this section when done updating
    # remove  refrerence columns from data frame
    # extra_cols = ['ref_old', 'ref_rec', 'ref_resp']
    # add the response column corresponding to the
    # other kind of model (time_ji for offr_mod)
    # offr_ji otherwise
    # if offr_mod:
    #     if turn != 'start_price_usd':
    #         extra_cols.append(get_resp_time(turn))
    # else:
    #     extra_cols.append(get_resp_offr(turn))
    #
    # for col in extra_cols:
    #     if col in df:
    #         print('Dropping %s' % col)
    #         df.drop(columns=col, inplace=True)
    # get class series (output later)
    # class_series = get_class_series(midpoints)
    # convert raw targets to classes if using cross entropy
    # if 'cross' in exp_name:
    #     df = get_resp_turn_classes(df, resp_col, class_series)

    # if stopping criterion is based ontest performance
    # initialize validation set
    if valid:
        # get the number of samples associated with the percentage size
        valid_size = int(offr_vals.shape[1] * valid_size)
        # valid_size samples from the range of integers in df.index (since this should
        # be a simple step index)
        valid_inds = np.random.random_integers(
            0, (offr_vals.shape[1] - 1), valid_size)
        # sort valid_inds in ascending order
        valid_inds = np.sort(valid_inds)

        # grab the corresponding data then drop the rows from the offr_vals, const_vals, and
        # targs df
        valid_offr_vals = offr_vals[:, valid_inds, :]
        valid_const_vals = const_vals[:, valid_inds, :]
        valid_targs = targ_vals[:, valid_inds]
        valid_length_vals = length_vals[valid_inds]
        # dropping corresponding values
        offr_vals = np.delete(offr_vals, valid_inds, axis=1)
        const_vals = np.delete(const_vals, valid_inds, axis=1)
        targ_vals = np.delete(targ_vals, valid_inds, axis=1)
        length_vals = np.delete(length_vals, valid_inds)

        # convert all three components to tensors for training
        # convert both valid_data and valid_targ to tensors for training
        # convert data to a float
        valid_offr_vals = torch.from_numpy(valid_offr_vals).float()
        valid_const_vals = torch.from_numpy(valid_const_vals).float()
        # assuming cross-entropy environment
        valid_targs = torch.from_numpy(valid_targs)
        valid_targs = valid_targs.long()
        # move all three valid tnesors to device
        valid_targs = valid_targs.to(device)
        valid_offr_vals = valid_offr_vals.to(device)
        valid_const_vals = valid_const_vals.to(device)
        valid_length_vals = valid_length_vals.to(device)

        # sort valid data components by associated sequence lengths
        # sort lengths and retrieved sorted indices as well
        valid_length_vals, sorted_inds = torch.sort(
            valid_length_vals, dim=0, descending=True)

        # apply same sorting to other inputs
        valid_offr_vals = valid_offr_vals[:, sorted_inds, :]
        valid_const_vals = valid_const_vals[:, sorted_inds, :]
        valid_targs = valid_targs[:, sorted_inds]

        # generate packed sequence
        valid_offr_vals = torch.nn.utils.rnn.pack_padded_sequence(valid_offr_vals,
                                                                  valid_length_vals)

    # complete the same conversion process for training components
    offr_vals = torch.from_numpy(offr_vals).float()
    const_vals = torch.from_numpy(const_vals).float()
    # assuming cross-entropy environment
    targ_vals = torch.from_numpy(targ_vals).long()
    length_vals = torch.from_numpy(targ_vals).long()

    # move offr_vals, const_vals, targ_vals, length_vals, to active device
    offr_vals = offr_vals.to(device)
    const_vals = const_vals.to(device)
    targ_vals = targ_vals.to(device)
    length_vals = length_vals.to(device)

    # calculating parameter values for the model from data
    # gives the number of hidden features in the rnn
    num_hidden_feats = const_vals.shape[2]
    num_classes = len(midpoint_ser.index)
    num_offr_feats = offr_vals.shape[2]

    # TODO: Consider removing this when done updating
    # num_units = get_num_units(exp_name)

    if not valid:
        num_batches = int(offr_vals.shape[1] / batch_size * num_batches)

    # initialize model class
    # CONFIRMED: IGNORED LOSS INDICES DO NOT CONTRIBUTE TO LOSS SUMS
    net = Net(num_offr_feats, num_classes, num_hidden_feats)
    criterion = nn.CrossEntropyLoss(
        reduce=False, size_average=False, ignore_index=-100)
    # move net to appropriate device
    net.to(device)

    # training loop prep
    loss_hist = []
    optimizer = get_optimizer(net, exp_name)

    # training loop iteration
    if not valid:
        # standard num iterations stopping criterion
        state_dict, loss_hist = batch_count_loop(
            num_batches, net, optimizer, offr_vals, const_vals, targ_vals, length_vals, batch_size, device, criterion)
    else:
        # validation set stopping criterion
        state_dict, loss_hist, val_hist = valid_loop(valid_dur, valid_offr_vals, valid_const_vals,
                                                     valid_targs, valid_length_vals, hist_len, net, optimizer, offr_vals,
                                                     const_vals, targ_vals, length_vals, batch_size, device, criterion)

    print('Done Training')
    sys.stdout.flush()
    print('Pickling')
    sys.stdout.flush()

    # save outputs
    # save model parameters
    torch.save(state_dict, 'data/exps/%s/model.pth.tar' %
               exp_name)

    # save loss history and validation loss history if its given
    loss_pickle = open('data/exps/%s/loss.pickle' %
                       exp_name, 'wb')
    loss_dict = {}
    loss_dict['train_loss'] = loss_hist

    # save validation history if validation set is used as stopping criterion
    if valid:
        loss_dict['valid_loss'] = val_hist

    # dump and close pickle
    pickle.dump(loss_dict, loss_pickle)
    loss_pickle.close()

    end_time = dt.now()
    print('Total Time: ' + str(end_time - start_time))


if __name__ == '__main__':
    main()
