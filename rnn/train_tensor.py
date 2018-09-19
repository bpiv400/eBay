import sys
import os
sys.path.append(os.path.abspath('repo/rnn/models.py'))

from models_tensor import *
from hooks_tensor import *
from models import *
import tensorflow as tf
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
        low=0, high=(num_samps - 1), size=(batch_size,), device=device).long()

    # debugging
    # print('offr val shape in minibatch processing')
    # print(offr_vals.shape)

    # grab sample data
    sample_offr_vals = offr_vals[:, sample_inds, :]
    sample_const_vals = const_vals[:, sample_inds, :]
    sample_targ_vals = targ_vals[:, sample_inds]
    sample_lengths = length_vals[sample_inds]

    # sort lengths and retrieved sorted indices as well
    sample_lengths, sorted_inds = torch.sort(
        sample_lengths, dim=0, descending=True)
    # extract max length
    max_length = sample_lengths[0]
    # apply same sorting to other inputs and exclude all indices after
    # max_length - 1
    sample_offr_vals = sample_offr_vals[:max_length, sorted_inds, :]
    sample_const_vals = sample_const_vals[:, sorted_inds, :]
    sample_targ_vals = sample_targ_vals[:max_length, sorted_inds]

    # generate packed sequence
    # will the printing ever stop

    # debugging
    # print('Max length: %d' % max_length)
    # print('sample offr input before padding')
    # print(sample_offr_vals.shape)
    sample_offr_vals = torch.nn.utils.rnn.pack_padded_sequence(sample_offr_vals,
                                                               sample_lengths)
    # forward pass
    net.set_init_state(sample_const_vals)
    output = net(x=sample_offr_vals)

    # debugging shapes
    # print('output')
    # print(output.shape)
    # print('sample targ')
    # print(sample_targ_vals.shape)

    # output should have shape (seq_len, num_classes, batch_size)
    # calculate loss on each entry, wrap to calculate mean loss
    loss = rnn_loss_wrapper(
        criterion(output, sample_targ_vals), sample_lengths.float())

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
    net.set_init_state(valid_const_vals)
    output = net(x=valid_offr_vals)

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
    if num_it < hist_len:
        return True
    else:
        return valid_hist[hist_len - 1] < valid_hist[0]


def delete_checkpoints(checkpoint_dict):
    for checkpoint_file in checkpoint_dict.values():
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
    pass


def valid_loop(valid_dur, hist_len, estimator=None, valid_input_fn=None, training_input_fn=None,
               valid_iterator_initializer_hook=None, train_iterator_initializer_hook=None,
               valid_loss_logging_hook=None, loss_logging_hook=None, model_dir=None,
               max_batches=None):

    # initialize the flag that determines whether validation error is decreasing
    valid_dec = True
    # initialize validation dictionary to contain dictionaries of model parameters so we can revert to last model before
    # error increased
    checkpoint_dict = {}
    # in the dictionary, higher integer keys indicate more recent model parameters
    for i in range(hist_len):
        checkpoint_dict[i] = None
    # initialize validation loss tracking list
    valid_hist = []
    # initialize evaluation batch count check
    mini_batch_count = 1
    # iterate while validation error is decreasing
    while valid_dec & mini_batch_count < max_batches:
        # train the estimator for valid_dur steps
        estimator.train(training_input_fn, hooks=[train_iterator_initializer_hook,
                                                  loss_logging_hook], steps=valid_dur)
        # calculate the loss on the validation set
        rec_loss = estimator.evaluate(valid_input_fn, hooks=[valid_iterator_initializer_hook,
                                                             valid_loss_logging_hook], steps=1)
        # store the path to the latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        # append the most recent loss to the list of losses if fewer than hist_len
        # steps have been execute
        if len(valid_hist) < hist_len:
            valid_hist.append(rec_loss)
        else:
            # otherwise move all existing losses 1 index lower and add the most recent
            # loss to the end
            for i in range(hist_len):
                valid_hist[i] = valid_hist[i + 1]
            valid_hist[hist_len - 1] = rec_loss
        # iterate over checkpoint dictionary pushing each checkpoint in the dictionary to a lower index
        for i in range(hist_len):
            checkpoint_dict[i] = copy.deepcopy(checkpoint_dict[i + 1])
            # make the most recent entry equal state dict
            checkpoint_dict[hist_len - 1] = copy.deepcopy(latest_checkpoint)

            # check the stopping criterion that the validation error has
            # increased on the last hist_dur iterations
            valid_dec = is_valid_dec(valid_hist, hist_len)
        mini_batch_count = mini_batch_count + 1
    # delete more recent checkpoint files
    output_checkpoint = checkpoint_dict[0]
    delete_checkpoints(checkpoint_dict)
    # return nothing
    pass


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


def get_data_name(exp_name):
    print(exp_name)
    sys.stdout.flush()
    if 'lstm' in exp_name:
        data_name = exp_name.replace('lstm', 'rnn')
    else:
        data_name = exp_name
    arch_type_str = r'_(simp|cat|sep)'
    type_match = re.search(arch_type_str, data_name)
    if type_match is None:
        raise ValueError('Invalid experiment name')
    type_match_end = type_match.span(0)[1]
    data_name = data_name[:type_match_end]
    return data_name


def main():
    # parse parameters
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
    # pre-processing middle layer hidden size
    parser.add_argument('--pre_bet_size', action='store',
                        type=int, default=None)
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
    # gives the size of the L2 regularization parameter
    parser.add_argument('--reg', action='store',
                        type=float, default=math.pow(10, -5))
    # extract parameters
    args = parser.parse_args()
    global exp_name
    exp_name = args.exp.strip()
    num_batches = args.batches
    batch_size = args.batch_size
    bet_hidden_size = args.pre_bet_size
    reg_weight = args.reg
    # parse flag out of zeros argument

    # set a flag for whether the stopping criterion being used is
    # a validation error set
    valid = 'val' in exp_name
    init = 'init' in exp_name
    lstm = 'lstm' in exp_name

    # initialize params dictionary
    params = {}
    # add args to estimator params dictionary
    params['pre'] = init
    params['lstm'] = lstm
    params['bet_hidden_size'] = bet_hidden_size
    params['reg_weight'] = reg_weight

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

    print('Loading Data')
    sys.stdout.flush()

    data_name = get_data_name(exp_name)

    # load data pickle
    data_loc = 'data/exps/%s/train_data.pickle' % data_name
    data_dict = unpickle(data_loc)
    feats_loc = 'data/exps/%s/feats.pickle' % data_name
    feats_dict = unpickle(feats_loc)

    # extract necessary components from data_dict
    const_vals = data_dict['const_vals']
    offr_vals = data_dict['offr_vals']
    targ_vals = data_dict['target_vals']
    midpoint_ser = data_dict['midpoint_ser']
    length_vals = data_dict['length_vals']

    # ######################################
    # TODO: UNSORT ALL INPUT VALUES
    ##########################################

    # replace ignore indices with 0's--equivalently any value
    replace_mask = targ_vals == -100
    print(np.sum(replace_mask)/replace_mask.size)
    targ_vals[replace_mask] = 0

    # if target size flag has been set in experiment name, add
    # additional 0's to hidden state data
    const_vals, num_layers, targ_hidden_size = transform_hidden_state(
        const_vals, exp_name)
    # add additional parameters relating to depth of rnn
    #  to params dictionary
    params['deep'] = num_layers > 1
    if params['deep']:
        params['num_deep'] = num_layers

    # debugging
    print(offr_vals.shape)
    print(targ_vals.shape)

    # delete the dictionary itself
    del data_dict
    print('Done Loading')
    sys.stdout.flush()

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

    # calculating parameter values for the model from data
    # gives the number of hidden features in the rnn
    num_classes = len(midpoint_ser.index) - 1  # subtract the filler class
    # add corresponding parameter value to dictionary
    params['num_classes'] = num_classes
    num_offr_feats = offr_vals.shape[2]
    # characterizing relative sizes

    # calculate the number of gradient descent steps to be taken if we are not
    # using a validation set
    if not valid:
        steps = int(offr_vals.shape[1] / batch_size * num_batches)
    else:
        batch_size = 32
        max_steps = 25 * int(offr_vals.shape[1] / batch_size)

    print('offer features: %d' % num_offr_feats)
    print('hidden state size: %d' % targ_hidden_size)
    print('number of classes: %d' % num_classes)
    print('org hidden size: %d' % const_vals.shape[2])

    # swap the first and second dimensions of the offrs, targets,
    #  and const_vals numpy arrays to prep them for the
    # which expects batch size as the primary axis input function
    offr_vals = torch_tensor_prep(offr_vals)
    const_vals = torch_tensor_prep(const_vals)
    targ_vals = torch_tensor_prep(targ_vals)
    # convert targets to integer
    targ_vals = targ_vals.astype(np.int32)

    # do the same for valid numpy arrays if necessary
    if valid:
        valid_offr_vals = torch_tensor_prep(valid_offr_vals)
        valid_const_vals = torch_tensor_prep(valid_const_vals)
        valid_targs = torch_tensor_prep(valid_targs)
        # convert targets to integer
        valid_targs = valid_targs.astype(np.int32)

    # give relative path from current directory to model directory
    model_dir = 'data/exps/%s' % exp_name
    # initialize estimator class using custom model_fn
    # and accumulated params history
    rnn_estimator = tf.estimator.Estimator(
        model_fn=rnn_fun,
        params=params,
        model_dir=model_dir
    )
    # DEPRECATED SINCE IT SHOULD BE POSSIBLE TO LOAD MODEL FROM CHECKPOINT
    ###############################################################################
    # define input spec
    # feature_spec = {
    #     'lens': tf.placeholder(length_vals.dtype, length_vals.shape),
    #     'offrs': tf.placeholder(offr_vals.dtype, offr_vals.shape),
    #     'consts': tf.placeholder(const_vals.dtype, const_vals.shape)
    # }
    # # generate server receiver function from input spec
    # serving_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
    #     feature_spec)
    #################################################################################

    # initialize training loss logging hook
    loss_logging_hook = LossOutputHook('loss:0', steps=500)

    # initialize training data initializer hook and input function for training loop
    train_input_fn, train_iterator_initializer_hook = get_inputs(offr_vals,
                                                                 const_vals, length_vals, targ_vals, valid=False)
    print(const_vals.dtype)

    if not valid:
        # training loop iteration
        # tf.logging.set_verbosity(tf.logging.INFO)
        rnn_estimator.train(train_input_fn, hooks=[train_iterator_initializer_hook,
                                                   loss_logging_hook], steps=steps)
    else:
        # initialize validation input function and initializer for validation
        # dataset
        valid_input_fn, valid_iterator_initializer_hook = get_inputs(valid_offr_vals,
                                                                     valid_const_vals, valid_length_vals, valid_targs, valid=True)
        # initialize validation loss logging loop
        valid_loss_logging_hook = LossOutputHook('loss', steps=1)

        # DEPRECATED
        ########################################################
        # initialize exporter for validation loss training loop
        # should be replaced with less costly custom exporter
        # exporter = tf.estimator.BestExporter(
        #     serving_input_receiver_fn=serving_receiver_fn)

        # initialize training spec
        # train_spec = tf.train.TrainSpec(train_input_fn, max_steps=2000,
        #                                 hooks=[train_iterator_initializer_hook, loss_logging_hook])
        # valid_spec = tf.train.EvalSpec(valid_input_fn, steps=exporters=exporter, throttle_secs=600,
        #                                start_delay_secs=120, hooks=[valid_iterator_initializer_hook, valid_loss_logging_hook])
        ########################################################

        # validation set stopping criterion
        valid_loop(valid_dur, hist_len, estimator=rnn_estimator, valid_input_fn=valid_input_fn, training_input_fn=training_input_fn,
                   valid_iterator_initializer_hook=valid_iterator_initializer_hook,
                   train_iterator_initializer_hook=train_iterator_initializer_hook,
                   valid_loss_logging_hook=valid_loss_logging_hook, loss_logging_hook=loss_logging_hook,
                   model_dir=model_dir)
    print('Done Training')
    sys.stdout.flush()
    print('Pickling')
    sys.stdout.flush()

    # Deprecated since for our purposes model can be loaded from latest checkpoint with warm start
    # in evaluation script -- hopefully this occurs automatically
    ###############################################################################################
    # export the saved model
    # rnn_estimator.export_savedmodel('data/exps/%s' % exp_name, serving_receiver_fn,
    #                                 strip_default_attrs=True)
    ###############################################################################################

    # save loss history and validation loss history if its given
    loss_pickle = open('data/exps/%s/loss.pickle' %
                       exp_name, 'wb')
    # export loss history from loss_logging hook
    loss_hist = loss_logging_hook.export_loss()
    # add to loss dictionary
    loss_dict = {}
    loss_dict['train_loss'] = loss_hist

    # save validation history if validation set is used as stopping criterion
    if valid:
        val_hist = valid_loss_logging_hook.export_loss()
        loss_dict['valid_loss'] = val_hist

    # dump and close pickle
    pickle.dump(loss_dict, loss_pickle)
    loss_pickle.close()

    end_time = dt.now()
    print('Total Time: ' + str(end_time - start_time))


if __name__ == '__main__':
    main()
