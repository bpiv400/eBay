import tensorflow as tf
import math
import sys
import os
import pickle
import re
import numpy as np


def rnn_fun(features, labels, mode, params):
    """
    Description: Generic custom estimator model function for transition
    probability rnn
    features: dictionary containing two keys:
        lens: 1d tensor of sequence lengths
        features: 3d tensor where first dimension gives batch size, second gives
        max sequence length, and third gives the number of features per offer
        consts: 3d tensor where first dimension gives batch size, second dimension
        gives number of layers of rnn cell, and third dimension gives hidden size
    labels: 2d tensor of batch size by sequence length
    params: dictionary of additional model parameters with the following string keys:
        pre: apply pre-processing
        bet_hidden_size: number of units in the pre-processing layer
        deep: have additional layers of rnns before output layer
        num_deep: total number of layers of rnns
        lstm: boolean giving whether the rnn cell should be an lstm
        num_classes: number of classes
        targ_hidden_size: number of units in target hidden state
        dropout: boolean for whether dropout regularization should be utilized
        reg_weight: scalar giving the weight of l2 regularization in the loss function
    """
    # parse inputs out of feature dictionary
    lens = features['lens']
    offrs = features['offrs']
    max_length = tf.shape(offrs)[1]
    consts = features['consts']
    # instantiate targ_hidden_size
    if 'targ_hidden_size' in params:
        targ_hidden_size = params['targ_hidden_size']
    else:
        targ_hidden_size = consts.get_shape().as_list()[2]
    # handle pre processing layers
    if params['pre']:
        # initialize size of pre processing layers
        if 'bet_hidden_size' is not None:
            bet_hidden_size = params['bet_hidden_size']
        else:
            # set units to be equal to number of units in hidden state
            bet_hidden_size = consts.get_shape().as_list()[2]
        # pass constants through two pre-processing layers where first
        # layer outputs arbitrary number of units
        # and second layer outputs target_hidden_size units
        consts = tf.layers.dense(
            consts, bet_hidden_size, activation=tf.nn.relu,
            kernel_initializer=tf.glorot_uniform_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='preprocess_1')
        consts = tf.layers.dense(
            consts, params['target_hidden_size'],
            activation=tf.nn.relu, name='prepocess_2',
            kernel_initializer=tf.glorot_uniform_initializer(),
            bias_initializer=tf.zeros_initializer())
    if not params['deep']:
        init_state = tf.squeeze(consts)
        if not params['lstm']:
            rnn_cell = tf.contrib.rnn.BasicRNNCell(targ_hidden_size)
        else:
            # unclear if this will work -- documentation is very poor
            init_state = tf.contrib.rnn.LSTMStateTuple(init_state,
                                                       init_state)
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(targ_hidden_size)
    else:
        num_deep = params['num_deep']
        # initialize tuples to contain layers of hidden state
        # and layers of hidden state tuples
        init_state = ()
        for i in range(num_deep):
            # extract row corresponding to current initial state from tensor
            curr_init_state = consts[:, i, :]
            # if we are working with an lstm, create lstm state tuple for initialization
            if params['lstm']:
                curr_init_state = tf.contrib.rnn.LSTMStateTuple(curr_init_state,
                                                                curr_init_state)
            # add extracted current state level to tuple of states
            init_state = init_state + (curr_init_state, )
        if not params['lstm']:
            rnn_layers = [tf.contrib.rnn.BasicRNNCell(
                targ_hidden_size) for _ in range(num_deep)]
        else:
            rnn_layers = [tf.contrib.rnn.BasicLSTMcell(
                targ_hidden_size) for _ in range(num_deep)]
        # compose layers into multi cell
        rnn_cell = [tf.contrib.rnn.MultiRNNCell(rnn_layers)]
    # pass input offers through dynamic rnn layer
    output, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                  inputs=offrs,
                                  initial_state=init_state)
    # pass output of dynamic rnn layer through logit layer
    logits = tf.layers.dense(output, params['num_classes'], activation=tf.nn.relu,
                             name='logit_layer', kernel_initializer=tf.glorot_uniform_initializer(),
                             bias_initializer=tf.zeros_initializer())
    logits = tf.cast(logits, tf.float64)
    # generate weight mask to hide padded predictions from loss
    sequence_mask = tf.sequence_mask(lens, maxlen=max_length, dtype=tf.float64)
    # arg max over logits for each turn-sequence combination
    predicted_classes = tf.argmax(logits, 2)
    # calculate probabilities for each offer-offer sequence pair
    probabilities = tf.nn.softmax(logits)
    # generate predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,
            'logits': logits,
            'probabilities': probabilities,
            'sequence_mask': sequence_mask
        }
        export_output = tf.estimator.export.ClassificationOutput(scores=probabilities,
                                                                 classes=predicted_classes)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_output)
    else:
        # calculate loss for evaluation and training mode
        loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                labels,
                                                weights=sequence_mask,
                                                average_across_timesteps=True,
                                                average_across_batch=True)
        # get trainable collection to implement l2 regularization
        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # initialize regularized loss
        reg_loss = tf.constant(0, shape=[], dtype=tf.float64)
        # iterate over trainable variables
        for var in trainables:
            # summing regularized loss
            reg_loss = tf.add(reg_loss, tf.cast(
                tf.nn.l2_loss(var), tf.float64))
        # initialize regularization weight
        reg_weight = tf.constant(
            params['reg_weight'], shape=[], dtype=tf.float64)
        # add weighted regularized loss to sequence loss
        loss = tf.add(loss, tf.cast(tf.multiply(
            reg_weight, reg_loss), tf.float64), name='loss')
        tf.summary.scalar('loss_metric', loss)
        # if in evaluation mode, additionally calculate accuracy and add it to tensor board
        if mode == tf.estimator.ModeKeys.EVAL:
            acc = tf.metrics.accuracy(labels, predicted_classes, weights=sequence_mask,
                                      name='accuracy_metric')
            tf.summary.scalar('accuracy', acc[1])
            metrics = {'accuracy': acc,
                       'loss': loss}
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        else:
            # initialize a print logging hook for loss
            # add soon if training doesn't automatically print loss
            printing_hook = tf.train.LoggingTensorHook(
                {'loss': loss}, every_n_iter=100)
            # add parameter optimizer logic here later..for now, simply
            # instantiate optimizer
            optimizer = tf.train.AdagradOptimizer(learning_rate=.001)
            train_op = optimizer.minimize(
                loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss,
                                              training_hooks=[printing_hook],
                                              train_op=train_op)


def torch_tensor_prep(offrs):
    """
    Swaps the first and second dimension of the numpy offers array
    since this array places batch size on the second dimension but
    the input function expects it on the first
    Input:
        offrs: 3-rank np.ndarray
    Output:
        3-rank np.ndarray
    """
    offrs = np.swapaxes(offrs, 0, 1)
    return offrs
