import tensorflow as tf
import numpy as np
import os
# custom run hook to allow initializable interator to be used with
# tf.estimator api


class LossOutputHook(tf.train.SessionRunHook):
    def __init__(self, tensor_name, steps=500):
        super(LossOutputHook, self).__init__()
        # create empty list to which losses will be stored
        self.recent_loss = []
        self.loss = []
        # number of steps between saving loss
        self.steps = steps
        # names of loss tensor in default graph
        self.tensor_name = tensor_name
        # timer for step tracking
        self._timer = tf.train.SecondOrStepTimer(every_steps=self.steps)

    def begin(self):
        # reset timer
        self._timer.reset()
        # reset iterator counter
        self._iter_count = 0
        # extract graph element
        self._current_tensor = tf.get_default_graph().get_tensor_by_name(
            name=self.tensor_name)

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)
        if self._should_trigger:
            return tf.train.SessionRunArgs(self._current_tensor)

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            # log loss every time the timer is triggered
            self.recent_loss = []
            self.loss.append(run_values.results)
        self._iter_count += 1

    def export_loss(self):
        return self.loss


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None  # Will be set in the input_fn

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)


def get_inputs(offrs, consts, lens, y, valid=False):
    iterator_initializer_hook = IteratorInitializerHook()

    def input_fn():
        # ensure features is transformed such that batch is on the 1st, rather than
        # the middle axis before uploading (or execute transformation here)

        # initialize place holders
        # should have shape batch_size x max_seq_length x num_offr_features
        offrs_pl = tf.placeholder(offrs.dtype, offrs.shape)
        y_pl = tf.placeholder(y.dtype, y.shape)
        lens_pl = tf.placeholder(lens.dtype, lens.shape)
        # should have shape batch size x num_const_features
        consts_pl = tf.placeholder(consts.dtype, consts.shape)

        # create feature dictionary
        features_dict = {}
        features_dict['offrs'] = offrs_pl
        features_dict['lens'] = lens_pl
        features_dict['consts'] = consts_pl

        if not valid:
            dataset = tf.data.Dataset.from_tensor_slices((features_dict, y_pl))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(1000)
            dataset = dataset.batch(32)
        else:
            dataset = tf.data.Dataset.from_tensors((features_dict, y_pl))
            dataset = dataset.repeat()

        iterator = dataset.make_initializable_iterator()
        next_feature_dict, next_label = iterator.get_next()

        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer,
                                                                                    feed_dict={offrs_pl: offrs,
                                                                                               lens_pl: lens,
                                                                                               consts_pl: consts,
                                                                                               y_pl: y})

        return next_feature_dict, next_label

    return input_fn, iterator_initializer_hook


"""
estimator.train(input_fn=train_input_fn,
                hooks=[train_iterator_initializer_hook])
estimator.evaluate(input_fn=test_input_fn,
                   hooks=[test_iterator_initializer_hook])
"""
