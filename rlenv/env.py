# policy gradient method environment
# Required functionalities
# 1. Load state transition probability model
# 2. initialize random policy
# 3. Generate sequences using current policy and state transition
# probability model

# import required packages

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch import optim
import numpy as np
import math
import sys
import torch.nn.functional as F
import os
import pickle
import re

sys.path.append(os.path.relpath('repo/rnn/'))
from models import SimpleRNN


class GradientEnvironment():
    def __init__(self, trans_prob_name=None):
        # error checking
        self.__is_none(trans_prob_name)
        # store name as a local variable
        self.trans_name = trans_prob_name
        # store data into fields
        self.__extract_data()

        # initialize transition probability model
        self.trans_model = self.__init_model()

        # transform constant features into appropriate format
        self.consts = SimpleRNN.transform_hidden_state(self.consts,
                                                       params_dict=self.trans_params)

    @staticmethod
    def __is_none(obj):
        '''
        Private static method that checks whether a given
        argument is None and throws an error if so
        '''
        if obj is None:
            raise ValueError("All arguments must be defined")

    def __extract_data(self):
        '''
        Private helper method that parses the experiment name
        to extract the data type then loads the corresponding
        data and feature dictionaries and stores their items
        in local variables

        Args:
            None
        Return:
            None
        '''
        # parse name of data from experiment name
        data_name = self.get_data_name(self.trans_name)
        # assign names of feature and data dictionaries
        data_loc = 'data/exps/%s/train_data.pickle' % data_name
        feats_loc = 'data/exps/%s/feats.pickle' % data_name
        self.feats_dict = self.unpickle(feats_loc)
        data_dict = self.unpickle(data_loc)

        # extract necessary components from data_dict
        # num_state_layers x n x state_size
        self.consts = data_dict['const_vals']
        # max_offers x n x feats_per_offer
        self.offrs = data_dict['offr_vals']
        # max_offers x n
        self.targs = data_dict['target_vals']
        # series where the values give the midpoint of each
        # offer bin and the indices give their corresponding
        # class indices from the perspective of the
        # trans probs model
        self.midpoint_ser = data_dict['midpoint_ser']
        # n x 1 numpy vector of lengths
        self.lengths = data_dict['length_vals']

    def gen_seqs(self, size=.20):
        '''
        Generate sequences using current policy and 
        trans probs model
        1. Ensure a policy model exists
        2. Randomly samples constant features of model
        3. Transforms constant features to input expected by
        '''
        if self.trans_model is None:
            raise ValueError(
                "Must initialize transition model before generating sequences")

        # sample a subset of constant features from the const features data

    @staticmethod
    def get_data_name(exp_name):
        '''
        Parse name of the dataset being used out of the
        name of the transition probability experiment used

        Args:
            exp_name: string giving name of transition probability
            experiment
        '''

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

    @staticmethod
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

    def __get_trans_params(self):
        '''
        Parses the name of the experiment and the data stored
        from __extract_data(...) to determine the parameters
        of the transition probability model

        Called after __extract_data(...)

        Args:
            None
        Returns:
            Dictionary string -> val
        '''
        trans_params = {}
        trans_params['num_offr_feats'] = self.offrs.shape[2]
        trans_params['org_hidden_size'] = self.consts.shape[2]
        trans_params['init'] = SimpleRNN.get_init(self.trans_name)
        trans_params['lstm'] = SimpleRNN.get_lstm(self.trans_name)
        trans_params['zeros'] = SimpleRNN.get_zeros(self.trans_name)
        trans_params['num_layers'] = SimpleRNN.get_num_layers(self.trans_name)
        trans_params['targ_hidden_size'] = SimpleRNN.get_hidden_size(self.consts,
                                                                     self.trans_name)
        trans_params['bet_hidden_size'] = SimpleRNN.get_bet_size(
            self.trans_name)
        trans_params['num_classes'] = SimpleRNN.get_num_classes(
            self.midpoint_ser)
        return trans_params

    def __init_model(self):
        '''
        Extracts transition probability model parameters, stores them in an
        instance variable, then initializes transition probability model using
        these params, and updates parameters to values stored at the end of training

        Args:
            None

        Returns:
            None
        '''
        # extract parameters from name of trans_probs model and data loaded earlier
        self.trans_params = self.__get_trans_params()
        Net = SimpleRNN.get_model_class(self.trans_name)
        trans_model = Net(self.trans_params['num_offr_feats'],
                          self.trans_params['num_classes'],
                          lstm=self.trans_params['lstm'],
                          targ_hidden_size=self.trans_params['targ_hidden_size'],
                          org_hidden_size=self.trans_params['org_hidden_size'],
                          bet_hidden_size=self.trans_params['bet_hidden_size'],
                          layers=self.trans_params['num_layers'],
                          init_processing=self.trans_params['init'])

        # parse location of model weights from trans_name
        weight_loc = 'data/exps/%s/model.pth.tar' % self.trans_name
        # load weights into Python
        model_dict = torch.load(weight_loc)
        # warm start trans model
        trans_model.load_state_dict(model_dict)
        return trans_model
