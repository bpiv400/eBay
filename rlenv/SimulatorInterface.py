# TODO UPDATE DOCUMENTATION
# WRITE METHODS

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.negative_binomial import NegativeBinomial
import pandas as pd
import utils
from torch.nn.functional import sigmoid
from rlenv.model_names import *
from rlenv.Composer import Composer
from rlenv import env_consts
from simulator.nets import FeedForward, RNN, LSTM
from constants import TOL_HALF


class SimulatorInterface:
    def __init__(self, params):
        """
        Use rl experiment params to initialize models

        :param params:
        """
        self.models = dict()
        self.params = params
        for model in MODELS:
            self.models[model] = self._init_model(model, params[model])
        self.composer = Composer(params)

    @staticmethod
    def _init_model(model_name, model_exp):
        """
        Initialize pytorch network for some model
        TODO: Make pathing names function in utils or in parsing file
        :param model_exp: experiment number for the model
        :return: PyTorch Module
        """
        # get pathing names
        if SLR_PREFIX in model_name:
            model_type = SLR_PREFIX
            model_name = model_name.replace('{}_'.format(SLR_PREFIX), '')
        elif BYR_PREFIX in model_name:
            model_type = BYR_PREFIX
            model_name = model_name.replace('{}_'.format(BYR_PREFIX), '')
        else:
            model_type = ARRIVAL_PREFIX

        model_dir = '{}/{}/{}/'.format(env_consts.MODEL_DIR,
                                       model_type, model_name)
        model_params_path = '{}params.csv'.format(model_dir)
        sizes_path = '{}sizes.pkl'.format(model_dir)
        model_path = '{}{}.pt'.format(model_dir, model_exp)
        sizes = utils.unpickle(sizes_path)
        params = pd.from_csv(model_params_path, index_col='id')
        params = params.loc[model_exp].to_dict()
        if model_name in FEED_FORWARD:
            net = FeedForward(params, sizes, toRNN=False)
        elif model_name in LSTM:
            net = LSTM(params, sizes)
        else:
            net = RNN(params, sizes)
        net.load_state_dict(torch.load(model_path))
        return net

    @staticmethod
    def _bernoulli_sample(logits, num_byrs, ff=True):
        """

        :param logits:
        :param num_byrs:
        :param ff:
        :return:
        """
        dist = Bernoulli(logits=logits)
        # this squeeze might create a bug when num_byrs = 1
        return SimulatorInterface.proper_squeeze(dist.sample((num_byrs, )))

    def loc(self, sources=None, num_byrs=None):
        """
        Returns indicators for whether each buyer in a set of size
        num_byrs is from the US
        :param sources:
        :param num_byrs: int number of buyers
        :return: 1d tensor giving whether each buyer is from the US
        """
        x_fixed, _ = self.composer.build_input_vector(model_name=LOC, sources=sources,
                                                      fixed=True, recurrent=False, size=1)
        output = self.models[LOC].simulate(x_fixed)
        locs = SimulatorInterface._bernoulli_sample(output, num_byrs, ff=True)
        return locs

    def hist(self, sources=None, byr_us=None):
        """
        Returns the number of previous best offer threads each
        buyer has participated in

        :param sources: source vectors
        :param byr_us: 1d np.array giving whether each byr is from the us
        :return: 1d np.array giving number of experiences each buyer has had
        """
        us_count = torch.nonzero(byr_us).item()
        foreign = us_count < byr_us.shape[0]
        us = us_count > 0
        x_fixed = self.composer.hist_input(sources=sources, us=us, foreign=foreign)
        output = self.models[HIST].simulate(x_fixed)

        params = torch.zeros(byr_us.shape[0], output.shape[1])
        if foreign and us:
            foreign = byr_us == 0
            params[foreign, :] = output[0, :]
            params[~foreign, :] = output[1, :]
        else:
            params[:, :] = output[0, :]
        hists = SimulatorInterface._sample_negative_binomial(params)
        return hists

    @staticmethod
    def proper_squeeze(tensor):
        """
        Squeezes a tensor to 1 rather than 0 dimensions
        :param tensor:
        :return:
        """
        tensor = tensor.squeeze()
        if len(tensor.shape) == 0:
            tensor = tensor.unsqueeze(-1)
        return tensor

    @staticmethod
    def _sample_negative_binomial(params):
        """
        Returns a sample from a batched negative binomial distribution

        :param params:
        :return:
        """
        dist = NegativeBinomial(total_count=torch.exp(params[:, 0]), logits=params[:, 1])
        sample = SimulatorInterface.proper_squeeze(dist.sample((1, )))
        return sample

    def days(self, sources=None, hidden=None):
        """
        Returns the number of buyers who arrive on a particular day

        """
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(DAYS, sources=sources,
                                                           fixed=fixed, recurrent=True,
                                                           size=1)
        params, hidden = self.models[DAYS].simulate(x_time, x_fixed=x_fixed, hidden=hidden)
        params = params.squeeze().unsqueeze(-1)
        sample = SimulatorInterface._sample_negative_binomial(params)
        return sample

    @staticmethod
    def _mixed_beta_sample(params, sample_size):
        """

        :param params:
        :return:
        """
        params = params.reshape(sample_size, 3, -1).permute(0, 2, 1)
        params[:, :, [0, 1]] = torch.exp(params[:, :, [0, 1]]) + 1
        ancestor = Categorical(logits=params[:, :, 2])
        draws = ancestor.sample(sample_shape=(1,))
        beta_params = params[torch.arange(params.shape[0]), draws[0, :], :]
        beta = Beta(beta_params[:, 0], beta_params[:, 1])
        sample = beta.sample((1,)).squeeze()
        if len(sample.shape) == 0:
            sample = sample.unsqueeze(-1)
        return sample

    def sec(self, sources=None, num_byrs=None):
        """

        :param sources:
        :param num_byrs:
        :return:
        """
        x_fixed, _ = self.composer.build_input_vector(SEC, sources=sources, recurrent=False,
                                                      size=num_byrs, fixed=True)
        params = self.models[SEC].simulate(x_fixed)
        times = SimulatorInterface._mixed_beta_sample(params, num_byrs)
        return times

    def bin(self, sources=None, num_byrs=None):
        """
        Returns whether each buyer chooses to buy the listing now

        :param consts:
        :param clock_feats:
        :param byr_us:
        :param byr_hist:
        :return:
        """
        x_fixed, _ = self.composer.build_input_vector(BIN, sources=sources, recurrent=False,
                                                      size=num_byrs, fixed=True)
        params = self.models[BIN].simulate(x_fixed)
        bins = SimulatorInterface._bernoulli_sample(params, num_byrs, ff=True)
        return bins

    def cn(self, model_name, sources=None, hidden=None, turn=1):
        """
        If turn = 1, or 2, ensure default values have been computed for all sources in environment
        :param model_name:
        :param sources:
        :param hidden:
        :param turn:
        :return:
        """
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(model_name, sources=sources,
                                                           recurrent=True, size=1, fixed=fixed)
        params, hidden = self.models[model_name].simulate(x_time, x_fixed=x_fixed, hidden=hidden)
        cn = SimulatorInterface._mixed_beta_sample(params, 1)
        # compute norm, split, and cn
        split = 1 if abs(.5 - cn) < TOL_HALF else 0
        # rethink this norm business...
        if turn == 1:
            norm = cn
        elif turn == 2:
            norm = cn - cn * sources[env_consts.O_OUTCOMES_MAP][2]
        else:
