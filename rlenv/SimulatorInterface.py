import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson
import pandas as pd
import utils
from models.model_names import *
from composer.Composer import Composer
from rlenv import env_consts
from simulator.nets import FeedForward, RNN, LSTM
from constants import TOL_HALF, SLR_PREFIX, BYR_PREFIX


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
    def _bernoulli_sample(logits, sample_size):
        """
        Returns sample of bernoulli distributions defined by logits

        :param logits: 2 dimensional tensor containing logits for a batch of
        distributions. Currently, expects 1 distribution
        :param sample_size: number of samples to draw from each distribution
        :return: 1-dimensional tensor containing
        """
        dist = Bernoulli(logits=logits)
        return SimulatorInterface.proper_squeeze(dist.sample((sample_size, )))



    def cn(self, sources=None, hidden=None, model_name=None, sample=True):
        """
        Samples an offer from the relevant concession model and returns
        the concession value along with the total normalized concession and an indicator for
        split

        :param sources: dictionary containing entries for all  input maps (see env_consts)
        required to construct the model's inputs from source vectors in the environment
        :param hidden: tensor giving the hidden state of the model up to this point
        :param model_name: string giving name of model
        :param sample: boolean giving whether a sample should be drawn from the parameterized dist
        :return: 4-tuple containing an a float [0, 1] drawn from the distribution parameters
        the model outputs, the corresponding normalized concession value, an indicator
        for split, and the hidden state after processing  the turn
        """
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(model_name, sources=sources,
                                                           recurrent=True, size=1, fixed=fixed)
        params, hidden = self.models[model_name].simulate(x_time, x_fixed=x_fixed, hidden=hidden)
        if not sample:
            return 0, 0, 0, hidden
        cn = SimulatorInterface._mixed_beta_sample(params[0, :, :])
        # compute norm, split, and cn
        split = 1 if abs(.5 - cn) < TOL_HALF else 0
        # slr norm
        if SLR_PREFIX in model_name:
            norm = 1 - cn * sources[env_consts.O_OUTCOMES_MAP][2] - \
                   (1 - sources[env_consts.L_OUTCOMES_MAP][2]) * (1 - cn)
        # byr norm
        else:
            norm = (1 - sources[env_consts.O_OUTCOMES_MAP][2]) * cn + \
                sources[env_consts.L_OUTCOMES_MAP][2] * (1 - cn)
        return cn, norm, split, hidden

    def offer_indicator(self, model_name, sources=None, hidden=None, sample=True):
        """
        Computes outputs of recurrent offer models that produce an indicator for some event
        (round, msg, nines, accept, reject, delay)

        :param model_name: str giving name of the target model (see model_names.MODELS for
        valid model names)
        :param sources: dictionary containing entries for all  input maps (see env_consts)
        required to construct the model's inputs from source vectors in the environment
        :param hidden: tensor giving the hidden state of the model up to this point
        :param sample: boolean indicating whether a sample should be drawn
        :return: 2-tuple containing an int {0, 1} drawn from the model's parameters and
        a tensor giving the hidden state after this time step
        """
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(model_name, sources=sources,
                                                           recurrent=True, size=1, fixed=True)
        params, hidden = self.models[model_name].simulate(x_time, x_fixed=x_fixed, hidden=hidden)
        samp = 0
        if sample:
            samp = SimulatorInterface._bernoulli_sample(params[0, :, :], 1)
        return samp, hidden
