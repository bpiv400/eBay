import torch
from torch.distributions import Poisson
from rlenv.env_utils import load_model, proper_squeeze, categorical_sample
from rlenv.interface.model_names import (model_str, CON, MSG, DELAY,
                                         BYR_HIST, NUM_OFFERS)


class PlayerInterface:
    def __init__(self, composer=None, byr=False):
        #composer
        self.composer = composer
        #store names for each
        self.con_model_name = model_str(CON, byr=byr)
        self.msg_model_name = model_str(MSG, byr=byr)
        self.delay_model_name = model_str(DELAY, byr=byr)
        # load models
        self.msg_model = load_model(self.msg_model_name)
        self.con_model = load_model(self.con_model_name)
        self.delay_model = load_model(self.delay_model_name)

    def con(self, sources=None, turn=0):
        x_fixed, x_time = self.composer.build_input_vector(self.con_model_name,
                                                           sources=sources,
                                                           fixed=True)
        params = self.con_model.simulate(x_time, x_fixed=x_fixed, turn=turn)
        return params

    def msg(self, sources=None, turn=0):
        x_fixed, x_time = self.composer.build_input_vector(self.msg_model_name,
                                                           sources=sources,
                                                           fixed=True)
        params = self.msg_model.simulate(x_time, x_fixed=x_fixed, turn=turn)
        return params

    def delay(self, sources=None, hidden=None):
        x_fixed, x_time = self.composer.build_input_vector(self.delay_model_name,
                                                           sources=sources,
                                                           recurrent=True, fixed=False)
        params, hidden = self.delay_model.simulate(x_time, x_fixed=x_fixed, hidden=hidden)
        return params, hidden

    def init_delay(self, sources=None):
        x_fixed, _ = self.composer.build_input_vector(self.delay_model_name,
                                                      sources=sources,
                                                      recurrent=False, fixed=True)
        hidden = self.delay_model.init(x_fixed=x_fixed)
        return hidden


class ArrivalInterface:
    def __init__(self, composer=None):
        # Load interface
        self.composer = composer
        self.hist_model = load_model(BYR_HIST)
        self.num_offers_model = load_model(NUM_OFFERS)
        self.hidden = None

    @staticmethod
    def _poisson_sample(params):
        """
        Draws a sample from a poisson distribution parameterized by
        the input tensor according to documentation in ebay/documents

        :param params: 1-dimensional torch.tensor output by a poisson model
        :return: torch.LongTensor containing 1 element drawn from poisson
        """
        params = torch.exp(params)
        dist = Poisson(params)
        sample = dist.sample((1, ))
        return sample

    def init(self, x_lstg):
        x_fixed = self.composer.build_arrival_init(x_lstg)
        self.hidden = self.num_offers_model.init(x_fixed=x_fixed)

    def num_offers(self, sources=None):
        _, x_time = self.composer.build_input_vector(NUM_OFFERS, sources=sources,
                                                     fixed=False, recurrent=True)
        params, self.hidden = self.num_offers_model.simulate(x_time, hidden=self.hidden)
        sample = ArrivalInterface._poisson_sample(params)
        return proper_squeeze(sample).numpy()

    def hist(self, sources=None):
        x_fixed, _ = self.composer.build_input_vector(model_name=BYR_HIST, sources=sources,
                                                      fixed=True, recurrent=False)
        params = self.hist_model.simulate(x_fixed)
        hist = categorical_sample(params, 1)
        hist = hist / 10
        return hist
