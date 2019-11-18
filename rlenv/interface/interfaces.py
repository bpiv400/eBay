import torch
from torch.distributions import Poisson
from rlenv.env_utils import load_model, proper_squeeze, categorical_sample
from rlenv.interface.model_names import (model_str, CON, MSG, DELAY,
                                         BYR_HIST, NUM_OFFERS)


class PlayerInterface:
    def __init__(self, msg=0, con=0, delay=0, composer=None, byr=False):
        #composer
        self.composer = composer
        #store names for each
        self.con_model_name = model_str(CON, byr=byr)
        self.msg_model_name = model_str(MSG, byr=byr)
        self.delay_model_name = model_str(DELAY, byr=byr)
        # load models
        self.msg_model = load_model(self.msg_model_name, msg)
        self.con_model = load_model(self.con_model_name, con)
        self.delay_model = load_model(self.delay_model_name, delay)

    def con(self, sources=None, hidden=None, turn=0):
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(self.con_model_name,
                                                           sources=sources,
                                                           recurrent=True, fixed=fixed)
        params, hidden = self.con_model.simulate(x_time, x_fixed=x_fixed, hidden=hidden,
                                                 turn=turn)
        return params, hidden

    def msg(self, sources=None, hidden=None, turn=0):
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(self.msg_model_name,
                                                           sources=sources,
                                                           recurrent=True, fixed=fixed)
        params, hidden = self.msg_model.simulate(x_time, x_fixed=x_fixed, hidden=hidden,
                                                 turn=turn)
        return params, hidden

    def delay(self, sources=None, hidden=None, turn=0):
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(self.msg_model_name,
                                                           sources=sources,
                                                           recurrent=True, fixed=fixed)
        params, hidden = self.delay_model.simulate(x_time, x_fixed=x_fixed, hidden=hidden,
                                                   turn=turn)
        return params, hidden

    def init_delay(self, sources=None):
        x_fixed, _ = self.composer.build_input_vector(self.delay_model_name,
                                                      sources=sources,
                                                      recurrent=False, fixed=True)
        hidden = self.delay_model.init(x_fixed=x_fixed)
        return hidden


class ArrivalInterface:
    def __init__(self, byr_hist=0, num_offers=0, composer=None):
        # Load interface
        self.composer = composer
        self.num_offers = load_model(BYR_HIST, byr_hist)
        self.byr_hist = load_model(NUM_OFFERS, num_offers)
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
        self.hidden = self.num_offers.init(x_fixed=x_fixed)

    def num_offers(self, sources=None):
        _, x_time = self.composer.build_input_vector(NUM_OFFERS, sources=sources,
                                                     fixed=False, recurrent=True,
                                                     size=1)
        params, self.hidden = self.num_offers.simulate(x_time, hidden=self.hidden)
        sample = ArrivalInterface._poisson_sample(params)
        return proper_squeeze(sample)

    def hist(self, sources=None):
        x_fixed, _ = self.composer.build_input_vector(model_name=BYR_HIST, sources=sources,
                                                      fixed=True, recurrent=False, size=1)
        params = self.byr_hist.simulate(x_fixed)
        hist = categorical_sample(params, 1)
        hist = hist / 10
        return hist
