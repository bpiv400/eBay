import torch
from torch.distributions.poisson import Poisson
from rlenv.interface.ModelInterface import ModelInterface
from interface.model_names import BYR_HIST, NUM_OFFERS
from rlenv.env_utils import proper_squeeze, categorical_sample
from constants import ARRIVAL_PREFIX


class ArrivalInterface:
    def __init__(self, byr_hist=0, num_offers=0, composer=None):
        # Load interface
        self.composer = composer
        self.num_offers = load_model(ARRIVAL_PREFIX, BYR_HIST, byr_hist)
        self.byr_hist = load_model(ARRIVAL_PREFIX, NUM_OFFERS, num_offers)
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
        self.hidden = self.num_offers.init(x_fixed)

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




