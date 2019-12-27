import torch, numpy as np
from datetime import datetime as dt
from scipy.stats import poisson
from rlenv.env_utils import load_model, proper_squeeze, categorical_sample
from rlenv.env_consts import BYR_HIST_MODEL, NUM_OFFERS_MODEL


class ArrivalInterface:
    def __init__(self, composer=None):
        # Load interface
        self.composer = composer
        self.hist_model = load_model(BYR_HIST_MODEL)
        self.num_offers_model = load_model(NUM_OFFERS_MODEL)
        self.hidden = None


    @staticmethod
    def _poisson_sample(lnmean):
        """
        Draws a sample from a poisson distribution parameterized by
        the input tensor according to documentation in ebay/documents

        :param params: 1-dimensional torch.tensor output by a poisson model
        :return: torch.LongTensor containing 1 element drawn from poisson
        """
        return poisson.rvs(np.exp(lnmean))


    def init(self, sources=None):
        input_dict = self.composer.build_input_vector(NUM_OFFERS_MODEL, sources=sources,
                                                      fixed=True, recurrent=False)
        self.hidden = self.num_offers_model.init(x=input_dict['x'])

    def num_offers(self, sources=None):
        input_dict = self.composer.build_input_vector(NUM_OFFERS_MODEL, sources=sources,
                                                      fixed=False, recurrent=True)
        lnmean, self.hidden = self.num_offers_model.step(x_time=input_dict['x_time'],
                                                         hidden=self.hidden)
        
        return ArrivalInterface._poisson_sample(lnmean.squeeze().numpy())


    def hist(self, sources=None):
        input_dict = self.composer.build_input_vector(model_name=BYR_HIST_MODEL, sources=sources,
                                                      fixed=True, recurrent=False)
        params = self.hist_model(input_dict['x'])
        hist = categorical_sample(params, 1)
        hist = hist / 10
        return hist
