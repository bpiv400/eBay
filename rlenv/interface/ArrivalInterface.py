import torch
from torch.distributions.poisson import Poisson
from rlenv.interface.ModelInterface import ModelInterface
from interface.model_names import BYR_HIST, NUM_OFFERS


class ArrivalInterface(ModelInterface):
    def __init__(self, byr_hist=0, num_offers=0, composer=0):
        # Load interface
        super(ArrivalInterface, self).__init__(composer)
        self.num_offers = ArrivalInterface._load_model(BYR_HIST, byr_hist)
        self.byr_hist = ArrivalInterface._load_model(NUM_OFFERS, num_offers)
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
        sample = dist.sample()
        return sample

    def init(self, x_lstg):
        x_fixed = self.composer.build_arrival_init(x_lstg)
        self.hidden = self.num_offers.init(x_fixed)

    def step(self, sources=None):
        num_offers = self._get_num_offers(sources=sources)
        if num_offers > 0:
            hist = self._get_hist(sources=sources, num_offers=num_offers)
        else:
            hist = None
        return hist, num_offers

    def _get_num_offers(self, sources=None):
        _, x_time = self.composer.build_input_vector(NUM_OFFERS, sources=sources,
                                                     fixed=False, recurrent=True,
                                                     size=1)
        params, self.hidden = self.num_offers.simulate(x_time, hidden=self.hidden)
        params = ArrivalInterface.proper_squeeze(params)
        sample = ArrivalInterface._poisson_sample(params)
        return sample

    def _get_hist(self, sources=None, num_offers=None):
        x_fixed, _ = self.composer.build_input_vector(model_name=BYR_HIST, sources=sources,
                                                      fixed=True, recurrent=False, size=1)
        params = self.byr_hist.simulate(x_fixed)
        hist = ModelInterface._categorical_sample(params, num_offers)
        hist = hist.float() / 10
        return hist




