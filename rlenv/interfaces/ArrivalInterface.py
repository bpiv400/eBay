from rlenv.env_utils import load_model, sample_categorical, sample_poisson
from rlenv.env_consts import BYR_HIST_MODEL, NUM_OFFERS_MODEL


class ArrivalInterface:
    def __init__(self, composer=None):
        # Load interface
        self.composer = composer
        self.hist_model = load_model(BYR_HIST_MODEL)
        self.num_offers_model = load_model(NUM_OFFERS_MODEL)

    def num_offers(self, sources=None):
        input_dict = self.composer.build_input_vector(NUM_OFFERS_MODEL, sources=sources)
        lnmean = self.num_offers_model(input_dict)
        return sample_poisson(lnmean.squeeze())

    def hist(self, sources=None):
        input_dict = self.composer.build_input_vector(BYR_HIST_MODEL, sources=sources)
        params = self.hist_model(input_dict)
        hist = sample_categorical(params)
        hist = hist / 10
        return hist
