import numpy as np
from constants import ARRIVAL_PREFIX
from rlenv.env_utils import load_model, sample_categorical
from rlenv.env_consts import BYR_HIST_MODEL, ARRIVAL_MODEL, INTERVAL


class ArrivalInterface:
    def __init__(self, composer=None):
        # Load interface
        self.composer = composer
        self.hist_model = load_model(BYR_HIST_MODEL)
        self.arrival_model = load_model(ARRIVAL_MODEL)

    def next_buyer(self, sources=None):
        input_dict = self.composer.build_input_dict(ARRIVAL_MODEL, sources=sources)
        params = self.arrival_model(input_dict)
        return sample_categorical(params)

    def hist(self, sources=None):
        input_dict = self.composer.build_input_dict(BYR_HIST_MODEL, sources=sources)
        params = self.hist_model(input_dict)
        hist = sample_categorical(params)
        hist = hist / 10
        return hist

    def inter_arrival(self, index):
        interval = self.composer.interval_attrs[INTERVAL][ARRIVAL_PREFIX]
        return int(np.random.randint((index - 1) * interval, (index * interval) - 1))
