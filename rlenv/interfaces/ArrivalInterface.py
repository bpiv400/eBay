import numpy as np, torch
from constants import ARRIVAL_PREFIX
from rlenv.env_utils import load_model, sample_categorical
from rlenv.env_consts import BYR_HIST_MODEL, ARRIVAL_MODEL


class ArrivalInterface:
    def __init__(self, composer=None):
        # Load interface
        self.composer = composer
        self.arrival_model = load_model(ARRIVAL_MODEL)
        self.hist_model = load_model(BYR_HIST_MODEL)

    def hist(self, sources=None):
        input_dict = self.composer.build_input_dict(BYR_HIST_MODEL, sources=sources, turn=None)
        params = self.hist_model(input_dict)
        hist = sample_categorical(params)
        hist = hist / 10
        return hist

    def inter_arrival(self, sources=None):
        input_dict = self.composer.build_input_dict(ARRIVAL_MODEL, sources=sources, turn=None)
        params = self.arrival_model(input_dict)
        intervals = sample_categorical(params)
        width = self.composer.intervals[ARRIVAL_PREFIX]
        return int((intervals + np.random.uniform()) * width)
