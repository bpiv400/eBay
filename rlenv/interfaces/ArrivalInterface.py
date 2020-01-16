import torch, numpy as np
from datetime import datetime as dt
from torch.distributions.poisson import Poisson
from rlenv.env_utils import load_model, proper_squeeze, categorical_sample
from rlenv.env_consts import BYR_HIST_MODEL, ARRIVAL_MODEL


class ArrivalInterface:
    def __init__(self, composer=None):
        # Load interface
        self.composer = composer
        self.arrival_model = load_model(ARRIVAL_MODEL)
        self.hist_model = load_model(BYR_HIST_MODEL)


    def arrival_interval(self, sources=None):
        x = self.composer.build_input_vector(ARRIVAL_MODEL, sources=sources)
        theta = self.arrival_model(x)
        arrival_interval = categorical_sample(theta)
        return arrival_interval


    def hist(self, sources=None):
        x = self.composer.build_input_vector(model_name=BYR_HIST_MODEL, sources=sources)
        theta = self.hist_model(x)
        hist = categorical_sample(theta)
        hist = hist / 10
        return hist
