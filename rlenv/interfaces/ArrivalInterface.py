from rlenv.env_utils import sample_categorical
from rlenv.env_consts import (BYR_HIST_MODEL, FIRST_ARRIVAL_MODEL,
                              INTERARRIVAL_MODEL)
from utils import load_model


class ArrivalInterface:
    def __init__(self):
        # Load interface
        self.first_arrival_model = load_model(FIRST_ARRIVAL_MODEL)
        self.interarrival_model = load_model(INTERARRIVAL_MODEL)
        self.hist_model = load_model(BYR_HIST_MODEL)

    def hist(self, input_dict=None):
        params = self.hist_model(input_dict)
        hist = sample_categorical(params)
        hist = hist / 10
        return hist

    def first_arrival(self, input_dict=None):
        model = self.interarrival_model
        return self.arrival(model=model, input_dict=input_dict)

    def inter_arrival(self, input_dict=None):
        model = self.first_arrival_model
        return self.arrival(model=model, input_dict=input_dict)

    @staticmethod
    def arrival(model=None, input_dict=None):
        params = model(input_dict)
        intervals = sample_categorical(params)
        return intervals
