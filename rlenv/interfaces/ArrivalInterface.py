from constants import (BYR_HIST_MODEL, FIRST_ARRIVAL_MODEL,
                       INTERARRIVAL_MODEL)
from rlenv.util import sample_categorical
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

    def first_arrival(self, input_dict=None, intervals=None):
        """
        :param input_dict: standard model input dictionary
        :param intervals: optional 2-tuple of integers that gives
        the intervals to sample over (inclusive of the first,
        exclusive of the second)
        """
        model = self.first_arrival_model
        return self.arrival(model=model, input_dict=input_dict,
                            intervals=intervals)

    def inter_arrival(self, input_dict=None):
        model = self.interarrival_model
        return self.arrival(model=model, input_dict=input_dict)

    @staticmethod
    def arrival(model=None, input_dict=None, intervals=None):
        params = model(input_dict).squeeze()
        if intervals is not None:
            params = params[intervals[0]:intervals[1]]
        intervals = sample_categorical(params)
        return intervals
