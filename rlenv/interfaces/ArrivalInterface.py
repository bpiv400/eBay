from rlenv.env_utils import sample_categorical
from rlenv.env_consts import BYR_HIST_MODEL, ARRIVAL_MODEL
from utils import load_model


class ArrivalInterface:
    def __init__(self):
        # Load interface
        self.arrival_model = load_model(ARRIVAL_MODEL)
        self.hist_model = load_model(BYR_HIST_MODEL)

    def hist(self, input_dict=None):
        params = self.hist_model(input_dict)
        hist = sample_categorical(params)
        hist = hist / 10
        return hist

    def inter_arrival(self, input_dict=None):
        params = self.arrival_model(input_dict)
        intervals = sample_categorical(params)
        return intervals
