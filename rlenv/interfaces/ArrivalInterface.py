import numpy as np
from constants import BYR_HIST_MODEL, INTERARRIVAL_MODEL
from rlenv.util import sample_categorical
from utils import load_model


class ArrivalInterface:
    def __init__(self):
        # Load interface
        self.interarrival_model = load_model(INTERARRIVAL_MODEL)
        self.hist_model = load_model(BYR_HIST_MODEL)

    def hist(self, input_dict=None):
        logits = self.hist_model(input_dict).cpu()
        hist = sample_categorical(logits)[0]
        hist = hist / 10
        return hist

    def inter_arrival(self, input_dict=None):
        logits = self.interarrival_model(input_dict).cpu().squeeze()
        sample = sample_categorical(logits)
        return sample

    @staticmethod
    def first_arrival(probs=None, intervals=None):
        if intervals is not None:
            probs = probs[intervals[0]:intervals[1]]
        probs = probs.values
        probs /= probs.sum()
        sample = np.random.choice(len(probs), p=probs)
        return sample
