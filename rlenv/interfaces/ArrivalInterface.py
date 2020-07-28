import numpy as np
from constants import BYR_HIST_MODEL, INTERARRIVAL_MODEL, PCTILE_DIR
from featnames import BYR_HIST
from rlenv.util import sample_categorical
from utils import load_model, unpickle


class ArrivalInterface:
    def __init__(self):
        # load models
        self.interarrival_model = load_model(INTERARRIVAL_MODEL)
        self.hist_model = load_model(BYR_HIST_MODEL)

        # for hist model
        s = unpickle(PCTILE_DIR + '{}.pkl'.format(BYR_HIST))
        self.hist_array = s.index.values
        self.hist_pctile = s.values

    def hist(self, input_dict=None):
        theta = self.hist_model(input_dict).cpu().squeeze().numpy()
        hist = self._draw_hist(theta)  # hist is a count
        idx = np.searchsorted(self.hist_array, hist)
        if idx == len(self.hist_pctile):
            pctile = 1.
        else:
            pctile = self.hist_pctile[idx]
        return pctile

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

    @staticmethod
    def _draw_hist(theta=None):
        # draw a random uniform for mass at 0
        pi = 1 / (1 + np.exp(-theta[0]))  # sigmoid
        if np.random.uniform() < pi:
            return 0

        # draw p of negative binomial from beta
        a = np.exp(theta[1])
        b = np.exp(theta[2])
        p = np.random.beta(a, b)

        # draw from negative binomial with n=1
        return np.random.negative_binomial(1, p)
