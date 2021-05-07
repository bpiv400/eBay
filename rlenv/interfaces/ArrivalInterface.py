import numpy as np
from featnames import BYR_HIST, INTERARRIVAL_MODEL, BYR_HIST_MODEL
from rlenv.util import sample_categorical
from utils import load_model, load_pctile


class ArrivalInterface:
    def __init__(self):
        # load models
        self.interarrival_model = load_model(INTERARRIVAL_MODEL)
        self.hist_model = load_model(BYR_HIST_MODEL)

        # for hist model
        s = load_pctile(name=BYR_HIST)
        s = s.reindex(index=range(s.index.max() + 1), method='pad')
        self.hist_pctile = s

    def hist(self, input_dict=None):
        theta = self.hist_model(input_dict).cpu().squeeze().numpy()
        hist = self._draw_hist(theta)  # hist is a count
        if hist > self.hist_pctile.index[-1]:
            pctile = 1.
        else:
            pctile = self.hist_pctile.loc[hist]
        return pctile

    def inter_arrival(self, input_dict=None):
        logits = self.interarrival_model(input_dict).cpu().squeeze()
        sample = sample_categorical(logits=logits)
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

        # r for negative binomial, of at least 1
        r = np.exp(theta[3]) + 1

        # draw from negative binomial
        return np.random.negative_binomial(r, p)
