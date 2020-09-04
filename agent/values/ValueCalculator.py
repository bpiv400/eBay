import numpy as np
from constants import LISTING_FEE, EPS

MIN_CT = 10     # run for at least this many simulations
SE_TOL = .01    # run until standard error is less than this


class ValueCalculator:
    def __init__(self, cut=None, start_price=None):
        self.cut = cut
        self.start_price = start_price

        # initialize sums and counts
        self.x_sum = 0.
        self.x2_sum = 0.
        self.ct = 0
        self.sales = 0

    def add_outcome(self, price):
        self.ct += 1
        if price > 0.:
            self.sales += 1
            self.x_sum += price
            self.x2_sum += price ** 2

    @property
    def mean_x(self):
        return self.x_sum / self.ct

    @property
    def var_x(self):
        return self.x2_sum / self.ct - (self.mean_x ** 2)
    
    @property
    def value(self):
        return ((1-self.cut) * self.mean_x - LISTING_FEE) / self.start_price

    @property
    def var(self):
        var = (1-self.cut) ** 2 * self.var_x / (self.start_price ** 2)
        return max(0., var)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def se(self):
        return self.std / np.sqrt(self.ct)

    @property
    def stabilized(self):
        if self.ct >= MIN_CT:
            return EPS < self.mean_x < self.start_price and EPS < self.se < SE_TOL
        else:
            return False
