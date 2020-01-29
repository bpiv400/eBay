import math
import numpy as np
from rlenv.env_utils import get_cut
from rlenv.env_consts import LISTING_FEE, SE_TOL, MIN_SALES
from featnames import META


class ValueCalculator:
    def __init__(self, lookup):
        self.x_sum = 0.0
        self.x2_sum = 0.0
        self.T_sum = 0
        self.T2_sum = 0
        self.xT_sum = 0.0
        self.num_sales = 0
        self.cut = get_cut(lookup[META])

    def add_outcome(self, price, T):
        self.x_sum += price
        self.x2_sum += math.pow(price, 2)
        self.T_sum += T
        self.T2_sum += math.pow(T, 2)
        self.xT_sum += price * T
        self.num_sales += 1

    @property
    def mean_x(self):
        return self.x_sum / self.num_sales

    @property
    def var_x(self):
        var = self.x2_sum / self.num_sales - math.pow(self.mean_x, 2)
        return max(0.0, var)

    @property
    def mean_T(self):
        return self.T_sum / self.num_sales

    @property
    def var_T(self):
        var = self.T2_sum / self.num_sales - math.pow(self.mean_T, 2)
        return max(0.0, var)

    @property
    def cov_xT(self):
        return self.xT_sum / self.num_sales - self.mean_x * self.mean_T
    
    @property
    def value(self):
        if self.num_sales == 0:
            raise RuntimeError("No sales, value undefined")
        proceeds = (1-self.cut) * self.mean_x
        fees = LISTING_FEE * self.mean_T
        return proceeds - fees

    @property
    def var(self):
        if self.num_sales == 0:
            raise RuntimeError("No sales, value undefined")

        return math.pow(1-self.cut, 2) * self.var_x \
            + math.pow(LISTING_FEE, 2) * self.var_T \
            - 2 * (1-self.cut) * LISTING_FEE * self.cov_xT

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def se(self):
        return self.std / np.sqrt(self.num_sales)

    @property
    def stabilized(self):
        return self.num_sales >= MIN_SALES and self.se < SE_TOL

    @property
    def p_sale(self):
        return 1 / self.mean_T

    @property
    def trials_until_stable(self):
        if self.num_sales == 0:
            raise RuntimeError("No sales")
        else:
            min_sales = self.var / math.pow(SE_TOL, 2)
            diff = min_sales - self.num_sales
            print('diff: {}'.format(diff))
            return int(diff / self.p_sale)