import math
import numpy as np
from rlenv.env_utils import get_cut
from rlenv.env_consts import META, ANCHOR_STORE_INSERT, SE_TOLS, SE_RELAX_WIDTH, MIN_SALES


class ValueCalculator:
    def __init__(self, lookup):
        self.sale_sum = 0
        self.exp_count = 0
        self.cut = 0
        self.sales = []
        self.cut = get_cut(lookup[META])
        self.se_tol = SE_TOLS[0]
        self.tol_counter = 1

    def add_outcome(self, sale, price):
        if sale:
            self.sale_sum += price
            self.sales.append(price)
        self.exp_count += 1
        self.update_tol()

    def update_tol(self):
        if self.exp_count % SE_RELAX_WIDTH == 0 and self.tol_counter != len(SE_TOLS):
            self.se_tol = SE_TOLS[self.tol_counter]
            self.tol_counter += 1

    @property
    def mean(self):
        if len(self.sales) == 0:
            raise RuntimeError("No sales, value undefined")
        # rate of no sale
        p = 1 - len(self.sales) / self.exp_count
        avg = self.sale_sum / len(self.sales)
        return (1 - self.cut) * avg - ANCHOR_STORE_INSERT / (1 - p)

    @property
    def var(self):
        if len(self.sales) == 0:
            raise RuntimeError("No sales, value undefined")
        return math.pow((1 - self.cut), 2) * np.var(self.sales)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def mean_se(self):
        return self.std / np.sqrt(len(self.sales))

    @property
    def has_sales(self):
        return len(self.sales) > 0

    @property
    def stabilized(self):
        if len(self.sales) < MIN_SALES:
            return False
        else:
            return self.mean_se < self.se_tol

    @property
    def p(self):
        rate_sale = len(self.sales) / self.exp_count
        return 1 - rate_sale

    @property
    def trials_until_stable(self):
        if not self.has_sales:
            raise RuntimeError("No sales")
        else:
            min_sales = self.var / math.pow(self.se_tol, 2)
            diff = min_sales - len(self.sales)
            print('diff: {}'.format(diff))
            p_sale = len(self.sales) / self.exp_count
            return int(diff / p_sale)
