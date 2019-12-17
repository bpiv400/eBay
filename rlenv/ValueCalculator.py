import math
import numpy as np
from rlenv.env_utils import get_cut
from rlenv.env_consts import META, ANCHOR_STORE_INSERT


class ValueCalculator:
    def __init__(self, tol, lookup):
        self.sale_sum = 0
        self.exp_count = 0
        self.cut = 0
        self.sales = []
        self.cut = get_cut(lookup[META])
        self.val_se_tol = tol

    def add_outcome(self, sale, price):
        if sale:
            self.sale_sum += price
            self.sales.append(price)
        self.exp_count += 1

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
        if len(self.sales) < 5:
            return False
        else:
            return self.mean_se < self.val_se_tol

    @property
    def p(self):
        rate_sale = len(self.sales) / self.exp_count
        return 1 - rate_sale

    @property
    def trials_until_stable(self):
        if not self.has_sales:
            raise RuntimeError("No sales")
        elif len(self.sales) < 5:
            return 10
        else:
            min_sales = self.var / math.pow(self.val_se_tol, 2)
            diff = min_sales - len(self.sales)
            print('diff: {}'.format(diff))
            p_sale = len(self.sales) / self.exp_count
            return diff / p_sale
