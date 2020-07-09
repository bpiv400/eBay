import torch
import numpy as np
from agent.algo.CrossEntropyPPO import CrossEntropyPPO
from constants import LISTING_FEE


class SellerPPO(CrossEntropyPPO):
    def __init__(self, ppo_params=None, econ_params=None):
        super().__init__(**ppo_params)

        # seller-specific parameters
        self.monthly_discount = econ_params['monthly_discount']
        self.action_discount = econ_params['action_discount']
        self.action_cost = econ_params['action_cost']

    def _get_return(self, months_to_sale=None, months_since_start=None,
                    sale_proceeds=None, action_diff=None):
        """
        Discounts proceeds from sale and listing fees paid.
        :param months_to_sale: months from listing start to sale
        :param months_since_start: months since start of listing
        :param sale_proceeds: sale price net of eBay cut
        :param action_diff: number of actions from current state until sale
        :return: discounted net proceeds
        """
        # discounted listing fees
        months = np.ceil(months_to_sale) - np.ceil(months_since_start) + 1
        k = months_since_start % 1
        factor = (1 - self.monthly_discount ** months) / (1 - self.monthly_discount)
        delta = (self.monthly_discount ** (1 - k)) * factor
        costs = LISTING_FEE * delta
        # add in action costs
        costs += self.action_cost * action_diff
        # discounted proceeds
        months_diff = months_to_sale - months_since_start
        assert (months_diff >= 0).all()
        sale_proceeds *= self.monthly_discount ** months_diff
        sale_proceeds *= self.action_discount ** action_diff
        return sale_proceeds - costs

    def _get_max_return(self, months_since_start=None, bin_proceeds=None):
        """
        Discounts proceeds from sale and listing fees paid.
        :param months_since_start: months since start of listing
        :param bin_proceeds: start price net of eBay cut
        :return: discounted maximum proceeds
        """
        # discounted listing fees
        k = months_since_start % 1
        costs = LISTING_FEE * (self.monthly_discount ** (1 - k))
        return bin_proceeds - costs

    def discount_return(self, reward=None, done=None, info=None):
        """
        Computes time-discounted sum of future rewards from each
        time-step to the end of the batch. Sum resets where `done`
        is 1. Operations vectorized across all trailing dimensions
        after the first [T,].
        :param tensor reward: slr's normalized gross return.
        :param tensor done: indicator for end of trajectory.
        :param namedtuple info: contains other required feats
        :return tensor return_: time-discounted return.
        """
        dtype = reward.dtype  # cast new tensors to this data type
        T, N = reward.shape  # time steps, number of environments

        # unpack info
        months = info.months.type(dtype)
        bin_proceeds = info.bin_proceeds.type(dtype)

        # initialize matrix of returns
        return_ = torch.zeros(reward.shape, dtype=dtype)

        # initialize variables that track sale outcomes
        months_to_sale = torch.tensor(1e8, dtype=dtype).expand(N)
        action_diff = torch.zeros(N, dtype=dtype)
        sale_proceeds = torch.zeros(N, dtype=dtype)

        for t in reversed(range(T)):
            # update sale outcomes when sales are observed
            months_to_sale = months_to_sale * (1 - done[t]) + months[t] * done[t]
            action_diff = (action_diff + 1) * (1 - done[t])
            sale_proceeds = sale_proceeds * (1 - done[t]) + reward[t] * done[t]

            # discounted sale proceeds
            return_[t] += self._get_return(months_to_sale=months_to_sale,
                                           months_since_start=months[t],
                                           sale_proceeds=sale_proceeds,
                                           action_diff=action_diff)

        # normalize
        max_return = self._get_max_return(months_since_start=months,
                                          bin_proceeds=bin_proceeds)
        return_ /= max_return

        return return_
