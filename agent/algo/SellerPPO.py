import torch
from agent.algo.CrossEntropyPPO import CrossEntropyPPO
from utils import slr_reward, max_slr_reward


class SellerPPO(CrossEntropyPPO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            return_[t] += slr_reward(months_to_sale=months_to_sale,
                                     months_since_start=months[t],
                                     sale_proceeds=sale_proceeds,
                                     action_diff=action_diff,
                                     action_discount=self.action_discount,
                                     action_cost=self.action_cost)

        # normalize
        max_return = max_slr_reward(months_since_start=months,
                                    bin_proceeds=bin_proceeds)
        return_ /= max_return

        return return_
