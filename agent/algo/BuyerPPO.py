import torch
from agent.algo.CrossEntropyPPO import CrossEntropyPPO
from utils import byr_reward


class BuyerPPO(CrossEntropyPPO):
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
        item_value = info.item_value.type(dtype)

        # initialize matrix of returns
        return_ = torch.zeros(reward.shape, dtype=dtype)

        # initialize variables that track sale outcomes
        action_diff = torch.zeros(N, dtype=dtype)
        net_value = torch.zeros(N, dtype=dtype)

        for t in reversed(range(T)):
            # update sale outcomes when sales are observed
            action_diff = (action_diff + 1) * (1 - done[t])
            net_value = net_value * (1 - done[t]) + reward[t] * done[t]

            # discounted sale proceeds
            return_[t] += byr_reward(net_value=net_value,
                                     action_diff=action_diff,
                                     action_discount=self.action_discount,
                                     action_cost=self.action_cost)

        # normalize by start_price
        return_ /= item_value

        return return_
