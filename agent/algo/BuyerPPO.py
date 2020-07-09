import torch
from agent.algo.CrossEntropyPPO import CrossEntropyPPO


class BuyerPPO(CrossEntropyPPO):
    def __init__(self, ppo_params=None, econ_params=None):
        super().__init__(**ppo_params)

        # buyer-specific parameters
        self.action_discount = econ_params['action_discount']
        self.action_cost = econ_params['action_cost']

    def _get_return(self, net_value=None, action_diff=None):
        """
        Discounts proceeds from sale and listing fees paid.
        :param net_value: value less price paid; 0 if no sale
        :param action_diff: number of actions from current state until sale
        :return: discounted net proceeds
        """
        net_value *= self.action_discount ** action_diff
        net_value -= self.action_cost * action_diff
        return net_value

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
            return_[t] += self._get_return(net_value=net_value,
                                           action_diff=action_diff)

        # normalize by start_price
        return_ /= item_value

        return return_
