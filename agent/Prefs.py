import torch


class Prefs:
    def __init__(self, delta=None):
        self.delta = delta
        self.byr = None

    def discount_return(self):
        raise NotImplementedError()

    def get_return(self):
        raise NotImplementedError()


class BuyerPrefs(Prefs):
    def __init__(self, delta=None):
        super().__init__(delta=delta)
        self.byr = True

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
        max_return = info.max_return.type(dtype)

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
            return_[t] += self.get_return(net_value=net_value,
                                          action_diff=action_diff)

        # normalize by start_price
        return_ /= max_return

        return return_

    def get_return(self, net_value=None, action_diff=None):
        """
        Discounts proceeds from sale and listing fees paid.
        :param net_value: value less price paid; 0 if no sale
        :param action_diff: number of actions from current state until sale
        :return: discounted net proceeds
        """
        return net_value * (self.delta ** action_diff)


class SellerPrefs(Prefs):
    def __init__(self, delta=None):
        super().__init__(delta=delta)
        self.byr = False

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

        # recast
        done = done.type(torch.int)
        max_return = info.max_return.type(dtype)

        # initialize matrix of returns
        return_ = torch.zeros(reward.shape, dtype=dtype)

        # initialize variables that track sale outcomes
        t_sale = torch.tensor(1e8, dtype=dtype).expand(N)
        sale_proceeds = torch.zeros(N, dtype=dtype)

        # to be later marked valid=False
        censored = torch.zeros(reward.shape, dtype=torch.bool)

        for t in reversed(range(T)):
            # time until sale
            t_sale = t_sale * (1 - done[t]) + info.months[t] * done[t]
            months_to_sale = (t_sale - info.months_last[t]).type(dtype)

            # price paid at sale, less fees
            sale_proceeds = sale_proceeds * (1 - done[t]) + reward[t] * done[t]
            sale_proceeds = torch.min(sale_proceeds, max_return[t])  # precision

            # discounted sale proceeds
            return_[t] += self.get_return(months_to_sale=months_to_sale,
                                          sale_proceeds=sale_proceeds)

            # offers made after listing sells
            censored[t] = months_to_sale < 0

        # normalize
        return_ /= max_return

        return return_, censored

    def get_return(self, months_to_sale=None, sale_proceeds=None):
        """
        Discounts proceeds from sale, net of fees.
        :param months_to_sale: months from now until sale
        :param sale_proceeds: sale price net of eBay cut and listing fees
        :return: discounted net proceeds
        """
        return (self.delta ** months_to_sale) * sale_proceeds
