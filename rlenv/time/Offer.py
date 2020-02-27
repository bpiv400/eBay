from random import random
from rlenv.env_consts import BYR_PREFIX, SLR_PREFIX
from rlenv.time.offer_types import BYR_REJECTION, SLR_REJECTION, OFFER


class Offer:
    def __init__(self, params=None, rej=False, accept=False):
        # set offer type
        if rej and params['player'] == BYR_PREFIX:
            self.otype = BYR_REJECTION
        elif rej and params['player'] == SLR_PREFIX:
            self.otype = SLR_REJECTION
        else:
            self.otype = OFFER
        # store params and make sure it contains necessary features
        self.params = params
        self.accept = accept
        self.reject = rej
        assert 'time' in self.params
        assert 'thread_id' in self.params

    @property
    def price(self):
        return self.params['price']

    @property
    def time(self):
        return self.params['time']

    @property
    def censored(self):
        if 'censored' not in self.params:
            return False
        else:
            return self.params['censored']

    @property
    def thread_id(self):
        return self.params['thread_id']

    @property
    def player(self):
        return self.params['player']

    def __lt__(self, other):
        """
        Overrides less-than comparison method to make comparisons
        based on priority alone

        Throws RuntimeError if both have the same lstg and same time
        """
        if self.time == other.time:
            return random() >= .5
        else:
            return self.time < other.time

    def __str__(self):
        rep = 'price: {} | time: {} | player: {}'.format(self.price, self.time, self.player)
        return rep


