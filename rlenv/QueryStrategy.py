import numpy as np
from utils import init_optional_arg
from constants import INTERVAL_TURN, MAX_DELAY_TURN


class QueryStrategy:

    def get_hist(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_con(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_delay(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_msg(self, *args, **kwargs):
        raise NotImplementedError("")


class DefaultQueryStrategy(QueryStrategy):
    def __init__(self, buyer=None, seller=None):
        super().__init__()
        self.buyer = buyer
        self.seller = seller

    def get_con(self, *args, **kwargs):
        if kwargs['turn'] % 2 == 0:
            con = self.seller.con(input_dict=kwargs['input_dict'],
                                  turn=kwargs['turn'])
        else:
            con = self.buyer.con(input_dict=kwargs['input_dict'],
                                 turn=kwargs['turn'])
        return con

    def get_msg(self, *args, **kwargs):
        if kwargs['turn'] % 2 == 0:
            msg = self.seller.msg(input_dict=kwargs['input_dict'],
                                  turn=kwargs['turn'])
        else:
            msg = self.buyer.msg(input_dict=kwargs['input_dict'],
                                 turn=kwargs['turn'])
        return msg

    def get_hist(self, *args, **kwargs):
        return self.buyer.hist(input_dict=kwargs['input_dict'])

    def get_delay(self, *args, **kwargs):
        init_optional_arg(kwargs=kwargs, name='max_interval', default=None)
        if kwargs['turn'] % 2 == 0:
            index = self.seller.delay(input_dict=kwargs['input_dict'],
                                      turn=kwargs['turn'],
                                      max_interval=kwargs['max_interval'])
        else:
            index = self.buyer.delay(input_dict=kwargs['input_dict'],
                                     turn=kwargs['turn'],
                                     max_interval=kwargs['max_interval'])
        seconds = int((index + np.random.uniform()) * INTERVAL_TURN)
        seconds = max(1, min(seconds, MAX_DELAY_TURN))
        return seconds
