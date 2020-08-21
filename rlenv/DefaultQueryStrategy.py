import numpy as np
from utils import init_optional_arg
from constants import INTERVAL_ARRIVAL, INTERVAL_TURN, MAX_DELAY_TURN
from rlenv.QueryStrategy import QueryStrategy


class DefaultQueryStrategy(QueryStrategy):
    def __init__(self, buyer=None, seller=None, arrival=None):
        super().__init__()
        self.buyer = buyer
        self.seller = seller
        self.arrival = arrival

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

    @staticmethod
    def _arrival_interval_to_seconds(interval=None):
        return int((interval + np.random.uniform()) * INTERVAL_ARRIVAL)

    def get_first_arrival(self, *args, **kwargs):
        init_optional_arg(kwargs=kwargs, name='intervals', default=None)
        interval = self.arrival.first_arrival(intervals=kwargs['intervals'],
                                              probs=self.p_arrival)
        return self._arrival_interval_to_seconds(interval=interval)

    def get_inter_arrival(self, *args, **kwargs):
        interval = self.arrival.inter_arrival(input_dict=kwargs['input_dict'])
        return self._arrival_interval_to_seconds(interval=interval)

    def get_hist(self, *args, **kwargs):
        return self.arrival.hist(input_dict=kwargs['input_dict'])

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
        seconds = min(seconds, MAX_DELAY_TURN)
        return seconds
