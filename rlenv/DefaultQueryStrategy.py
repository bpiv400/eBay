import numpy as np
from utils import init_optional_arg
from inputs.const import INTERVAL_ARRIVAL, INTERVAL_TURN, MAX_DELAY_TURN
from rlenv.QueryStrategy import QueryStrategy


class DefaultQueryStrategy(QueryStrategy):
    def __init__(self, buyer=None, seller=None, arrival=None):
        self.buyer = buyer
        self.seller = seller
        self.arrival = arrival

    def get_con(self, *args, **kwargs):
        if kwargs['turn'] % 2 == 0:
            con = self.seller.con(input_dict=kwargs['input_dict'], turn=kwargs['turn'])
        else:
            con = self.buyer.con(input_dict=kwargs['input_dict'], turn=kwargs['turn'])
        return con

    def get_msg(self, *args, **kwargs):
        if kwargs['turn'] % 2 == 0:
            msg = self.seller.msg(input_dict=kwargs['input_dict'], turn=kwargs['turn'])
        else:
            msg = self.buyer.msg(input_dict=kwargs['input_dict'], turn=kwargs['turn'])
        return msg

    def get_arrival(self, *args, **kwargs):
        init_optional_arg(kwargs=kwargs, name='first', default=False)
        init_optional_arg(kwargs=kwargs, name='intervals', default=None)
        if kwargs['first']:
            intervals = self.arrival.first_arrival(input_dict=kwargs['input_dict'],
                                                   intervals=kwargs['intervals'])
        else:
            intervals = self.arrival.inter_arrival(input_dict=kwargs['input_dict'])
        return int((intervals + np.random.uniform()) * INTERVAL_ARRIVAL)

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
