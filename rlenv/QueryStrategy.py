"""
Abstract superclass for objects which
determine whether models or testing logs are used
to determine turn and arrival outcomes
"""


class QueryStrategy(object):
    def __init__(self):
        self.p_arrival = None

    def update_p_arrival(self, p_arrival=None):
        self.p_arrival = p_arrival

    def get_first_arrival(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_inter_arrival(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_hist(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_con(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_delay(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_msg(self, *args, **kwargs):
        raise NotImplementedError("")