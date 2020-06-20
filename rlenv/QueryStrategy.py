"""
Abstract superclass for objects which
determine whether models or testing logs are used
to determine turn and arrival outcomes
"""


class QueryStrategy(object):
    def get_arrival(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_hist(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_con(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_delay(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_msg(self, *args, **kwargs):
        raise NotImplementedError("")