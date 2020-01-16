import torch
from rlenv.events.Event import Event
from rlenv.env_consts import ARRIVAL, INTERVAL
from constants import ARRIVAL_PREFIX, MAX_DELAY


class Arrival(Event):
    """
    Event that corresponds to processing arrivals for a listing
    for some day

    Attributes:
        priority: inherited from Event
    """
    def __init__(self, priority=None, sources=None, interval_attrs=None):
        """
        Constructor

        :param priority: integer time of the event
        """
        super(Arrival, self).__init__(ARRIVAL, priority=int(priority))
        self.sources = sources
        self.lstg_start = priority
        self.last = priority
        self.spi = interval_attrs[INTERVAL][ARRIVAL_PREFIX]


    def update_arrival(self, clock_feats=None, thread_count=None):
        months_since_lstg = (self.priority - self.lstg_start) / MONTH
        months_since_last = (self.priority - self.last) / MONTH
        self.sources.update_arrival(clock_feats=clock_feats,
                                    months_since_lstg=months_since_lstg,
                                    months_since_last=months_since_last,
                                    thread_count=thread_count)
        self.last = priority