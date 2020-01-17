import torch
from rlenv.events.Event import Event
from rlenv.env_consts import ARRIVAL, INTERVAL_COUNT
from constants import ARRIVAL_PREFIX


class Arrival(Event):
    """
    Event that corresponds to processing arrivals for a listing
    for some day

    Attributes:
        priority: inherited from Event
    """
    def __init__(self, priority=None, sources=None):
        """
        Constructor

        :param priority: integer time of the event
        """
        super(Arrival, self).__init__(ARRIVAL, priority=int(priority))
        self.sources = sources

    def update_arrival(self, clock_feats=None, time_feats=None):
        self.sources.update_arrival(clock_feats=clock_feats,
                                    time_feats=time_feats)
