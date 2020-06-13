from constants import MONTH
from rlenv.events.Event import Event
from rlenv.constants import ARRIVAL
from rlenv.utils import time_delta
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from util import get_months_since_lstg


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
        self.start = priority

    def update_arrival(self, clock_feats=None, thread_count=None):
        months_since_lstg = get_months_since_lstg(lstg_start=self.start, start=self.priority)
        self.sources.update_arrival(clock_feats=clock_feats, thread_count=thread_count,
                                    months_since_lstg=months_since_lstg)

    def inter_arrival(self):
        seconds = self.interface.inter_arrival(self.sources())
        return seconds
