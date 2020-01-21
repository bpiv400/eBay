from constants import MONTH
from rlenv.events.Event import Event
from rlenv.env_consts import ARRIVAL
from rlenv.env_utils import time_delta
from rlenv.interfaces.ArrivalInterface import ArrivalInterface


class Arrival(Event):
    """
    Event that corresponds to processing arrivals for a listing
    for some day

    Attributes:
        priority: inherited from Event
    """
    def __init__(self, priority=None, sources=None, interface=None):
        """
        Constructor

        :param priority: integer time of the event
        """
        super(Arrival, self).__init__(ARRIVAL, priority=int(priority))
        self.sources = sources
        self.start = priority
        self.interface = interface  # type: ArrivalInterface

    def update_arrival(self, clock_feats=None, thread_count=None):
        months_since_lstg = time_delta(self.start, self.priority, unit=MONTH)
        self.sources.update_arrival(clock_feats=clock_feats, thread_count=thread_count,
                                    months_since_lstg=months_since_lstg)

    def inter_arrival(self):
        seconds = self.interface.inter_arrival(self.sources())
        return seconds
