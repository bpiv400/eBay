from rlenv.events.Event import Event
from rlenv.env_consts import ARRIVAL, RL_ARRIVAL_EVENT
from utils import get_months_since_lstg


class Arrival(Event):
    """
    Event that corresponds to processing arrivals for a listing
    for some day

    Attributes:
        priority: inherited from Event
    """
    def __init__(self, priority=None, sources=None, rl=False):
        """
        Constructor

        :param priority: integer time of the event
        """
        if not rl:
            event_type = ARRIVAL
        else:
            event_type = RL_ARRIVAL_EVENT
        super(Arrival, self).__init__(event_type, priority=int(priority))
        self.sources = sources
        self.start = priority

    def update_arrival(self, **kwargs):
        months_since_lstg = get_months_since_lstg(lstg_start=self.start, start=self.priority)
        self.sources.update_arrival(months_since_lstg=months_since_lstg, **kwargs)
