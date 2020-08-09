from rlenv.events.Event import Event
from rlenv.const import ARRIVAL, RL_ARRIVAL_EVENT
from rlenv.util import get_clock_feats
from utils import get_months_since_lstg
from constants import MONTH


class Arrival(Event):
    """
    Event that corresponds to processing arrivals for a listing
    for some day

    Attributes:
        priority: inherited from Event
        turn:  inherited from Event (fixed to 1)
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

    def update_arrival(self, thread_count=None, last_arrival_time=None):
        """
        :return:
        """
        update_args = dict(months_since_lstg=get_months_since_lstg(lstg_start=self.start,
                                                                   time=self.priority),
                           clock_feats=get_clock_feats(self.priority))
        if thread_count is not None:
            update_args['thread_count'] = thread_count

        if last_arrival_time is not None:
            months_since_last = (self.priority - last_arrival_time) / MONTH
            update_args['months_since_last'] = months_since_last

        self.sources.update_arrival(**update_args)
