from rlenv.events.Event import Event
from rlenv.const import ARRIVAL, RL_ARRIVAL_EVENT
from rlenv.util import get_clock_feats
from utils import get_months_since_lstg
from constants import MONTH, DAY


class Arrival(Event):
    """
    Event that corresponds to processing arrivals for a listing
    for some day

    Attributes:
        priority: inherited from Event
    """
    def __init__(self, priority=None, sources=None, event_type=ARRIVAL):
        """
        Constructor

        :param priority: integer time of the event
        """
        super().__init__(event_type, priority=int(priority))
        self.sources = sources
        self.start = priority

    def update_arrival(self, thread_count=None, last_arrival_time=None):
        """
        :return:
        """
        months_since_lstg = get_months_since_lstg(lstg_start=self.start,
                                                  time=self.priority)
        update_args = dict(months_since_lstg=months_since_lstg,
                           clock_feats=get_clock_feats(self.priority))
        if thread_count is not None:
            update_args['thread_count'] = thread_count

        if last_arrival_time is not None:
            months_since_last = (self.priority - last_arrival_time) / MONTH
            update_args['months_since_last'] = months_since_last

        self.sources.update_arrival(**update_args)


class RlArrival(Arrival):
    def __init__(self, priority=None, sources=None):
        super().__init__(priority=priority,
                         sources=sources,
                         event_type=RL_ARRIVAL_EVENT)
        self.update_arrival()  # updates sources from priority

    def delay(self):
        self.priority += DAY
        self.update_arrival()
