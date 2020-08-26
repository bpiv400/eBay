from rlenv.events.Event import Event
from rlenv.const import ARRIVAL
from rlenv.util import get_clock_feats
from utils import get_days_since_lstg
from constants import DAY
from featnames import DAYS_SINCE_LAST, THREAD_COUNT


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
        days_since_lstg = get_days_since_lstg(lstg_start=self.start,
                                              time=self.priority)
        update_args = dict(days_since_lstg=days_since_lstg,
                           clock_feats=get_clock_feats(self.priority))
        if thread_count is not None:
            update_args[THREAD_COUNT] = thread_count

        if last_arrival_time is not None:
            days_since_last = (self.priority - last_arrival_time) / DAY
            update_args[DAYS_SINCE_LAST] = days_since_last

        self.sources.update_arrival(**update_args)
