from rlenv.events.Event import Event
from rlenv.const import ARRIVAL
from rlenv.util import get_clock_feats
from utils import get_weeks_since_lstg
from constants import WEEK
from featnames import WEEKS_SINCE_LAST, THREAD_COUNT


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
        weeks_since_lstg = get_weeks_since_lstg(lstg_start=self.start,
                                                time=self.priority)
        update_args = dict(weeks_since_lstg=weeks_since_lstg,
                           clock_feats=get_clock_feats(self.priority))
        if thread_count is not None:
            update_args[THREAD_COUNT] = thread_count

        if last_arrival_time is not None:
            weeks_since_last = (self.priority - last_arrival_time) / WEEK
            update_args[WEEKS_SINCE_LAST] = weeks_since_last

        self.sources.update_arrival(**update_args)
