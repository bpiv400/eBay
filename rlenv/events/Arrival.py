import torch
from rlenv.events.Event import Event
from rlenv.events.event_types import ARRIVAL
from constants import INTERVAL_COUNTS
from rlenv.interface.model_names import ARRIVAL_PREFIX


class Arrival(Event):
    """
    Event that corresponds to processing arrivals for a listing
    for some day

    Attributes:
        priority: inherited from Event
    """
    def __init__(self, priority, sources):
        """
        Constructor

        :param priority: integer time of the event
        """
        super(Arrival, self).__init__(ARRIVAL, priority=int(priority), thread_id=None)
        self.sources = sources
        self.interval = 0

    def update_arrival(self, clock_feats=None, time_feats=None):
        duration = torch.tensor(self.interval / INTERVAL_COUNTS[ARRIVAL_PREFIX])
        duration = duration.float()
        self.sources.update_arrival(clock_feats=clock_feats,
                                    time_feats=time_feats,
                                    duration=duration)
        self.interval += 1
