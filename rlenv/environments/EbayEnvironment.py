from rlenv.time.TimeFeatures import TimeFeatures
from rlenv.events.EventQueue import EventQueue
from rlenv.events.Arrival import Arrival
from rlenv.sources import ArrivalSources
from rlenv.env_consts import START_DAY


class EbayEnvironment:
    def __init__(self, arrival):
        # arrival process interface
        self.arrival = arrival

        # features
        self.x_lstg = None
        self.lookup = None

        self.time_feats = TimeFeatures()
        # queue
        self.queue = EventQueue()

        # end time
        self.end_time = None
        self.thread_counter = 0

    def reset(self):
        self.queue.reset()
        self.time_feats.reset()
        self.thread_counter = 0
        sources = ArrivalSources(x_lstg=self.x_lstg, composer=self.arrival.composer)
        self.arrival.init(sources=sources())
        self.queue.push(Arrival(self.lookup[START_DAY], sources))


