from rlenv.time.TimeFeatures import TimeFeatures
from rlenv.events.EventQueue import EventQueue

class RewardEnvironment:
    def __init__(self, **kwargs):
        # environment models
        self.buyer = kwargs['buyer']
        self.seller = kwargs['seller']
        self.arrival = kwargs['arrival']

        # features
        self.x_lstg = kwargs['x_lstg']
        self.lookup = kwargs['lookup']
        self.time_feats = TimeFeatures()

        # queue
        self.queue = EventQueue()

        # end time
        self.end_time =

    def reset(self):
        self.queue.clear()
        self.time_feats.clear()
