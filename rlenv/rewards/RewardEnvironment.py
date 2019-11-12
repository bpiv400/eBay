import numpy as np
from rlenv.time.TimeFeatures import TimeFeatures
from rlenv.events.EventQueue import EventQueue
from rlenv.env_consts import MONTH, HOUR
from rlenv.events.Arrival import Arrival
from events.event_types import *
from rlenv.Sources import Sources
from rlenv.events.FirstOffer import FirstOffer

class RewardEnvironment:
    def __init__(self, **kwargs):
        # environment interface
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
        self.end_time = self.lookup['start_date'] + MONTH
        self.outcome = None
        self.thread_counter = 0

    def reset(self):
        self.queue.reset()
        self.time_feats.reset()
        self.arrival.init(self.x_lstg)
        self.thread_counter = 0
        sources = Sources(num_offers=True, start_date=self.lookup['start_date'], x_lstg=self.x_lstg)
        self.queue.push(Arrival(self.lookup['start_date'], sources))

    def run(self):
        """
        Runs a simulation of a single lstg until sale or expiration

        :return: a 3-tuple of (bool, float, int) giving whether the listing sells,
        the amount it sells for if it sells, and the amount of time it took to sell
        """
        complete = False
        while not complete:
            complete = self._process_event(self.queue.pop())
        return self.outcome

    def _process_event(self, event):
        if event.type == ARRIVAL:
            return self._process_arrival(event)
        elif event.type == FIRST_OFFER:
            return self._process_first_offer(event)
        elif event.type == BUYER_OFFER:
            return self._process_offer(event, byr=True)
        elif event.type == SELLER_OFFER:
            return self._process_offer(event, byr=False)
        elif event.type == BUYER_DELAY:
            return self._process_delay(event, byr=True)
        elif event.type == SELLER_DELAY:
            return self._process_delay(event, byr=False)

    def _process_arrival(self, event):
        """
        Updates queue with results of an Arrival Event

        :param event: Event corresponding to current event
        :return: boolean indicating whether the lstg has ended
        """
        if self._lstg_expiration(event):
            return True

        event.sources.update_arrival(time_feats=self.time_feats.get_feats(time=event.priority),
                                     time=event.priority)
        num_byrs, byr_hist = self.arrival.step(event.sources())

        if num_byrs > 0:
            # place each into the queue
            for i in range(num_byrs):
                curr_hist = byr_hist[i]
                priority = event.priority + np.random.randint(0, HOUR)
                self.thread_counter += 1
                offer_event = FirstOffer(priority, hist=curr_hist,
                                         thread_id=self.thread_counter)
                self.queue.push(offer_event)
        # Add arrival check
        event.priority = event.priority + HOUR
        self.queue.push(event)
        return False

    def _process_first_offer(self, event):
        """
        Processes the buyer's first offer in a thread

        :param event:
        :return:
        """
        # expiration
        if self._lstg_expiration(event):
            return True
        sources = Sources(num_offers=False, start_date=self.lookup['start_date'])
        sources.init_offer()

        hidden = self._make_hidden()
        event.sources = sources
        event.hidden = hidden
        return self._process_offer(event, byr=True)

    def _lstg_expiration(self, event):
        """
        Checks whether the lstg has expired by the time of the event
        If so, record the reward as negative insertion fees
        :param event: rlenv.Event subclass
        :return: boolean
        """
        if event.priority >= self.end_time:
            self.outcome = (False, 0, MONTH)
            return True
        else:
            return False
