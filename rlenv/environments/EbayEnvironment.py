import numpy as np
from rlenv.time.TimeFeatures import TimeFeatures
from rlenv.events.EventQueue import EventQueue
from rlenv.events.Arrival import Arrival
from rlenv.events import event_types
from rlenv.sources import ArrivalSources
from rlenv.sources import ThreadSources
from rlenv.env_consts import START_DAY, INTERACT, VERBOSE, MONTH
from rlenv.env_utils import get_clock_feats, time_delta
from constants import INTERVAL


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

    def run(self):
        complete = False
        lstg_complete = None
        while not complete:
            event = self.queue.pop()
            lstg_complete = self._process_event(event)
            if not lstg_complete:
                complete = self._check_complete(event)
            else:
                complete = lstg_complete
        return lstg_complete

    def _process_event(self, event):
        if INTERACT and event.type != event_types.ARRIVAL:
            input('Press Enter to continue...')
        if event.type == event_types.ARRIVAL:
            return self._process_arrival(event)
        elif event.type == event_types.FIRST_OFFER:
            return self._process_first_offer(event)
        elif event.type == event_types.BUYER_OFFER:
            return self._process_byr_offer(event)
        elif event.type == event_types.SELLER_OFFER:
            return self._process_slr_offer(event)
        elif event.type == event_types.BUYER_DELAY:
            return self._process_byr_delay(event)
        elif event.type == event_types.SELLER_DELAY:
            return self._process_slr_delay(event)
        else:
            raise NotImplementedError()

    def _process_byr_offer(self, event):
        raise NotImplementedError()

    def _process_slr_offer(self, event):
        raise NotImplementedError()

    def _process_byr_delay(self, event):
        raise NotImplementedError()

    def _process_slr_delay(self, event):
        raise NotImplementedError()

    def _process_first_offer(self, event):
        """
        Processes the buyer's first offer in a thread

        :param event:
        :return:
        """
        # expiration
        sources = ThreadSources(x_lstg=self.x_lstg, composer=self.arrival.composer)
        months_since_lstg = time_delta(self.lookup[START_DAY], event.priority, unit=MONTH)
        time_feats = self.time_feats.get_feats(time=event.priority, thread_id=event.thread_id)
        sources.prepare_hist(time_feats=time_feats, clock_feats=get_clock_feats(event.priority),
                             months_since_lstg=months_since_lstg)
        hist = self.arrival.hist(sources())
        if VERBOSE:
            print('Thread {} initiated | Buyer hist: {}'.format(event.thread_id, hist))
        event.init_thread(sources=sources, hist=hist)
        return hist

    def _process_arrival(self, event):
        """
        Updates queue with results of an Arrival Event

        :param event: Event corresponding to current event
        :return: boolean indicating whether the lstg has ended
        """
        if self._lstg_expiration(event):
            return True

        event.update_arrival(time_feats=self.time_feats.get_feats(time=event.priority),
                             clock_feats=get_clock_feats(event.priority))
        num_byrs = self.arrival.num_offers(event.sources())
        if VERBOSE and num_byrs > 0:
            print('Arrival Interval Start: {}'.format(event.priority))
            print('Number of arrivals: {}'.format(num_byrs))
        if num_byrs > 0:
            # place each into the queue
            for i in range(num_byrs[0].astype(int)):
                priority = event.priority + np.random.randint(0, INTERVAL['arrival'])
                self.thread_counter += 1
                offer_event = self.make_thread(priority)
                self.queue.push(offer_event)
        # Add arrival check
        event.priority = event.priority + INTERVAL['arrival']
        self.queue.push(event)
        return False

    def _check_complete(self, event):
        raise NotImplementedError()

    def _lstg_expiration(self, event):
        """
        Checks whether the lstg has expired by the time of the event
        If so, record the reward as negative insertion fees
        :param event: rlenv.Event subclass
        :return: boolean
        """
        if event.priority >= self.end_time:
            self.outcome = (False, 0, MONTH)
            if VERBOSE:
                print('Lstg expired')
            return True
        else:
            return False

    def make_thread(self, priority):
        raise NotImplementedError()
