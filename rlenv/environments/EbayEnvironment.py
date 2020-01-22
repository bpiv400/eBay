from collections import namedtuple
import numpy as np
from constants import BYR_PREFIX, MONTH
from featnames import START_TIME, ACC_PRICE, DEC_PRICE, START_PRICE
from rlenv.Heap import Heap
from rlenv.time.TimeFeatures import TimeFeatures
from rlenv.time.Offer import Offer
from rlenv.events.Event import Event
from rlenv.events.Arrival import Arrival
from rlenv.sources import ArrivalSources
from rlenv.sources import ThreadSources
from rlenv.env_consts import INTERACT, SALE, PRICE, DUR, ACC_IND, \
    REJ_IND, OFF_IND, ARRIVAL, FIRST_OFFER, BUYER_OFFER, SELLER_OFFER, \
    BUYER_DELAY, SELLER_DELAY
from rlenv.env_utils import time_delta, get_clock_feats


class EbayEnvironment:
    Outcome = namedtuple('outcome', [SALE, PRICE, DUR])

    def __init__(self, arrival, verbose):
        # arrival process interface
        self.arrival = arrival
        self.verbose = verbose

        # features
        self.x_lstg = None
        self.lookup = None

        self.time_feats = TimeFeatures()
        # queue
        self.queue = Heap(entry_type=Event)

        # end time
        self.end_time = None
        self.thread_counter = 0
        self.outcome = None

        # interval data
        self.intervals = self.arrival.composer.intervals

    def reset(self):
        self.queue.reset()
        self.time_feats.reset()
        self.outcome = None
        self.thread_counter = 0
        sources = ArrivalSources(x_lstg=self.x_lstg)
        event = Arrival(priority=self.lookup[START_TIME], sources=sources, interface=self.arrival)
        self.queue.push(event)

    def run(self):
        while True:
            event = self.queue.pop()
            lstg_complete = self._process_event(event)
            if lstg_complete:
                return True
            if self._check_complete(event):
                return False

    def _process_event(self, event):
        if INTERACT and event.type != ARRIVAL:
            input('Press Enter to continue...')
        if event.type == ARRIVAL:
            return self._process_arrival(event)
        elif event.type == FIRST_OFFER:
            return self._process_first_offer(event)
        elif event.type == BUYER_OFFER:
            return self._process_byr_offer(event)
        elif event.type == SELLER_OFFER:
            return self._process_slr_offer(event)
        elif event.type == BUYER_DELAY:
            return self._process_byr_delay(event)
        elif event.type == SELLER_DELAY:
            return self._process_slr_delay(event)
        else:
            raise NotImplementedError()

    def _record(self, event, byr_hist=None, censored=False):
        raise NotImplementedError()

    def _process_byr_offer(self, event):
        return self._process_offer(event)

    def _process_offer(self, event):
        # check whether the lstg expired, censoring this offer
        if self._lstg_expiration(event):
            return True
        # otherwise check whether this offer corresponds to an expiration rej
        if event.thread_expired():
            if event.turn % 2 == 0:
                self._process_slr_expire(event)
            else:
                self._process_byr_expire(event)
            return False
        # otherwise updates thread features
        self._prepare_offer(event)
        byr_turn = event.turn % 2 == 1
        # generate the offer outcomes
        if byr_turn:
            offer = event.buyer_offer()
        else:
            offer = event.seller_offer()
        self._record(event, censored=False)
        # check whether the offer is an acceptance
        if event.is_sale():
            self._process_sale(offer)
            return True
        # otherwise check whether the offer is a rejection
        elif event.is_rej():
            if byr_turn:
                self._process_byr_rej(offer)
                return False
            else:
                self._process_slr_rej(event, offer)
                return False
        else:
            if byr_turn:
                auto = self._check_slr_autos(offer.price)
                if auto == ACC_IND:
                    self._process_slr_auto_acc(event)
                    return True
                elif auto == REJ_IND:
                    self._process_slr_auto_rej(event, offer)
                    return False
            self.time_feats.update_features(offer=offer)
            self._init_delay(event)
            return False

    def _process_sale(self, offer):
        if offer.player == BYR_PREFIX:
            start_norm = offer.price
        else:
            start_norm = 1 - offer.price
        sale_price = start_norm * self.lookup[START_PRICE]
        self.outcome = self.Outcome(True, sale_price, offer.time)
        self.empty_queue()

    def _process_slr_offer(self, event):
        return self._process_offer(event)

    def _process_byr_delay(self, event):
        return self._process_delay(event)

    def _process_slr_delay(self, event):
        return self._process_delay(event)

    def _process_first_offer(self, event):
        """
        Processes the buyer's first offer in a thread
        :return:
        """
        # expiration
        sources = ThreadSources(x_lstg=self.x_lstg)
        months_since_lstg = time_delta(self.lookup[START_TIME], event.priority, unit=MONTH)
        time_feats = self.time_feats.get_feats(time=event.priority, thread_id=event.thread_id)
        sources.prepare_hist(time_feats=time_feats, clock_feats=get_clock_feats(event.priority),
                             months_since_lstg=months_since_lstg)
        hist = self.arrival.hist(sources())
        if self.verbose:
            print('Thread {} initiated | Buyer hist: {}'.format(event.thread_id, hist))
        event.init_thread(sources=sources, hist=hist)
        self._record(event, byr_hist=hist)
        return self._process_byr_offer(event)

    def _prepare_offer(self, event):
        # if offer not expired and thread still active, prepare this turn's inputs
        if event.turn != 1:
            time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                                   time=event.priority)
            clock_feats = get_clock_feats(event.priority)
            event.init_offer(time_feats=time_feats, clock_feats=clock_feats)
        return False

    def _process_arrival(self, event):
        """
        Updates queue with results of an Arrival Event

        :param event: Event corresponding to current event
        :return: boolean indicating whether the lstg has ended
        """
        if self._lstg_expiration(event):
            return True

        # update sources with clock feats
        clock_feats = get_clock_feats(event.priority)
        event.update_arrival(thread_count=self.thread_counter, clock_feats=clock_feats)

        # call model to sample inter arrival time and update arrival check priority
        seconds = event.inter_arrival()
        event.priority = min(event.priority + seconds, self.end_time)

        # if a buyer arrives, create a thread at the arrival time
        if event.priority < self.end_time:
            self.thread_counter += 1
            self.queue.push(self.make_thread(event.priority))

        self.queue.push(event)

        # increment thread counter
        self.thread_counter += 1

        return False

    def _process_byr_expire(self, event):
        event.byr_expire()
        self._record(event, censored=False)
        offer_params = {
            'thread_id': event.thread_id,
            'time': event.priority,
            'player': BYR_PREFIX
        }
        self.time_feats.update_features(offer=Offer(params=offer_params, rej=True))

    def _process_slr_expire(self, event):
        # update sources with new clock and features
        clock_feats = get_clock_feats(event.priority)
        time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                               time=event.priority)
        event.init_offer(time_feats=time_feats, clock_feats=clock_feats)
        offer = event.slr_expire_rej()
        self._record(event, censored=False)
        self.time_feats.update_features(offer=offer)
        self._init_delay(event)
        return False

    def _process_delay(self, event):
        # no need to check expiration since this must occur at the same time as the previous offer
        seconds = event.delay()
        event.prepare_offer(seconds)
        self.queue.push(event)

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
            self.outcome = self.Outcome(False, 0, MONTH)
            self.queue.push(event)
            self.empty_queue()
            if self.verbose:
                print('Lstg expired')
            return True
        else:
            return False

    def empty_queue(self):
        while not self.queue.empty:
            event = self.queue.pop()
            if not isinstance(event, Arrival) and event.type != FIRST_OFFER:
                event.priority = min(event.priority, self.end_time)
                self._record(event=event, censored=True)

    def make_thread(self, priority):
        raise NotImplementedError()

    def _check_slr_autos(self, norm):
        """ """
        if norm < self.lookup[ACC_PRICE] / self.lookup[START_PRICE]:
            if norm < self.lookup[DEC_PRICE] / self.lookup[START_PRICE]:
                return REJ_IND
            else:
                return OFF_IND
        else:
            return ACC_IND

    def _process_byr_rej(self, offer):
        self.time_feats.update_features(offer=offer)

    def _process_slr_rej(self, event, offer):
        self.time_feats.update_features(offer=offer)
        self._init_delay(event)

    def _process_slr_auto_rej(self, event, offer):
        self.time_feats.update_features(offer=offer)
        event.change_turn()
        offer = event.slr_auto_rej()
        self._record(event)
        self.time_feats.update_features(offer=offer)
        self._init_delay(event)

    def _process_slr_auto_acc(self, event):
        offer = event.slr_auto_acc()
        self._record(event, censored=False)
        self._process_sale(offer)

    def _init_delay(self, event):
        event.change_turn()
        event.init_delay(self.lookup[START_TIME])
        self.queue.push(event)

