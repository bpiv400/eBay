import numpy as np
from rlenv.time.TimeFeatures import TimeFeatures
from rlenv.time import time_triggers
from rlenv.events.EventQueue import EventQueue
from rlenv.env_consts import (MONTH, START_DAY, ACC_PRICE, META, INTERACT, VERBOSE,
                              DEC_PRICE, START_PRICE, ANCHOR_STORE_INSERT)
from rlenv.env_utils import get_clock_feats, get_value_fee, time_delta
from rlenv.events.Arrival import Arrival
from rlenv.events import event_types
from rlenv.sources import ThreadSources, ArrivalSources
from rlenv.events.Thread import Thread
from constants import INTERVAL, BYR_PREFIX
from rlenv.simulators import SimulatedSeller, SimulatedBuyer


ACC_IND = 0
REJ_IND = 1
OFF_IND = 2


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
        self.end_time = self.lookup[START_DAY] + MONTH
        self.outcome = None
        self.thread_counter = 0

        # recorder
        self.recorder = kwargs['recorder']

    def reset(self):
        self.queue.reset()
        self.time_feats.reset()
        self.recorder.reset_sim()
        self.thread_counter = 0
        sources = ArrivalSources(x_lstg=self.x_lstg, composer=self.arrival.composer)
        self.arrival.init(sources=sources())
        self.queue.push(Arrival(self.lookup[START_DAY], sources))
        if VERBOSE:
            print('Initializing Simulation {}'.format(self.recorder.sim))

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
        if INTERACT and event.type != event_types.ARRIVAL:
            input('Press Enter to continue...')
        if event.type == event_types.ARRIVAL:
            return self._process_arrival(event)
        elif event.type == event_types.FIRST_OFFER:
            return self._process_first_offer(event)
        elif event.type == event_types.OFFER:
            return self._process_offer(event)
        elif event.type == event_types.DELAY:
            return self._process_delay(event)
        else:
            raise NotImplementedError()

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
                offer_event = Thread(priority, thread_id=self.thread_counter,
                                     buyer=SimulatedBuyer(model=self.buyer),
                                     seller=SimulatedSeller(model=self.seller))
                self.queue.push(offer_event)
        # Add arrival check
        event.priority = event.priority + INTERVAL['arrival']
        self.queue.push(event)
        return False

    def _process_first_offer(self, event):
        """
        Processes the buyer's first offer in a thread

        :param event:
        :return:
        """
        # expiration
        sources = ThreadSources(x_lstg=self.x_lstg, composer=self.arrival.composer)
        months_since_lstg = time_delta(self.lookup[START_DAY], event.priority, unit=MONTH)
        time_feats = self.time_feats.get_feats(time=event.priority,thread_id=event.thread_id)
        sources.prepare_hist(time_feats=time_feats, clock_feats=get_clock_feats(event.priority),
                             months_since_lstg=months_since_lstg)
        hist = self.arrival.hist(sources())
        if VERBOSE:
            print('Thread {} initiated | Buyer hist: {}'.format(event.thread_id, hist))
        event.init_thread(sources=sources, hist=hist)
        self.recorder.start_thread(thread_id=event.thread_id, byr_hist=hist)
        return self._process_offer(event)

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

    def _process_offer(self, event):
        # TODO: ADD event tracking logic throughout
        # check whether the lstg has expired and close the thread if so
        if self._lstg_expiration(event):
            return True
        if event.turn != 1:
            time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                                   time=event.priority)
            clock_feats = get_clock_feats(event.priority)
            event.init_offer(time_feats=time_feats, clock_feats=clock_feats)
        byr_turn = event.turn % 2 == 1
        # generate the offer outcomes
        if byr_turn:
            offer = event.buyer_offer()
        else:
            offer = event.seller_offer()
        self.recorder.add_offer(event)
        # check whether the offer is an acceptance
        if event.is_sale():
            self._process_sale(offer)
            return True
        # otherwise check whether the offer is a rejection
        elif event.is_rej():
            if byr_turn:
                self._process_byr_rej(event)
                return False
            else:
                self._process_slr_rej(event, offer)
                return False
        else:
            if byr_turn:
                auto = self._check_slr_autos(offer['price'])
                if auto == ACC_IND:
                    self._process_sale(offer)
                    return True
                elif auto == REJ_IND:
                    self._process_slr_auto_rej(event, offer)
                    return False
            self.time_feats.update_features(trigger_type=time_triggers.OFFER,
                                            thread_id=event.thread_id,
                                            offer=offer)
            self._init_delay(event)
            return False

    def _check_slr_autos(self, norm):
        """ """
        if norm < self.lookup[ACC_PRICE] / self.lookup[START_PRICE]:
            if norm < self.lookup[DEC_PRICE] / self.lookup[START_PRICE]:
                return REJ_IND
            else:
                return OFF_IND
        else:
            return ACC_IND

    def _process_sale(self, offer):
        if offer['type'] == BYR_PREFIX:
            start_norm = offer['price']
        else:
            start_norm = 1 - offer['price']
        sale_price = start_norm * self.lookup[START_PRICE]
        insertion_fees = ANCHOR_STORE_INSERT
        value_fee = get_value_fee(sale_price, self.lookup[META])
        net = sale_price - insertion_fees - value_fee
        dur = offer['time'] - self.lookup[START_DAY]
        self.outcome = (True, net, dur)

    def _process_byr_rej(self, event):
        self.time_feats.update_features(trigger_type=time_triggers.BYR_REJECTION,
                                        thread_id=event.thread_id)

    def _process_slr_rej(self, event, offer):
        self.time_feats.update_features(trigger_type=time_triggers.SLR_REJECTION,
                                        thread_id=event.thread_id, offer=offer)
        self._init_delay(event)

    def _process_slr_auto_rej(self, event, offer):
        self.time_feats.update_features(trigger_type=time_triggers.OFFER,
                                        thread_id=event.thread_id, offer=offer)
        event.change_turn()
        offer = event.slr_rej(expire=False)
        self.recorder.add_offer(event)
        self.time_feats.update_features(trigger_type=time_triggers.SLR_REJECTION,
                                        thread_id=event.thread_id, offer=offer)
        self._init_delay(event)

    def _init_delay(self, event):
        event.change_turn()
        event.init_delay(self.lookup[START_DAY])
        self.queue.push(event)

    def _process_delay(self, event):
        if self._lstg_expiration(event):
            return True
        elif event.thread_expired():
            if event.turn % 2 == 0:
                self._process_slr_expire(event)
            else:
                self._process_byr_expire(event)
            return False
        time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                               time=event.priority)
        clock_feats = get_clock_feats(event.priority)
        make_offer = event.delay(clock_feats=clock_feats, time_feats=time_feats)
        if VERBOSE and make_offer == 1:
            actor = 'Seller' if event.turn % 2 == 0 else 'Buyer'
            print('{} will make an offer in the upcoming interval'.format(actor))
        if make_offer == 1:
            delay_dur = np.random.randint(0, event.spi)
            event.prepare_offer(delay_dur)
        else:
            event.priority += event.spi
        self.queue.push(event)
        return False

    def _process_byr_expire(self, event):
        event.byr_expire()
        self.recorder.add_offer(event)
        self.time_feats.update_features(trigger_type=time_triggers.BYR_REJECTION,
                                        thread_id=event.thread_id)

    def _process_slr_expire(self, event):
        event.prepare_offer(0)
        # update sources with new time and clock features
        time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                               time=event.priority)
        clock_feats = get_clock_feats(event.priority)
        event.init_offer(time_feats=time_feats, clock_feats=clock_feats)
        offer = event.slr_rej(expire=True)
        self.recorder.add_offer(event)
        self.time_feats.update_features(trigger_type=time_triggers.SLR_REJECTION,
                                        thread_id=event.thread_id,
                                        offer=offer)
        self._init_delay(event)
        return False



