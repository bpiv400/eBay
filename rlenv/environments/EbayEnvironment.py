import numpy as np
from collections import namedtuple
from constants import BYR_PREFIX, MONTH, ARRIVAL_PREFIX
from featnames import START_TIME, ACC_PRICE, DEC_PRICE, START_PRICE
from rlenv.Heap import Heap
from rlenv.time.TimeFeatures import TimeFeatures
from rlenv.time.Offer import Offer
from rlenv.events.Event import Event
from rlenv.events.Arrival import Arrival
from rlenv.sources import ArrivalSources
from rlenv.sources import ThreadSources
from rlenv.events.Thread import Thread
from rlenv.env_consts import (INTERACT, SALE, PRICE, DUR, ACC_IND,
                              REJ_IND, OFF_IND, ARRIVAL, FIRST_OFFER,
                              OFFER_EVENT, DELAY_EVENT, ARRIVAL_MODEL,
                              BYR_HIST_MODEL)
from rlenv.env_utils import get_clock_feats, get_con_outcomes, need_msg
from utils import get_months_since_lstg


class EbayEnvironment:
    Outcome = namedtuple('outcome', [SALE, PRICE, DUR])

    def __init__(self, params=None):
        # interfaces
        self.arrival = params['arrival']
        self.buyer = params['buyer']
        self.seller = params['seller']
        self.verbose = params['verbose']

        # features
        self.x_lstg = None
        self.lookup = None

        self.time_feats = TimeFeatures()
        # queue
        self.queue = Heap(entry_type=Event)

        # end time
        self.end_time = None
        self.thread_counter = 1
        self.outcome = None

        # interval data
        self.composer = params['composer']
        self.intervals = self.composer.intervals

    def reset(self):
        self.queue.reset()
        self.time_feats.reset()
        self.outcome = None
        self.thread_counter = 1
        sources = ArrivalSources(x_lstg=self.x_lstg)
        event = Arrival(priority=self.lookup[START_TIME], sources=sources)
        self.queue.push(event)

    def run(self):
        while True:
            event = self.queue.pop()
            if self.is_agent_turn(event):
                return event, False
            else:
                lstg_complete = self.process_event(event)
                if lstg_complete:
                    return event, True

    def process_event(self, event):
        if INTERACT and event.type != ARRIVAL:
            input('Press Enter to continue...')
        if event.type == ARRIVAL:
            return self.process_arrival(event)
        elif event.type == FIRST_OFFER:
            return self.process_first_offer(event)
        elif event.type == OFFER_EVENT:
            return self.process_offer(event)
        elif event.type == DELAY_EVENT:
            return self.process_delay(event)
        else:
            raise NotImplementedError()

    def record(self, event, byr_hist=None, censored=False):
        raise NotImplementedError()

    def process_offer(self, event):
        # check whether the lstg expired, censoring this offer
        if self.is_lstg_expired(event):
            return self.process_lstg_expiration(event)
        # otherwise check whether this offer corresponds to an expiration rej
        slr_offer = event.turn % 2 == 0
        if event.thread_expired():
            if slr_offer:
                self.process_slr_expire(event)
            else:
                self.process_byr_expire(event)
            return False
        # otherwise updates thread features
        self.prepare_offer(event)
        # generate concession and msg if necessary
        offer = self.get_offer_outcomes(event, slr=slr_offer)
        return self.process_post_offer(event, offer)

    def process_post_offer(self, event, offer):
        slr_offer = event.turn % 2 == 0
        self.record(event, censored=False)
        # check whether the offer is an acceptance
        if event.is_sale():
            self._process_sale(offer)
            return True
        # otherwise check whether the offer is a rejection
        elif event.is_rej():
            if slr_offer:
                self._process_slr_rej(event, offer)
                return False
            else:
                self._process_byr_rej(offer)
                return False
        else:
            if not slr_offer:
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

    def process_first_offer(self, event):
        """
        Processes the buyer's first offer in a thread
        :return:
        """
        # expiration
        sources = ThreadSources(x_lstg=self.x_lstg)
        months_since_lstg = get_months_since_lstg(lstg_start=self.lookup[START_TIME], start=event.priority)
        time_feats = self.time_feats.get_feats(time=event.priority, thread_id=event.thread_id)
        sources.prepare_hist(time_feats=time_feats, clock_feats=get_clock_feats(event.priority),
                             months_since_lstg=months_since_lstg)
        input_dict = self.composer.build_input_dict(BYR_HIST_MODEL, sources=sources(), turn=None)
        hist = self.get_hist(input_dict=input_dict, time=event.priority, thread_id=event.thread_id)
        if self.verbose:
            print('Thread {} initiated | Buyer hist: {}'.format(event.thread_id, hist))
        event.init_thread(sources=sources, hist=hist)
        self.record(event, byr_hist=hist)
        return self.process_offer(event)

    def prepare_offer(self, event):
        # if offer not expired and thread still active, prepare this turn's inputs
        if event.turn != 1:
            time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                                   time=event.priority)
            clock_feats = get_clock_feats(event.priority)
            event.init_offer(time_feats=time_feats, clock_feats=clock_feats)
        return False

    def process_arrival(self, event):
        """
        Updates queue with results of an Arrival Event

        :param event: Event corresponding to current event
        :return: boolean indicating whether the lstg has ended
        """
        if self.is_lstg_expired(event):
            return self.process_lstg_expiration(event)

        # update sources with clock feats
        clock_feats = get_clock_feats(event.priority)
        event.update_arrival(thread_count=self.thread_counter, clock_feats=clock_feats)

        # call model to sample inter arrival time and update arrival check priority
        input_dict = self.composer.build_input_dict(ARRIVAL_MODEL, sources=event.sources(), turn=None)
        seconds = self.get_inter_arrival(time=event.priority, input_dict=input_dict)
        event.priority = min(event.priority + seconds, self.end_time)

        # if a buyer arrives, create a thread at the arrival time
        if event.priority < self.end_time:
            self.thread_counter += 1
            self.queue.push(self.make_thread(event.priority))
        self.queue.push(event)
        return False

    def process_byr_expire(self, event):
        event.byr_expire()
        self.record(event, censored=False)
        offer_params = {
            'thread_id': event.thread_id,
            'time': event.priority,
            'player': BYR_PREFIX
        }
        self.time_feats.update_features(offer=Offer(params=offer_params, rej=True))

    def process_slr_expire(self, event):
        # update sources with new clock and features
        clock_feats = get_clock_feats(event.priority)
        time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                               time=event.priority)
        event.init_offer(time_feats=time_feats, clock_feats=clock_feats)
        offer = event.slr_expire_rej()
        self.record(event, censored=False)
        self.time_feats.update_features(offer=offer)
        self._init_delay(event)
        return False

    def process_delay(self, event):
        # no need to check expiration since this must occur at the same time as the previous offer
        if event.turn % 2 == 0:
            index = self.seller.delay(event.sources(), turn=event.turn)
        else:
            index = self.buyer.delay(event.sources(), turn=event.turn)
        event.prepare_offer(index)
        self.queue.push(event)

    def is_agent_turn(self, event):
        raise NotImplementedError()

    def is_lstg_expired(self, event):
        return event.priority >= self.end_time

    def process_lstg_expiration(self, event):
        """
        Checks whether the lstg has expired by the time of the event
        If so, record the reward as negative insertion fees
        :param event: rlenv.Event subclass
        :return: boolean
        """
        self.outcome = self.Outcome(False, 0, MONTH)
        self.queue.push(event)
        self.empty_queue()
        if self.verbose:
            print('Lstg expired')
        return True

    def empty_queue(self):
        while not self.queue.empty:
            event = self.queue.pop()
            if not isinstance(event, Arrival) and event.type != FIRST_OFFER:
                event.priority = min(event.priority, self.end_time)
                self.record(event=event, censored=True)

    def make_thread(self, priority):
        return Thread(priority=priority, thread_id=self.thread_counter,
                      intervals=self.intervals)

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
        self.record(event)
        self.time_feats.update_features(offer=offer)
        self._init_delay(event)

    def _process_slr_auto_acc(self, event):
        offer = event.slr_auto_acc()
        self.record(event, censored=False)
        self._process_sale(offer)

    def _init_delay(self, event):
        event.change_turn()
        event.init_delay(self.lookup[START_TIME])
        self.queue.push(event)

    def get_con(self, input_dict=None, time=None, thread_id=None, turn=None):
        if turn % 2 == 0:
            con = self.seller.con(input_dict=input_dict, turn=turn)
        else:
            con = self.buyer.con(input_dict=input_dict, turn=turn)
        return con

    def get_msg(self, input_dict=None, time=None, turn=None, thread_id=None):
        if turn % 2 == 0:
            msg = self.seller.msg(input_dict=input_dict)
        else:
            msg = self.buyer.msg(input_dict=input_dict)
        return msg

    def get_inter_arrival(self, input_dict=None, time=None):
        intervals = self.arrival.inter_arrival(input_dict)
        width = self.intervals[ARRIVAL_PREFIX]
        return int((intervals + np.random.uniform()) * width)

    def get_hist(self, input_dict=None, time=None, thread_id=None):
        return self.arrival.hist(input_dict=input_dict)

    def get_offer_outcomes(self, event, slr=False):
        # sample concession
        model_name = self.seller.con_model_name if slr else self.buyer.con_model_name
        input_dict = self.composer.build_input_dict(model_name, sources=event.sources(),
                                                    turn=event.turn)
        con = self.get_con(input_dict=input_dict, time=event.priority, turn=event.turn,
                           thread_id=event.thread_id)
        con_outcomes = get_con_outcomes(con=con, sources=event.sources(), turn=event.turn)
        # update features
        offer = event.update_con_outcomes(con_outcomes=con_outcomes)
        # sample msg if necessary
        if need_msg(con):
            model_name = self.seller.msg_model_name if slr else self.buyer.msg_model_name
            input_dict = self.composer.build_input_dict(model_name, sources=event.sources(),
                                                        turn=event.turn)
            msg = self.get_msg(input_dict=input_dict, time=event.priority, turn=event.turn,
                               thread_id=event.thread_id)
            event.update_msg(msg=msg)
        return offer
