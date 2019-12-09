import numpy as np
from rlenv.time import time_triggers
from rlenv.env_consts import (MONTH, START_DAY, ACC_PRICE, META, VERBOSE,
                              DEC_PRICE, START_PRICE, ANCHOR_STORE_INSERT)
from rlenv.env_utils import get_clock_feats, get_value_fee
from constants import BYR_PREFIX
from rlenv.simulators import SimulatedSeller, SimulatedBuyer
from rlenv.events.RewardThread import RewardThread
from rlenv.environments.EbayEnvironment import EbayEnvironment

ACC_IND = 0
REJ_IND = 1
OFF_IND = 2


class RewardEnvironment(EbayEnvironment):
    def __init__(self, **kwargs):
        super(RewardEnvironment, self).__init__(kwargs['arrival'])
        # environment interface
        self.buyer = kwargs['buyer']
        self.seller = kwargs['seller']

        # features
        self.x_lstg = kwargs['x_lstg']
        self.lookup = kwargs['lookup']

        # end time
        self.end_time = self.lookup[START_DAY] + MONTH
        self.outcome = None
        self.thread_counter = 0

        # recorder
        self.recorder = kwargs['recorder']

    def reset(self):
        super(RewardEnvironment, self).reset()
        self.recorder.reset_sim()
        if VERBOSE:
            print('Initializing Simulation {}'.format(self.recorder.sim))

    def run(self):
        """
        Runs a simulation of a single lstg until sale or expiration

        :return: a 3-tuple of (bool, float, int) giving whether the listing sells,
        the amount it sells for if it sells, and the amount of time it took to sell
        """
        super(RewardEnvironment, self).run()
        return self.outcome

    def _check_complete(self, event):
        return False

    def _process_first_offer(self, event):
        hist = super(RewardEnvironment, self)._process_first_offer(event)
        self.recorder.start_thread(thread_id=event.thread_id, byr_hist=hist)
        return self._process_byr_offer(event)

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

    def make_thread(self, priority):
        return RewardThread(priority=priority, thread_id=self.thread_counter,
                      buyer=SimulatedBuyer(model=self.buyer),
                      seller=SimulatedSeller(model=self.seller))



