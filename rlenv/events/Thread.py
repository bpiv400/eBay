"""
class for encapsulating data and methods related to the first buyer offer
"""
import numpy as np
from rlenv.events.Event import Event
from rlenv.env_consts import (FIRST_OFFER, BUYER_OFFER, SELLER_OFFER,
                              INTERVAL_COUNT, INTERVAL, BUYER_DELAY, SELLER_DELAY)
from constants import (DAY, SLR_PREFIX, BYR_PREFIX, MAX_DELAY, MAX_DAYS)
from rlenv.env_utils import slr_rej, slr_auto_acc
from rlenv.time.Offer import Offer

class Thread(Event):
    """
    Attributes:
    """
    def __init__(self, priority=None, thread_id=None, interval_attrs=None):
        super(Thread, self).__init__(event_type=FIRST_OFFER,
                                     priority=priority)
        # participants
        self.buyer = None
        self.seller = None
        # sources object
        self.sources = None  # initialized later in init_thread
        self.turn = 1
        self.thread_id = thread_id
        # delay
        self.interval_attrs = interval_attrs
        self.max_delay = None # max length of delay
        self.spi = None  # seconds per interval
        self.remaining = None  # initial periods remaining

    def init_thread(self, sources=None, hist=None):
        self.sources = sources
        self.sources.init_thread(hist=hist)

    def thread_expired(self):
        if self.turn == 1:
            return False
        else:
            return self.sources.is_expire(self.turn)

    def offer(self, interface=None, player_type=None):
        outcomes = interface.make_offer(sources=self.sources(), turn=self.turn)
        norm = self.sources.update_offer(outcomes=outcomes, turn=self.turn)
        offer_params = {
            'price': norm,
            'player': player_type,
            'time': self.priority,
            'thread_id': self.thread_id
        }
        return Offer(params=offer_params, rej=self.is_rej())

    def buyer_offer(self, *args):
        return self.offer(interface=self.buyer, player_type=BYR_PREFIX)

    def seller_offer(self, *args):
        return self.offer(interface=self.seller, player_type=SLR_PREFIX)

    def init_delay(self, lstg_start):
        if self.turn % 2 == 0:
            self.type = SELLER_DELAY
        else:
            self.type = BUYER_DELAY
        self._init_delay_params(lstg_start)

    def change_turn(self):
        self.turn += 1
        self.sources.change_turn(self.turn)

    def _init_delay_params(self, lstg_start):
        # update object to contain relevant delay attributes
        if self.turn % 2 == 0:
            delay_type = SLR_PREFIX
        elif self.turn == 7:
            delay_type = '{}_{}'.format(BYR_PREFIX, 7)
        else:
            delay_type = BYR_PREFIX
        self.max_delay = MAX_DELAY[delay_type]
        self.spi = self.interval_attrs[INTERVAL][delay_type]
        # create remaining
        self._init_remaining(lstg_start, self.max_delay)
        self.sources.init_remaining(remaining=self.remaining)

    def _init_remaining(self, lstg_start, max_delay):
        self.remaining = (MAX_DAYS + lstg_start) - self.priority
        self.remaining = self.remaining / max_delay
        self.remaining = min(self.remaining, 1)

    def delay(self):
        # add new features
        # generate delay
        if self.turn % 2 == 0:
            index = self.seller.delay(self.sources())
        else:
            index = self.buyer.delay(self.sources())
        seconds = int((index + np.random.uniform()) * self.spi)
        return min(seconds, self.max_delay)

    def prepare_offer(self, seconds):
        # update outcomes
        delay = seconds / self.max_delay
        days = seconds / DAY
        self.sources.prepare_offer(days=days, delay=delay, turn=self.turn)
        # change event type
        if self.turn % 2 == 0:
            self.type = SELLER_OFFER
        else:
            self.type = BUYER_OFFER
        self.priority += seconds

    def is_sale(self):
        return self.sources.is_sale(self.turn)

    def is_rej(self):
        return self.sources.is_rej(self.turn)

    def slr_rej(self, expire=False):
        outcomes = slr_rej(self.sources(), self.turn, expire=expire)
        norm = self.sources.update_offer(outcomes=outcomes, turn=self.turn)
        offer_params = {
            'price': norm,
            'player': SLR_PREFIX,
            'time': self.priority,
            'thread_id': self.thread_id
        }
        return Offer(params=offer_params, rej=True)

    def slr_auto_acc(self):
        self.change_turn()
        outcomes = slr_auto_acc(self.sources(), self.turn)
        norm = self.sources.update_offer(outcomes=outcomes, turn=self.turn)
        offer_params = {
            'price': norm,
            'type': SLR_PREFIX,
            'time': self.priority,
            'thread_id': self.thread_id
        }
        return Offer(params=offer_params, rej=False)

    def init_offer(self, time_feats=None, clock_feats=None):
        self.sources.init_offer(time_feats=time_feats, clock_feats=clock_feats, turn=self.turn)

    def summary(self):
        con, norm, msg, split = self.sources.summary(self.turn)
        if self.turn % 2 == 0:
            norm = 100 - norm
        return con, norm, msg, split

    def delay_outcomes(self):
        return self.sources.get_delay_outcomes(self.turn)

    def slr_outcomes(self):
        return self.sources.get_slr_outcomes(self.turn)

    def byr_expire(self):
        self.sources.byr_expire(turn=self.turn)

    def get_obs(self):
        pass
        # raise NotImplementedError()
        # srcs = self.sources()
        # maybe move to composer or sources
        # return OBS_SPACE(LSTG_MAP=srcs[LSTG_MAP].values,
        #                 THREAD_MAP=srcs[THREAD_MAP].values,
        #                 TURN_IND_MAP=srcs[TURN_IND_MAP].values.astype(np.int),
        #                 X_TIME_MAP=srcs[X_TIME_MAP][INT_REMAINING])

