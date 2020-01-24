"""
class for encapsulating data and methods related to the first buyer offer
"""
import numpy as np
from rlenv.events.Event import Event
from rlenv.env_consts import (FIRST_OFFER, DELAY_EVENT, OFFER_EVENT)
from constants import (SLR_PREFIX, BYR_PREFIX, MAX_DELAY)
from rlenv.env_utils import slr_rej_outcomes, slr_auto_acc_outcomes, get_delay_outcomes
from rlenv.time.Offer import Offer
from utils import get_remaining


class Thread(Event):
    """
    Attributes:
    """
    def __init__(self, priority=None, thread_id=None, intervals=None):
        super(Thread, self).__init__(event_type=FIRST_OFFER,
                                     priority=priority)

        # sources object
        self.sources = None  # initialized later in init_thread
        self.turn = 1
        self.thread_id = thread_id

        # delay
        self.intervals = intervals
        self.max_delay = None # max length of delay
        self.spi = None  # seconds per interval

    def init_thread(self, sources=None, hist=None):
        self.sources = sources
        self.sources.init_thread(hist=hist)

    def thread_expired(self):
        if self.turn == 1:
            return False
        else:
            return self.sources.is_expire(self.turn)

    def update_offer(self, offer_outcomes=None):
        norm = self.sources.update_offer(offer_outcomes=offer_outcomes, turn=self.turn)
        offer_params = {
            'price': norm,
            'player': SLR_PREFIX if self.turn % 2 == 0 else BYR_PREFIX,
            'time': self.priority,
            'thread_id': self.thread_id
        }
        return Offer(params=offer_params, rej=self.is_rej())

    def init_delay(self, lstg_start):
        self.type = DELAY_EVENT
        self._init_delay_params(lstg_start)

    def change_turn(self):
        self.turn += 1

    def _init_delay_params(self, lstg_start):
        # update object to contain relevant delay attributes
        if self.turn % 2 == 0:
            delay_type = SLR_PREFIX
        elif self.turn == 7:
            delay_type = '{}_{}'.format(BYR_PREFIX, 7)
        else:
            delay_type = BYR_PREFIX
        self.max_delay = MAX_DELAY[delay_type]
        self.spi = self.intervals[delay_type]
        # create remaining
        remaining = get_remaining(
            lstg_start, self.priority, self.max_delay)
        self.sources.init_remaining(remaining=remaining)

    def prepare_offer(self, index):
        seconds = int((index + np.random.uniform()) * self.spi)
        seconds = min(seconds, self.max_delay)
        # update outcomes
        delay_outcomes = get_delay_outcomes(seconds=seconds, max_delay=self.max_delay,
                                            turn=self.turn)
        self.sources.prepare_offer(delay_outcomes=delay_outcomes, turn=self.turn)
        # change event type
        self.type = OFFER_EVENT
        self.priority += seconds

    def is_sale(self):
        return self.sources.is_sale(self.turn)

    def is_rej(self):
        return self.sources.is_rej(self.turn)

    def slr_expire_rej(self):
        return self.slr_rej()

    def slr_auto_rej(self):
        delay_outcomes = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float)
        self.sources.prepare_offer(delay_outcomes=delay_outcomes, turn=self.turn)
        return self.slr_rej()

    def slr_rej(self):
        offer_outcomes = slr_rej_outcomes(self.sources(), self.turn)
        norm = self.sources.update_offer(offer_outcomes=offer_outcomes, turn=self.turn)
        offer_params = {
            'price': norm,
            'player': SLR_PREFIX,
            'time': self.priority,
            'thread_id': self.thread_id
        }
        return Offer(params=offer_params, rej=True)

    def slr_auto_acc(self):
        self.change_turn()
        delay_outcomes = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float)
        self.sources.prepare_offer(delay_outcomes=delay_outcomes, turn=self.turn)
        outcomes = slr_auto_acc_outcomes(self.sources(), self.turn)
        norm = self.sources.update_offer(offer_outcomes=outcomes, turn=self.turn)
        offer_params = {
            'price': norm,
            'player': SLR_PREFIX,
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

