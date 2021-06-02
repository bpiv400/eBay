"""
class for encapsulating data and methods related to the first buyer offer
"""
import numpy as np
from rlenv.events.Event import Event
from rlenv.const import DELAY_EVENT, OFFER_EVENT
from featnames import SLR, BYR
from rlenv.util import last_norm, prev_norm, get_delay_outcomes, get_clock_feats
from rlenv.time.Offer import Offer
from constants import MAX_DELAY_TURN
from utils import get_remaining


class Thread(Event):
    """
    Attributes:
    """
    def __init__(self, priority=None, thread_id=None, hist=None, sources=None):
        super().__init__(event_type=OFFER_EVENT, priority=priority, thread_id=thread_id)
        self.turn = 1
        self.hist = hist
        self.sources = sources

    def thread_expired(self):
        if self.turn == 1:
            return False
        else:
            return self.sources.is_expire(self.turn)

    def update_con_outcomes(self, con_outcomes=None):
        assert self.thread_id is not None
        norm = self.sources.update_con_outcomes(con_outcomes=con_outcomes,
                                                turn=self.turn)
        offer_params = {
            'price': norm,
            'player': SLR if self.turn % 2 == 0 else BYR,
            'time': self.priority,
            'thread_id': self.thread_id
        }
        return Offer(params=offer_params, rej=self.is_rej())

    def update_msg(self, msg=None):
        self.sources.update_msg(msg=msg, turn=self.turn)

    def init_delay(self, lstg_start):
        self.type = DELAY_EVENT
        self._set_remaining(lstg_start)

    def change_turn(self):
        self.turn += 1

    def _set_remaining(self, lstg_start):
        remaining = get_remaining(lstg_start, self.priority)
        self.sources.init_remaining(remaining=remaining)

    def update_delay(self, seconds=None):
        if seconds > MAX_DELAY_TURN:
            raise RuntimeError('Excessive delay: {} seconds'.format(seconds))
        # update outcomes
        delay_outcomes = get_delay_outcomes(seconds=seconds, turn=self.turn)
        self.sources.update_delay(delay_outcomes=delay_outcomes, turn=self.turn)
        # change event type
        self.type = OFFER_EVENT
        self.priority += seconds

    def is_sale(self):
        return self.sources.is_sale(self.turn)

    def is_rej(self):
        return self.sources.is_rej(self.turn)

    def slr_expire_rej(self):
        return self.slr_rej()

    def slr_auto_rej(self, time_feats=None):
        self.init_offer(time_feats=time_feats)
        delay_outcomes = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float)
        self.sources.update_delay(delay_outcomes=delay_outcomes, turn=self.turn)
        return self.slr_rej()

    def slr_rej(self):
        last = last_norm(sources=self.sources(), turn=self.turn)
        con_outcomes = np.array([0.0, 1.0, last, 0.0], dtype=np.float32)
        norm = self.sources.update_con_outcomes(con_outcomes=con_outcomes,
                                                turn=self.turn)
        offer_params = {
            'price': norm,
            'player': SLR,
            'time': self.priority,
            'thread_id': self.thread_id
        }
        return Offer(params=offer_params, rej=True)

    def slr_auto_acc(self):
        self.change_turn()
        delay_outcomes = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float)
        self.sources.update_delay(delay_outcomes=delay_outcomes, turn=self.turn)
        prev = prev_norm(sources=self.sources(), turn=self.turn)
        outcomes = np.array([1.0, 0.0, 1.0 - prev, 0.0], dtype=np.float32)
        norm = self.sources.update_con_outcomes(con_outcomes=outcomes, turn=self.turn)
        offer_params = {
            'price': norm,
            'player': SLR,
            'time': self.priority,
            'thread_id': self.thread_id
        }
        return Offer(params=offer_params, rej=False)

    def init_offer(self, time_feats=None):
        """
        Updates the clock and time features to their current
        values before sampling a concession / executing an offer
        :param time_feats: np.array containing time feats
        """
        clock_feats = get_clock_feats(self.priority)
        self.sources.init_offer(time_feats=time_feats,
                                clock_feats=clock_feats,
                                turn=self.turn)

    def summary(self):
        return self.sources.summary(self.turn)

    def delay_outcomes(self):
        return self.sources.get_delay_outcomes(self.turn)

    def slr_outcomes(self):
        return self.sources.get_slr_outcomes(self.turn)

    def byr_expire(self):
        self.sources.byr_expire(turn=self.turn)
