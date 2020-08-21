"""
class for encapsulating data and methods related to the first buyer offer
"""
import numpy as np
from rlenv.Sources import ThreadSources
from rlenv.events.Event import Event
from rlenv.const import FIRST_OFFER, DELAY_EVENT, OFFER_EVENT
from featnames import SLR, BYR
from rlenv.util import (slr_rej_outcomes, slr_auto_acc_outcomes,
                        get_delay_outcomes, get_clock_feats,
                        get_con_outcomes)
from rlenv.time.Offer import Offer
from utils import get_remaining


class Thread(Event):
    """
    Attributes:
    """
    def __init__(self, priority=None, thread_id=None):
        super().__init__(event_type=FIRST_OFFER, priority=priority)
        self.sources = None  # to be initialized later
        self.thread_id = thread_id

    def init_thread(self, sources=None, hist=None):
        self.sources = sources  # type: ThreadSources
        self.sources.init_thread(hist=hist)

    def thread_expired(self):
        if self.turn == 1:
            return False
        else:
            return self.sources.is_expire(self.turn)

    def update_con_outcomes(self, con_outcomes=None):
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
        con_outcomes = slr_rej_outcomes(self.sources(), self.turn)
        norm = self.sources.update_con_outcomes(con_outcomes=con_outcomes, turn=self.turn)
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
        outcomes = slr_auto_acc_outcomes(self.sources(), self.turn)
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


class RlThread(Thread):
    def __init__(self, priority=None, sources=None, con=None,
                 thread_id=None, rl_buyer=False):
        super().__init__(priority=priority, thread_id=thread_id)
        self.rl_buyer = rl_buyer
        self.sources = sources
        self.stored_concession = con
        if self.rl_buyer:
            self.type = OFFER_EVENT

    def set_thread_id(self, thread_id=None):
        self.thread_id = thread_id

    def prep_rl_offer(self, con=None, priority=None):
        """
        Updates sources with the delay outcomes associated with an upcoming
        RL offer. Called at the moment the agent chooses the delay/con
        :param con: concession associated with incoming offer
        :param priority: time that the offer will take place
        """
        assert self.turn != 1
        self.type = OFFER_EVENT
        self.stored_concession = con
        delay_outcomes = get_delay_outcomes(seconds=priority-self.priority,
                                            turn=self.turn)
        self.sources.update_delay(delay_outcomes=delay_outcomes,
                                  turn=self.turn)
        self.priority = priority

    def init_rl_offer(self, time_feats=None, months_since_lstg=None):
        """
        Updates sources with the time and clock features of the moment
        the RL makes its offer. If this is the first turn, it also
        updates months_since_lstg to the value of the current time
        :param time_feats: raw current time features
        :param months_since_lstg: float giving months_since_lstg
        """
        clock_feats = get_clock_feats(self.priority)
        if self.turn == 1:
            self.sources.prepare_hist(time_feats=time_feats,
                                      months_since_lstg=months_since_lstg,
                                      clock_feats=clock_feats)
        else:
            self.sources.init_offer(time_feats=time_feats,
                                    clock_feats=clock_feats,
                                    turn=self.turn)

    def execute_offer(self):
        """
        After the prescribed delay has elapsed, executes the agent's selected
        concession by updating the relevant sources

        Errors if encountering slr expiration rejection or buyer rejection
        because these should have already been processed

        slr expiration rejection encoded as con > 1
        :return Offer representing the executed turn
        """
        # process slr rejection
        if self.stored_concession == 0:
            if self.turn % 2 == 0:
                return self.slr_rej()
            else:
                raise RuntimeError("Buyer rejections should have"
                                   " already executed")
        elif self.stored_concession > 1:
            raise RuntimeError("Slr expiration rejections should have"
                               "already been executed")
        # process ordinary concession or acceptance
        else:
            con_outcomes = get_con_outcomes(con=self.stored_concession,
                                            sources=self.sources(),
                                            turn=self.turn)
            self.stored_concession = None
            return self.update_con_outcomes(con_outcomes=con_outcomes)
