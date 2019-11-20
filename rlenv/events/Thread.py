"""
class for encapsulating data and methods related to the first buyer offer
"""
import torch
from rlenv.events.Event import Event
from rlenv.events import event_types
from constants import (SLR_PREFIX, BYR_PREFIX, INTERVAL_COUNTS,
                       INTERVAL, MAX_DELAY, MAX_DAYS)
from rlenv.env_utils import time_delta
from rlenv.env_consts import MONTH, DAY


class Thread(Event):
    """
    Attributes:
        hidden: representation of buyer and seller hidden states (dictionary containing 'byr', 'slr')

        ids: identifier dictionary
    """
    def __init__(self, priority, thread_id=None, buyer=None, seller=None):
        super(Thread, self).__init__(event_type=event_types.FIRST_OFFER,
                                     priority=priority)
        # participants
        self.buyer = buyer
        self.seller = seller
        self.thread_start = priority
        # sources object
        self.sources = None  # initialized later in init_thread
        self.turn = 1
        self.thread_id = thread_id
        # delay
        self.interval = 0
        self.max_interval = None  # max number of intervals
        self.max_delay = None # max length of delay
        self.spi = None  # seconds per interval
        self.init_remaining = None  # initial periods remaining

    def init_thread(self, sources=None, hist=None):
        self.sources = sources
        self.sources.init_thread(hist=hist)

    def buyer_offer(self):
        outcomes = self.buyer.make_offer(sources=self.sources(), turn=self.turn)
        norm = self.sources.update_offer(outcomes)
        return {
            'price': norm,
            'type': BYR_PREFIX,
            'time': self.priority
        }

    def seller_offer(self):
        outcomes = self.seller.make_offer(sources=self.sources(), turn=self.turn)
        norm = self.sources.update_offer(outcomes)
        return {
            'price': norm,
            'type': SLR_PREFIX,
            'time': self.priority
        }

    def change_turn(self):
        self.turn += 1
        self.sources.change_turn(self.turn)

    def init_delay(self, lstg_start):
        # update object to contain relevant delay attributes
        if self.turn % 2 == 0 or self.turn == 7:
            self.max_interval = INTERVAL_COUNTS[SLR_PREFIX]
            self.max_delay = MAX_DELAY[SLR_PREFIX]
        else:
            self.max_interval = INTERVAL_COUNTS[BYR_PREFIX]
            self.max_delay = MAX_DELAY[BYR_PREFIX]
        # create init remaining
        self._init_remaining(lstg_start, self.max_delay)
        self.interval = 0
        self.type = event_types.DELAY
        # add delay features
        months_since_lstg = time_delta(lstg_start, self.priority, unit=MONTH)
        days_since_thread = time_delta(self.thread_start, self.priority, unit=DAY)
        self.sources.init_delay(months_since_lstg=months_since_lstg,
                                days_since_thread=days_since_thread)
        # update hidden state
        if self.turn % 2 == 0:
            self.spi = INTERVAL[SLR_PREFIX]
            self.seller.init_delay(self.sources())
        else:
            self.spi = INTERVAL[BYR_PREFIX]
            self.buyer.init_delay(self.sources())

    def _init_remaining(self, lstg_start, max_delay):
        self.init_remaining = (MAX_DAYS + lstg_start) - self.priority
        self.init_remaining = self.init_remaining / max_delay
        self.init_remaining = min(self.init_remaining, 1)

    def delay(self, time_feats=None, clock_feats=None):
        # add new features
        duration = self.interval / self.max_interval
        remaining = self.init_remaining - duration
        self.sources.update_delay(time_feats=time_feats, clock_feats=clock_feats,
                                  duration=torch.tensor(duration).float(),
                                  remaining=torch.tensor(remaining).float())
        # generate delay
        if self.turn % 2 == 0:
            make_offer = self.seller.delay(self.sources())
        else:
            make_offer = self.buyer.delay(self.sources())
        if make_offer == 0:
            self.interval += 1
        return make_offer

    def prepare_offer(self, add_delay):
        # update outcomes
        delay_dur = self.spi * self.interval + add_delay
        delay = delay_dur / self.max_delay
        days = delay_dur / DAY
        self.sources.prepare_offer(days=days, delay=delay)
        # reset delay markers
        self.interval = 0
        self.max_interval = None
        self.max_delay = None
        self.spi = None
        self.init_remaining = None
        # change event type
        self.type = event_types.OFFER
        self.priority += add_delay

    def thread_expired(self):
        return self.interval >= self.max_interval

    def is_sale(self):
        return self.sources.is_sale()

    def is_rej(self):
        return self.sources.is_rej()

    def rej(self, expire=False):
        outcomes = self.seller.rej(self.sources(), self.turn, expire=expire)
        norm = self.sources.update_offer(outcomes)
        return {
            'price': norm,
            'type': SLR_PREFIX,
            'time': self.priority
        }

    def init_offer(self, time_feats=None, clock_feats=None):
        self.sources.init_offer(time_feats=time_feats, clock_feats=clock_feats)

    def summary(self):
        con, norm, msg, split = self.sources.summary()
        if self.turn % 2 == 0:
            norm = 100 - norm
        return con, norm, msg, split

    def delay_outcomes(self):
        return self.sources.get_delay_outcomes()

    def slr_outcomes(self):
        return self.sources.get_slr_outcomes()


    def byr_expire(self):
        self.priority += self.spi
        days = self.max_delay / DAY
        self.sources.byr_expire(days=days)
