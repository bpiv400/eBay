"""
class for encapsulating data and methods related to the first buyer offer
"""
from events.Event import Event
from events import event_types
from constants import SLR_PREFIX, BYR_PREFIX, INTERVAL_COUNTS
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
                                     priority=priority, thread_id=thread_id)
        # participants
        self.buyer = buyer
        self.seller = seller
        self.thread_start = priority
        # sources object
        self.sources = None  # initialized later in init_thread
        self.interval = 0
        self.turn = 1

    def init_thread(self, sources=None, hist=None):
        self.sources = sources
        self.sources.init_thread(hist=hist)

    def buyer_offer(self):
        outcomes = self.buyer(sources=self.sources(), turn=self.turn)
        norm = self.sources.update_offer(outcomes)
        return {
            'price': norm,
            'type': BYR_PREFIX,
            'time': self.priority
        }

    def seller_offer(self):
        outcomes = self.seller(sources=self.sources(), turn=self.turn)
        norm = self.sources.update_offer(outcomes)
        return {
            'price': norm,
            'type': SLR_PREFIX,
            'time': self.priority
        }

    def change_turn(self):
        self.turn += 1
        self.sources.change_turn()

    def init_delay(self):
        self.interval = 0
        self.type = event_types.DELAY
        months_since_lstg = time_delta(self.sources.start_date, self.priority, unit=MONTH)
        days_since_thread = time_delta(self.thread_start, self.priority, unit=DAY)
        self.sources.init_delay(months_since_lstg=months_since_lstg,
                                days_since_thread=days_since_thread)

    def thread_expired(self):
        if self.turn % 2 == 0 or self.turn == 7:
            max_interval = INTERVAL_COUNTS[SLR_PREFIX]
        else:
            max_interval = INTERVAL_COUNTS[BYR_PREFIX]
        return self.interval >= max_interval

    def is_sale(self):
        return self.sources.is_sale()

    def is_rej(self):
        return self.sources.is_rej()

    def auto_rej(self):
        outcomes = self.seller.auto_rej(self.sources(), self.turn)
        norm = self.sources.update_offer(outcomes)
        return {
            'price': norm,
            'type': SLR_PREFIX,
            'time': self.priority
        }
