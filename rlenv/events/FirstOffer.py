"""
class for encapsulating data and methods related to the first buyer offer
"""
from events.Event import Event
from events.event_types import FIRST_OFFER
from constants import SLR_PREFIX, BYR_PREFIX

class FirstOffer(Event):
    """
    Attributes:
        hidden: representation of buyer and seller hidden states (dictionary containing 'byr', 'slr')

        ids: identifier dictionary
    """
    def __init__(self, priority, thread_id=None, buyer=None, seller=None):
        super(FirstOffer, self).__init__(event_type=FIRST_OFFER,
                                         priority=priority, thread_id=thread_id)
        self.turn = 1
        self.buyer = buyer
        self.seller = seller
        self.sources = None

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

    def is_sale(self):
        return self.sources.is_sale()

    def is_rej(self):
        return self.sources.is_rej()


