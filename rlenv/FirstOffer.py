"""
class for encapsulating data and methods related to the first buyer offer
"""
from Event import Event
from event_types import FIRST_OFFER

class FirstOffer(Event):
    """
    Attributes:
        hidden: representation of buyer and seller hidden states (dictionary containing 'byr', 'slr')

        ids: identifier dictionary
    """
    def __init__(self, priority, thread_id=None, byr_attr=None, bin=None):
        super(FirstOffer, self).__init__(event_type=FIRST_OFFER,
                                         priority=priority, thread_id=thread_id)
        self.bin = bin == 1  # make sure this actually works
        self.byr_attr = byr_attr
        self.turn = 1
