"""
class for encapsulating data and methods related to the first buyer offer
"""
from events.Event import Event
from events.event_types import FIRST_OFFER


class FirstOffer(Event):
    """
    Attributes:
        hidden: representation of buyer and seller hidden states (dictionary containing 'byr', 'slr')

        ids: identifier dictionary
    """
    def __init__(self, priority, thread_id=None, hist=None):
        super(FirstOffer, self).__init__(event_type=FIRST_OFFER,
                                         priority=priority, thread_id=thread_id)
        self.hist = hist
        self.turn = 1