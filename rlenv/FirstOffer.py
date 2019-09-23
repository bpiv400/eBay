"""
class for encapsulating data and methods related to a buyer offer
"""
from Event import Event
from event_types import FIRST_OFFER

class FirstOffer(Event):
    """
    Attributes:
        consts: 1d np.array of constants
        hidden: representation of buyer and seller hidden states (dictionary containing 'byr', 'slr')
        prev_slr_offer: slr offer output by SimulatorInterface.get_slr_offer (currently dict)
        prev_byr_offer: byr offer output by SimulatorInterface.get_byr_offer (currently dict)
        prev_slr_delay: integer denoting how long the previous seller action was delayed
        prev_byr_delay: integer denoting how long the previous buyer action was delayed
        delay: integer denoting how long this event has been delayed thusfar
        lstg_expiration: integer denoting when the relevant lstg expires
        priority: integer giving time inherited from Event
        ids: identifier dictionary
    """
    def __init__(self, priority, thread_id=None, byr_attr=None, bin=None):
        super(FirstOffer, self).__init__(event_type=FIRST_OFFER,
                                         priority=priority, thread_id=thread_id)
        self.bin = bin == 1  # make sure this actually works
        self.byr_attr = byr_attr
