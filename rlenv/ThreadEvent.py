"""
class for encapsulating data and methods related to a buyer offer
"""
from Event import Event


class ThreadEvent(Event):
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
    def __init__(self, priority=None, ids=None, data=None, event_type=None):
        super(ThreadEvent, self).__init__(event_type, priority=priority, ids=ids)
        self.consts = data['consts']
        self.hidden = data['hidden']
        self.prev_slr_offer = data['prev_slr_offer']
        self.prev_slr_delay = data['prev_slr_delay']
        self.prev_byr_delay = data['prev_byr_delay']
        self.prev_byr_offer = data['prev_byr_offer']
        self.current_delay = data['delay']
        self.lstg_expiration = data['lstg_expiration']
