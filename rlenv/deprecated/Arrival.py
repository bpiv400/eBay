"""
class for encapsulating data and methods related to a buyer offer
"""
from Event import Event
from event_types import ARRIVAL


class Arrival(Event):
    """
    Attributes:
        consts: np.array of constants for the lstg
        hidden: hidden state representation for arrival model(s)
        priority: int timestamp
        type: event_types.ARRIVAL
        ids: identifier dictionary for the event
        lstg_expiration: timestamp giving when the relevant lstg expires
    """
    def __init__(self, priority=None, ids=None, data=None):
        """

        :param priority: int time stamp of this event
        :param ids: id dictionary for the relevant lstg
        :param data: dictionary containing entries for consts, hidden, and lstg_expiration
        """
        super(Arrival, self).__init__(ARRIVAL, priority=priority, ids=ids)
        self.consts = data['consts']
        self.hidden = data['hidden']
        self.lstg_expiration = data['lstg_expiration']
