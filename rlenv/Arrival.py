from Event import Event
from event_types import ARRIVAL


class Arrival(Event):
    """
    Event that corresponds to processing arrivals for a listing
    for some day

    Attributes:
        priority: inherited from Event
        ids: dictionary containing ids for the Event
        type: string giving the type of the event
        hidden_days: tensor giving the hidden state of the arrival days model
    """
    def __init__(self, priority, ids, hidden_days=None):
        """
        Constructor

        :param ids: dictionary giving identifiers of the current lstg
        :param priority: integer time of the event
        """
        super(Arrival, self).__init__(ARRIVAL, priority=int(priority), ids=ids)
        self.hidden_days = hidden_days
