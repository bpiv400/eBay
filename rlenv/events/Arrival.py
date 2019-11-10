from events.Event import Event
from events.event_types import ARRIVAL


class Arrival(Event):
    """
    Event that corresponds to processing arrivals for a listing
    for some day

    Attributes:
        priority: inherited from Event
        ids: dictionary containing ids for the Event
        type: string giving the type of the event
        hidden_days: tensor giving the hidden state of the arrival days model
        turn: integer giving that the current turn is turn 1
    """
    def __init__(self, priority, sources):
        """
        Constructor

        :param priority: integer time of the event
        """
        super(Arrival, self).__init__(ARRIVAL, priority=int(priority), thread_id=None)
        self.sources = sources

