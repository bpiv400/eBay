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
        lstg_expiration: int denoting when the given lstg expires
        hidden: placeholder None
    """
    def __init__(self, priority, ids, consts=None, end_time=None):
        """
        Constructor

        :param ids: dictionary giving identifiers of the current lstg
        :param priority: integer time of the event
        :param consts: np.array containing constant features of lstg
        :param end_time: integer denoting when this lstg was sold or expired in the data
        """
        super(Arrival, self).__init__(ARRIVAL, priority=int(priority), ids=ids)
        self.consts = consts
        self.lstg_expiration = end_time
        self.hidden = None
