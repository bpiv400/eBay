from Event import Event
from event_types import NEW_ITEM

WEEK = 7 * 24 * 60 * 60


class NewItem(Event):
    """
    Event that corresponds to drawing a new item which has never
    been bid on before

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
        super(NewItem, self).__init__(NEW_ITEM, priority=int(priority), ids=ids)
        self.consts = consts
        self.lstg_expiration = end_time + WEEK
        self.hidden = None
