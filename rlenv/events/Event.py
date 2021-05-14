from rlenv.const import EXPIRATION


class Event:
    """
    Event gives information relating to an individual event.

    attributes:
        type: string giving the type of event
        priority: integer giving time of event
        ids: tuple giving identifiers for event (lstg, thread_id)
    """
    def __init__(self, event_type, priority=None):
        super(Event, self).__init__()
        # error checking
        assert(isinstance(event_type, str))
        assert(isinstance(priority, int))
        self.type = event_type
        self.priority = priority
        self.turn = 1
        self.thread_id = None

    def __lt__(self, other):
        """
        Overrides less-than comparison method to make comparisons
        based on priority alone

        If events have the same priority, any expiration takes precedence
        over any non-expiration event. Between two non-expiration events,
        lower thread_id takes precendence.
        """
        if self.priority == other.priority:
            if self.type == EXPIRATION and other.type != EXPIRATION:
                return True
            elif other.type == EXPIRATION:
                return False
            else:
                assert self.thread_id is not None
                assert other.thread_id is not None
                return self.thread_id < other.thread_id
        else:
            return self.priority < other.priority

