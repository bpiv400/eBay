from rlenv.const import ARRIVAL


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

    def __lt__(self, other):
        """
        Overrides less-than comparison method to make comparisons
        based on priority alone

        If events have the same priority, any non-arrival event takes
        priority over an arrival event.
        """
        if self.priority == other.priority:
            return self.type != ARRIVAL
        else:
            return self.priority < other.priority

