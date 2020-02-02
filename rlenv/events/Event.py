from rlenv.env_consts import ARRIVAL


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

    def __lt__(self, other):
        """
        Overrides less-than comparison method to make comparisons
        based on priority alone

        Throws RuntimeError if both have the same lstg and same time
        """
        if self.priority == other.priority:
            if self.type == ARRIVAL:
                return True
            else:
                return False
        else:
            return self.priority < other.priority

