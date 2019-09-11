"""
Class that encapsulates data related to an individual event
in a slr queue
"""
import random


class Event:
    """
    Event gives information relating to an individual event

    attributes:
        type: string giving the type of event
        priority: integer giving time of event
        ids: tuple giving identifiers for event (lstg, thread_id)
    """
    def __init__(self, event_type, priority=None, ids=None):
        super(Event, self).__init__()
        # error checking
        assert(isinstance(event_type, str))
        assert(isinstance(priority, int))
        assert(isinstance(ids, dict))
        self.type = event_type
        self.priority = priority
        self.ids = ids

    def __lt__(self, other):
        """
        Overrides less-than comparison method to make comparisons
        based on priority alone

        Throws RuntimeError if both have the same lstg and same time
        """
        if self.priority == other.priority and self.ids['lstg'] == other.ids['lstg']:
            raise RuntimeError("Two events for one lstg executed at same time")
        elif self.priority == other.priority:
            return random.randint(1, 2) == 1
        return self.priority < other.priority

