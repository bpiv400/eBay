"""
Class for encapsulating event queue functionality
"""
import heapq

# TODO Document


class EventQueue:
    """
    Attributes:
        queue: list containing events
        lstg: lstg id
    """
    def __init__(self, lstg, init=None):
        """
        Initialize queue (empty or from seed)
        """
        self.lstg = lstg
        if init is None:
            self.queue = []
        else:
            self.queue = init
            heapq.heapify(self.queue)

    def push(self, event):
        heapq.heappush(self.queue, event)

    def pop(self):
        if not self.queue:
            raise RuntimeError('Cant remove from empty queue')
        else:
            event = heapq.heappop(self.queue)
        return event
