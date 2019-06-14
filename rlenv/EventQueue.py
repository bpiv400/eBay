"""
Class for encapsulating event queue functionality
"""
import heapq

class EventQueue:
    """
    Attributes:
        queue: list containing events
        interval: integer specifying what 1 clock tick corresponds to
    """
    def __init__(self, interval=None, init=None):
        """
        Initialize queue (empty or from seed)
        """
        assert isinstance(interval, int)
        if init is None:
            self.queue = []
        else:
            self.queue = init
            for event in self.queue:
                event.priority = self.round(event.priority)
            heapq.heapify(self.queue)

        self.interval = interval

    def round(self, priority):
        """
        Rounds given priority to nearest interval if necessary

        :param priority: integer containing given priority
        :return: integer priority
        """
        if priority % self.interval != 0:
            priority = self.interval * round(priority / self.interval)
        return priority

    def push(self, event):
        event.priority = self.round(event.priority)

        heapq.heappush(self.queue, event)

    def pop(self):
        if not self.queue:
            raise RuntimeError('Cant remove from empty queue')
        else:
            event = heapq.heappop(self.queue)
        return event
