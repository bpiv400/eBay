"""
Class for encapsulating event queue functionality
"""
import heapq

# TODO Document


class EventQueue:
    """
    Attributes:
        queue: list containing events
    Public Methods:
        push: Add an event to the heap
        pop: pop the Event with the minimum priority off the heap
        empty: gives whether the heap is empty
    """
    def __init__(self):
        """
        Initialize queue (empty or from seed)
        """
        self.queue = []

    def push(self, event):
        """
        Adds an event to the appropriate location in the heap
        (Event implements __lt__ to enable sorting as necessary)

        :param event: instance of rlenv.Event
        :return: None
        """
        heapq.heappush(self.queue, event)

    def pop(self):
        """
        Pop an item off the top of the heap

        :return: instance of rlenv.Event subclass
        """
        if self.empty:
            raise RuntimeError('Cant remove from empty queue')
        else:
            event = heapq.heappop(self.queue)
        return event

    def reset(self):
        self.queue = []

    @property
    def empty(self):
        """
        Gives whether the queue is empty

        :return: boolean
        """
        return not self.queue
