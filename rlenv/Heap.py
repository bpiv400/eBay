import heapq


class Heap:
    """
    Class for encapsulating event queue functionality.

    Attributes:
        queue: list containing events
    Public Methods:
        push: Add an event to the heap
        pop: pop the Event with the minimum priority off the heap
        empty: gives whether the heap is empty
    """
    def __init__(self, entry_type=None):
        """
        Initialize queue (empty or from seed)
        """
        self.entry_type = entry_type
        self.queue = []

    def push(self, entry):
        """
        Adds an event to the appropriate location in the heap
        (Event implements __lt__ to enable sorting as necessary)

        :param entry: instance of self.event_type
        :return: None
        """
        assert isinstance(entry, self.entry_type)
        heapq.heappush(self.queue, entry)

    def pop(self):
        """
        Pop an item off the top of the heap

        :return: instance of rlenv.Event subclass
        """
        if self.empty:
            raise RuntimeError('Cant remove from empty queue')
        else:
            return heapq.heappop(self.queue)

    def peek(self):
        return self.queue[0]

    def reset(self):
        self.queue = []

    @property
    def empty(self):
        """
        Gives whether the queue is empty

        :return: boolean
        """
        return not self.queue
