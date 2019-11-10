"""
Class for encapsulating data and methods related to all offers and delays after
the initial buyer offer
"""
from events.Event import Event


class ThreadEvent(Event):
    """
    Attributes:
        hidden: dictionary containing hidden states for all buyer and seller decision interface
        sources: dictionary containing all input maps generated during previous turns
        byr: boolean giving whether this is a buyer turn
        thread_id: integer giving the thread id
        priority: integer giving the time of the event
        turn: gives the turn number
    """
    def __init__(self, priority, event_type=None, thread_id=None, hidden=None, sources=None, byr=False, turn=0):
        """
        Constructor
        :param priority: integer giving the time of the event
        :param thread_id: integer giving the thread id
        :param hidden: dictionary containing hidden states for all buyer and seller decision interface
        :param: turn: gives the turn number
        :param sources: dictionary containing all input maps generated during previous turns
        """
        super(ThreadEvent, self).__init__(event_type=event_type, priority=priority, thread_id=thread_id)
        self.sources = sources
        self.hidden = hidden
        self.byr = byr
        self.turn = turn
