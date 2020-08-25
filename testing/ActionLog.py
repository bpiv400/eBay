class ActionLog:
    def __init__(self, con=None, input_dict=None, days=None,
                 thread_id=None, turn=None, censored=False):
        self.con = con
        self.input_dict = input_dict
        self.days = days
        self.thread_id = thread_id
        self.turn = turn
        self.censored = censored

    def __lt__(self, other):
        if self.days == other.days:
            raise RuntimeError("Multiple agent actions should not happen at the same time")
        else:
            return self.days < other.days
