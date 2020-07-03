class ActionLog:
    def __init__(self, con=None, input_dict=None, months=None,
                 thread_id=None, turn=None, censored=False):
        self.con = con
        self.input_dict = input_dict
        self.months = months
        self.thread_id = thread_id
        self.turn = turn
        self.censored = censored

    def __lt__(self, other):
        if self.months == other.months:
            raise RuntimeError("Multiple agent actions should not happen at the same time")
        else:
            return self.months < other.months