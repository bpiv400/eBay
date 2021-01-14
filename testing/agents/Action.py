class Action:
    def __init__(self,
                 con=None,
                 input_dict=None,
                 time=None,
                 thread_id=None,
                 turn=None,
                 censored=False):
        self.con = con
        self.input_dict = input_dict
        self.time = time
        self.thread_id = thread_id
        self.turn = turn
        self.censored = censored

    def __lt__(self, other):
        if self.time == other.time:
            if self.censored and not other.censored:
                return False
            elif not self.censored and other.censored:
                return True
            elif self.censored and other.censored:
                assert self.thread_id != other.thread_id
                return self.thread_id < other.thread_id
            else:
                raise RuntimeError("Simultaneous uncensored agents actions.")
        else:
            return self.time < other.time
