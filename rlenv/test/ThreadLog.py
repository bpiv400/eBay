class ThreadLog:
    def __init__(self, params=None):
        self.turns = dict()
        uncensored_turns = self.uncensored(params)
        for turn in range(1, uncensored_turns + 1):
            self.turns[turn] = self.generate_turn_log(params=params, turn=turn)

    def generate_turn_log(self, params=None, turn=None):
        #TODO: Generate a turn log for the current turn
        pass

    def get_con(self, event=None, x_lstg=None):
        con = self.turns[event.turn].get_con(event=event, x_lstg=x_lstg)

    def get_msg(self, event=None, x_lstg=None):
        msg = self.turns[event.turn].get_msg(event=event, x_lstg=x_lstg)

    def get_delay(self, event=None, x_lstg=None):
        delay =



