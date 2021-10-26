from env.util import model_str
from testing.util import compare_delay_inputs
from constants import MAX_DELAY_TURN
from featnames import DELAY


class CensoredTurn:
    def __init__(self,
                 turn=None,
                 delay_inputs=None,
                 delay_time=None,
                 agent=False):

        # copy parameters to self
        self.turn = turn
        self.agent = agent
        self.delay_time = delay_time
        self.delay_inputs = delay_inputs
        self.auto = False
        self.expired = True

        # model name
        self.delay_model_name = model_str(DELAY, turn=turn)

    @property
    def is_censored(self):
        return True

    def get_delay(self, check_time=None, input_dict=None):
        compare_delay_inputs(turn=self, check_time=check_time, input_dict=input_dict)
        return MAX_DELAY_TURN

    @property
    def byr(self):
        return self.turn % 2 != 0
