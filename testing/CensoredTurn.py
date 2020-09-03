from rlenv.util import model_str
from testing.util import compare_input_dicts
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
        if self.delay_inputs is None:
            print("Environment unexpectedly queried delay model.")
            input("Exiting listing. Press Enter to continue...")
            return None
        if check_time != self.delay_time:
            print('-- INCONSISTENCY IN delay time --')
            print('stored value = {} | env value = {}'.format(
                self.delay_time, check_time))
            input("Exiting listing. Press Enter to continue...")
            return None
        compare_input_dicts(model=self.delay_model_name,
                            stored_inputs=self.delay_inputs,
                            env_inputs=input_dict)
        return MAX_DELAY_TURN

    @property
    def byr(self):
        return self.turn % 2 != 0
