from constants import MAX_DELAY
from rlenv.env_utils import compare_input_dicts, model_str
from rlenv.env_consts import MSG, CON, DELAY, SLR_PREFIX, BYR_PREFIX


class TurnLog:
    def __init__(self, outcomes=None, turn=None, con_inputs=None, delay_inputs=None,
                 msg_inputs=None, delay_time=None):
        # outcomes
        self.outcomes = outcomes
        self.turn = turn
        # model names
        self.msg_model_name = model_str(MSG, byr=self.byr)
        self.con_model_name = model_str(CON, byr=self.byr)
        self.delay_model_name = model_str(DELAY, byr=self.byr)
        # model inputs
        self.con_inputs = con_inputs
        self.delay_inputs = delay_inputs
        self.msg_inputs = msg_inputs
        # timings
        self.delay_time = delay_time
        self.offer_time = self._init_offer_time()

    def get_con(self, check_time=None, input_dict=None):
        if self.con_inputs is None:
            raise RuntimeError("Environment unexpectedly queried concession model")
        assert check_time == self.offer_time
        compare_input_dicts()

    def get_delay(self, check_time=None, input_dict=None):
        if self.delay_inputs is None:
            raise RuntimeError("Environment unexpectedly queried delay model")
        assert check_time == self.delay_time

    @property
    def byr(self):
        return self.turn % 2 != 0

    def _init_offer_time(self):
        if self.turn % 2 == 0:
            delay_type = SLR_PREFIX
        elif self.turn == 7:
            delay_type = '{}_{}'.format(BYR_PREFIX, 7)
        else:
            delay_type = BYR_PREFIX
        delay = int(MAX_DELAY[delay_type] * self.outcomes[DELAY])
        return self.delay_time + delay
