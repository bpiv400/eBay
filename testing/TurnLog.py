from constants import MONTH, MAX_DELAY_TURN, MAX_DELAY_ARRIVAL
from featnames import MSG, CON, DELAY, EXP, AUTO
from rlenv.util import model_str
from testing.util import compare_input_dicts


class TurnLog:
    def __init__(self, outcomes=None, turn=None, con_inputs=None, delay_inputs=None,
                 msg_inputs=None, delay_time=None, agent=False):
        # outcomes
        self.con = outcomes[CON]
        self.msg = outcomes[MSG]
        self.auto = outcomes[AUTO]
        self.delay = outcomes[DELAY]
        self.expired = outcomes[EXP]
        self.censored = outcomes[EXP] and outcomes[DELAY] < 1
        # turn
        self.turn = turn
        # model names
        self.msg_model_name = model_str(MSG, turn=turn)
        self.con_model_name = model_str(CON, turn=turn)
        self.delay_model_name = model_str(DELAY, turn=turn)
        # model inputs
        self.con_inputs = con_inputs
        self.delay_inputs = delay_inputs
        self.msg_inputs = msg_inputs
        # timings
        self.delay_time = delay_time
        self.offer_time = self._init_offer_time()

        self.agent = agent

    def agent_con(self):
        if self.agent:
            if self.censored:
                return 50
            elif not self.byr and self.expired:
                return 101
            else:
                return int(self.con * 100)
        else:
            raise RuntimeError("Queried concession for agent from a turn"
                               " not stored as an agent turn")

    def agent_time(self):
        return self.delay_time

    def agent_check(self, model):
        if self.agent:
            raise RuntimeError("Environment unexpectedly queried" +
                               "{} model on an agent turn".format(model))

    @property
    def is_censored(self):
        return self.censored

    def get_con(self, check_time=None, input_dict=None):
        self.agent_check('con{}'.format(self.turn))
        if self.con_inputs is None:
            raise RuntimeError("Environment unexpectedly queried concession model")
        assert check_time == self.offer_time
        compare_input_dicts(model=self.con_model_name, stored_inputs=self.con_inputs,
                            env_inputs=input_dict)
        return self.con

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
        if self.is_censored:
            return MONTH
        else:
            return int(self.offer_time - self.delay_time)

    def get_msg(self, check_time=None, input_dict=None):
        self.agent_check('msg{}'.format(self.turn))
        if self.msg_inputs is None:
            raise RuntimeError("Environment unexpectedly queried msg model")
        assert check_time == self.offer_time
        compare_input_dicts(model=self.delay_model_name, stored_inputs=self.msg_inputs,
                            env_inputs=input_dict)
        return self.msg

    @property
    def byr(self):
        return self.turn % 2 != 0

    def _init_offer_time(self):
        max_delay = MAX_DELAY_TURN if self.turn > 1 else MAX_DELAY_ARRIVAL
        delay = int(max_delay * self.delay)
        return self.delay_time + delay
