from featnames import OUTCOME_FEATS, CON, EXP, AUTO, MSG, DELAY
from rlenv.env_utils import populate_test_model_inputs, model_str, need_msg
from rlenv.test.TurnLog import TurnLog


class ThreadLog:
    def __init__(self, params=None, arrival_time=None):
        self.arrival_time = arrival_time
        self.turns = dict()
        uncensored_turns = self.uncensored(params=params)
        for turn in range(1, uncensored_turns + 1):
            self.turns[turn] = self.generate_turn_log(params=params, turn=turn)
        if self.has_censored(params=params):
            censored = self.generate_censored_turn_log(params=params, turn=uncensored_turns + 1)
            self.turns[uncensored_turns + 1] = censored

    def generate_censored_turn_log(self, params=None, turn=None):
        print('Turn: {}'.format(turn))
        outcomes = params['x_offer'].loc[turn, :]
        outcomes = outcomes[OUTCOME_FEATS]
        byr = turn % 2 != 0
        model = model_str(DELAY, byr=byr)
        delay_inputs = populate_test_model_inputs(full_inputs=params['inputs'][model], value=turn)
        delay_time = self.delay_time(turn=turn)
        return TurnLog(outcomes=outcomes, delay_inputs=delay_inputs, delay_time=delay_time, turn=turn)

    def generate_turn_log(self, params=None, turn=None):
        outcomes = params['x_offer'].loc[turn, :].squeeze()
        outcomes = outcomes[OUTCOME_FEATS]
        byr = turn % 2 != 0
        # print(outcomes)
        # concession inputs if necessary
        if not outcomes[AUTO] and not outcomes[EXP]:
            model = model_str(CON, byr=byr)
            con_inputs = populate_test_model_inputs(full_inputs=params['inputs'][model], value=turn)
        else:
            con_inputs = None
        # msg inputs if necessary
        if not outcomes[AUTO] and not outcomes[EXP] and need_msg(outcomes[CON], slr=not byr):
            model = model_str(MSG, byr=byr)
            msg_inputs = populate_test_model_inputs(full_inputs=params['inputs'][model], value=turn)
        else:
            msg_inputs = None
        # delay inputs if necessary
        if turn != 1 and not outcomes[AUTO]:
            model = model_str(DELAY, byr=byr)
            delay_inputs = populate_test_model_inputs(full_inputs=params['inputs'][model], value=turn)
        else:
            delay_inputs = None
        delay_time = self.delay_time(turn=turn)
        turn_log = TurnLog(outcomes=outcomes, delay_inputs=delay_inputs, con_inputs=con_inputs,
                           msg_inputs=msg_inputs, delay_time=delay_time, turn=turn)
        print('Turn: {} at time: {}'.format(turn, turn_log.offer_time))
        return turn_log

    def delay_time(self, turn=None):
        # delay time
        if turn == 1:
            delay_time = self.arrival_time
        else:
            delay_time = self.turns[turn - 1].offer_time
        return delay_time

    @staticmethod
    def has_censored(params=None):
        num_offers = len(params['x_offer'].index)
        last_censored = params['x_offer'].loc[num_offers, :].squeeze()
        last_censored = last_censored['censored']
        return last_censored

    @staticmethod
    def uncensored(params=None):
        num_offers = len(params['x_offer'].index)
        last_censored = ThreadLog.has_censored(params=params)
        if not last_censored:
            return num_offers
        else:
            if num_offers == 1:
                raise RuntimeError("Initial buyer offer should never be censored")
            return num_offers - 1

    def get_con(self, time=None, turn=None, input_dict=None):
        return self.turns[turn].get_con(check_time=time, input_dict=input_dict)

    def get_msg(self, time=None, turn=None, input_dict=None):
        return self.turns[turn].get_msg(check_time=time, input_dict=input_dict)

    def get_delay(self, time=None, turn=None, input_dict=None):
        if turn not in self.turns:
            print("Environment unexpectedly queried delay model.")
            input("Exiting listing. Press Enter to continue...")
            return None
        return self.turns[turn].get_delay(check_time=time, input_dict=input_dict)



