from featnames import OUTCOME_FEATS, CON, EXP, AUTO, MSG, DELAY
from rlenv.util import model_str, need_msg
from testing.util import populate_test_model_inputs
from testing.TurnLog import TurnLog


class ThreadLog:
    def __init__(self, params=None, arrival_time=None,
                 agent=True, agent_buyer=True):
        self.arrival_time = arrival_time
        self.agent = agent
        self.agent_buyer = agent_buyer
        self.turns = dict()
        self.uncensored_turns = self.uncensored(params=params)
        for turn in range(1, self.uncensored_turns + 1):
            self.turns[turn] = self.generate_turn_log(params=params, turn=turn)
        if self.has_censored(params=params):
            censored = self.generate_censored_turn_log(params=params,
                                                       turn=self.uncensored_turns + 1)
            self.turns[self.uncensored_turns + 1] = censored

    def is_agent_turn(self, turn=None, outcomes=None):
        if self.agent:
            if self.agent_buyer:
                return turn % 2 == 1
            else:
                slr = turn % 2 == 0
                auto = outcomes[AUTO] if outcomes is not None else\
                    self.turns[turn].auto
                return slr and not auto
        else:
            return False

    def generate_censored_turn_log(self, params=None, turn=None):
        outcomes = params['x_offer'].loc[turn, :]
        outcomes = outcomes[OUTCOME_FEATS]
        agent_turn = self.is_agent_turn(turn=turn,
                                        outcomes=outcomes)
        model = model_str(DELAY, turn=turn)
        delay_inputs = populate_test_model_inputs(full_inputs=params['inputs'][model],
                                                  agent=self.agent, agent_byr=self.agent_buyer)
        delay_time = self.delay_time(turn=turn)
        return TurnLog(outcomes=outcomes, delay_inputs=delay_inputs, delay_time=delay_time, turn=turn,
                       agent=agent_turn)

    def generate_turn_log(self, params=None, turn=None):
        outcomes = params['x_offer'].loc[turn, :].squeeze()
        outcomes = outcomes[OUTCOME_FEATS]
        # print(outcomes)
        # concession inputs if necessary
        if not outcomes[AUTO] and not outcomes[EXP]:
            model = model_str(CON, turn=turn)
            con_inputs = populate_test_model_inputs(full_inputs=params['inputs'][model],
                                                    agent=self.agent, agent_byr=self.agent_buyer)
        else:
            con_inputs = None
        # msg inputs if necessary
        if not outcomes[AUTO] and not outcomes[EXP] and need_msg(outcomes[CON], slr=turn % 2 == 0):
            model = model_str(MSG, turn=turn)
            msg_inputs = populate_test_model_inputs(full_inputs=params['inputs'][model],
                                                    agent=self.agent, agent_byr=self.agent_buyer)
        else:
            msg_inputs = None
        # delay inputs if necessary
        if turn != 1 and not outcomes[AUTO]:
            model = model_str(DELAY, turn=turn)
            if outcomes[DELAY] == 0:
                # TODO fix if Etan fixes the instant buyer delay processing issue
                delay_inputs = None
                print('instant buyer offer')
            else:
                delay_inputs = populate_test_model_inputs(full_inputs=params['inputs'][model],
                                                          agent=self.agent,
                                                          agent_byr=self.agent_buyer)
        else:
            delay_inputs = None
        delay_time = self.delay_time(turn=turn)
        agent_turn = self.is_agent_turn(turn=turn,
                                        outcomes=outcomes)
        turn_log = TurnLog(outcomes=outcomes, delay_inputs=delay_inputs, con_inputs=con_inputs,
                           msg_inputs=msg_inputs, delay_time=delay_time, turn=turn,
                           agent=agent_turn)
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

    def get_agent_turns(self):
        agent_turns = dict()
        if not self.agent:
            raise RuntimeError("Querying agent turns from a thread that the agent"
                               "doesn't participate in")
        # for delay models, the agent will be queried for censored offers

        last_turn = len(self.turns)
        for i in range(1, last_turn + 1):
            # checks that turn corresponds to the current agent and offer isn't automatic
            if self.is_agent_turn(turn=i):
                # adds turn if the agent is a delay agent or the offer isn't an expiration
                agent_turns[i] = (self.turns[i])
        return agent_turns

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



