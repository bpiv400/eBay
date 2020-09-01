from rlenv.util import model_str, need_msg
from testing.util import populate_inputs
from testing.Turn import Turn
from testing.CensoredTurn import CensoredTurn
from featnames import OUTCOME_FEATS, CON, EXP, AUTO, MSG, DELAY, X_OFFER


class Thread:
    def __init__(self,
                 params=None,
                 arrival_time=None,
                 agent=False,
                 agent_buyer=False):
        self.arrival_time = arrival_time
        self.agent = agent
        self.agent_buyer = agent_buyer
        self.turns = dict()
        self.offers = params[X_OFFER]

        num_offers = len(self.offers.index)
        for turn in range(1, num_offers + 1):
            self.turns[turn] = self.generate_turn(params=params,
                                                  turn=turn)
        if self.has_censored():
            turn = num_offers + 1
            self.turns[turn] = self.generate_censored_turn(params=params,
                                                           turn=turn)

    def is_agent_turn(self, turn=None, outcomes=None):
        if self.agent:
            if self.agent_buyer:
                return turn % 2 == 1
            else:
                if turn % 2 == 0:
                    if outcomes is None:
                        return not self.turns[turn].auto
                    else:
                        return not outcomes[AUTO]
        else:
            return False

    def generate_censored_turn(self, params=None, turn=None):
        agent_turn = self.is_agent_turn(turn=turn)
        model = model_str(DELAY, turn=turn)
        full_inputs = params['inputs'][model]
        delay_inputs = populate_inputs(full_inputs=full_inputs,
                                       agent=self.agent,
                                       agent_byr=self.agent_buyer)
        delay_time = self.delay_time(turn=turn)
        return CensoredTurn(delay_inputs=delay_inputs,
                            delay_time=delay_time,
                            turn=turn,
                            agent=agent_turn)

    def generate_turn(self, params=None, turn=None):
        outcomes = params[X_OFFER].loc[turn, :].squeeze()
        outcomes = outcomes[OUTCOME_FEATS]
        # concession inputs if necessary
        if not outcomes[AUTO] and not outcomes[EXP]:
            model = model_str(CON, turn=turn)
            con_inputs = populate_inputs(full_inputs=params['inputs'][model],
                                         agent=self.agent,
                                         agent_byr=self.agent_buyer)
        else:
            con_inputs = None
        # msg inputs if necessary
        if not outcomes[AUTO] and not outcomes[EXP] and need_msg(outcomes[CON], slr=turn % 2 == 0):
            model = model_str(MSG, turn=turn)
            full_inputs = params['inputs'][model]
            msg_inputs = populate_inputs(full_inputs=full_inputs,
                                         agent=self.agent,
                                         agent_byr=self.agent_buyer)
        else:
            msg_inputs = None
        # delay inputs if necessary
        if turn != 1 and not outcomes[AUTO]:
            model = model_str(DELAY, turn=turn)
            full_inputs = params['inputs'][model]
            delay_inputs = populate_inputs(full_inputs=full_inputs,
                                           agent=self.agent,
                                           agent_byr=self.agent_buyer)
        else:
            delay_inputs = None
        delay_time = self.delay_time(turn=turn)
        agent_turn = self.is_agent_turn(turn=turn, outcomes=outcomes)
        return Turn(outcomes=outcomes,
                    delay_inputs=delay_inputs,
                    con_inputs=con_inputs,
                    msg_inputs=msg_inputs,
                    delay_time=delay_time,
                    turn=turn,
                    agent=agent_turn)

    def delay_time(self, turn=None):
        # delay time
        if turn == 1:
            delay_time = self.arrival_time
        else:
            delay_time = self.turns[turn - 1].offer_time
        return delay_time

    def has_censored(self):
        last_turn = self.offers.index.max()
        last_con = self.offers.loc[last_turn, CON]
        print('{}: {}'.format(last_turn, last_con))
        if last_con == 1:
            return False
        elif last_con == 0 and last_turn % 2 == 1:
            return False
        else:
            return True

    def get_agent_turns(self):
        agent_turns = dict()
        if not self.agent:
            raise RuntimeError("Querying agents turns from a thread that the agents"
                               "doesn't participate in")
        # for delay models, the agents will be queried for censored offers

        last_turn = len(self.turns)
        for i in range(1, last_turn + 1):
            # checks that turn corresponds to the current agents and offer isn't automatic
            if self.is_agent_turn(turn=i) and self.turns[i].delay < 1.:
                # adds turn if an expiration does not occur on agents's turn
                agent_turns[i] = (self.turns[i])
        return agent_turns

    def get_con(self, time=None, turn=None, input_dict=None):
        return self.turns[turn].get_con(check_time=time,
                                        input_dict=input_dict)

    def get_msg(self, time=None, turn=None, input_dict=None):
        return self.turns[turn].get_msg(check_time=time,
                                        input_dict=input_dict)

    def get_delay(self, time=None, turn=None, input_dict=None):
        if turn not in self.turns:
            print("Environment unexpectedly queried delay model.")
            input("Exiting listing. Press Enter to continue...")
            return None
        return self.turns[turn].get_delay(check_time=time,
                                          input_dict=input_dict)
